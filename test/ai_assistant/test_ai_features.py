import json
import requests
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from ai_assistant.capabilities import collect_local_context
from ai_assistant.config import AIConfigStore, AIProfile, SCHEMA_VERSION, validate_profile
from ai_assistant.conversations import ConversationStore, MAX_CONTENT_CHARACTERS
from ai_assistant.markdown_render import render_markdown_fragment
from ai_assistant.model_catalog import ModelCatalogClient, ModelOperationCancelled
from ai_assistant.providers import PRESETS, ProviderConfig, validate_config
from ai_assistant.web_search import (
    BAIDU_ENDPOINT,
    BING_ENDPOINT,
    BUILTIN_ENGINES,
    DUCKDUCKGO_ENDPOINT,
    GOOGLE_ENDPOINT,
    SEARCH_ENGINES,
    SHENMA_ENDPOINT,
    SO360_ENDPOINT,
    SOGOU_ENDPOINT,
    SearchHumanVerificationRequired,
    SearchResult,
    WebSearchClient,
    _normalize_results,
    validate_search_settings,
)
from ai_assistant import ChatMessage
from score_statistics import calculate_score_statistics


FIXTURE_DIR = Path(__file__).with_name("fixtures")


class FakeResponse:
    def __init__(
        self,
        payload=None,
        *,
        status=200,
        text="",
        url="https://example.test/",
        headers=None,
    ):
        self.payload = payload
        self.status_code = status
        self.text = text
        self.content = text.encode() if text else json.dumps(payload or {}).encode()
        self.url = url
        self.closed = False
        self.encoding = "utf-8"
        self.headers = dict(headers or {})

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def iter_lines(self, **_kwargs):
        for row in self.payload or []:
            yield json.dumps(row).encode()

    def iter_content(self, **_kwargs):
        yield self.content

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, get_response=None, post_response=None, head_response=None):
        self.get_response = get_response
        self.post_response = post_response
        self.head_response = head_response
        self.calls = []

    @staticmethod
    def _next(response):
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._next(self.get_response)

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", url, kwargs))
        return self._next(self.head_response)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.post_response


class IndependentProtocolConfigTest(unittest.TestCase):
    def test_provider_preset_is_only_a_quick_configuration(self):
        profile = replace(
            AIProfile.default(),
            protocol="anthropic",
            base_url="https://gateway.example/v1",
            model="custom-claude",
        )
        checked = validate_profile(profile)
        self.assertEqual(checked.preset_id, "deepseek")
        self.assertEqual(checked.protocol, "anthropic")

    def test_v1_profile_migrates_to_default_off_capabilities_and_current_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            profile = AIProfile.default()
            legacy = {key: value for key, value in profile.__dict__.items() if key not in {
                "capability_ids", "search_engine", "search_endpoint", "search_result_limit"
            }}
            path.write_text(json.dumps({"version": 1, "profiles": [legacy]}), encoding="utf-8")
            store = AIConfigStore(path, keyring_backend=object())
            loaded = store.load_profiles()[0]
            self.assertEqual(loaded.capability_ids, ())
            store.save_profiles([loaded])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], SCHEMA_VERSION)

    def test_v2_search_engines_migrate_without_changing_network_route(self):
        with tempfile.TemporaryDirectory() as directory:
            for legacy_engine, expected_engine, expected_endpoint in (
                ("duckduckgo", "duckduckgo", ""),
                ("bing", "searxng", "https://search.example/"),
                ("baidu", "searxng", "https://search.example/"),
                ("google", "searxng", "https://search.example/"),
                ("searxng", "searxng", "https://search.example/"),
            ):
                with self.subTest(engine=legacy_engine):
                    path = Path(directory) / f"{legacy_engine}.json"
                    legacy = replace(
                        AIProfile.default(),
                        search_engine=legacy_engine,
                        search_endpoint="https://search.example/",
                    )
                    path.write_text(
                        json.dumps({"version": 2, "profiles": [legacy.__dict__]}),
                        encoding="utf-8",
                    )

                    loaded = AIConfigStore(path, keyring_backend=object()).load_profiles()[0]

                    self.assertEqual(loaded.search_engine, expected_engine)
                    self.assertEqual(loaded.search_endpoint, expected_endpoint)

    def test_current_search_profiles_round_trip_builtin_and_self_hosted_modes(self):
        self.assertEqual(SCHEMA_VERSION, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            profiles = [
                replace(AIProfile.default(), id="auto", search_engine="auto"),
                replace(
                    AIProfile.default(),
                    id="bing",
                    search_engine="bing",
                    search_endpoint="https://ignored.example",
                ),
                replace(
                    AIProfile.default(),
                    id="searxng",
                    search_engine="searxng",
                    search_endpoint="https://search.example",
                ),
            ]
            store = AIConfigStore(path, keyring_backend=object())
            store.save_profiles(profiles)

            loaded = {one.id: one for one in store.load_profiles()}

        self.assertEqual(loaded["auto"].search_endpoint, "")
        self.assertEqual(loaded["bing"].search_endpoint, "")
        self.assertEqual(loaded["searxng"].search_endpoint, "https://search.example/")

    def test_retired_deepseek_models_are_migrated_and_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            legacy = replace(AIProfile.default(), model="deepseek-chat")
            path.write_text(
                json.dumps({"version": SCHEMA_VERSION, "profiles": [legacy.__dict__]}),
                encoding="utf-8",
            )
            loaded = AIConfigStore(path, keyring_backend=object()).load_profiles()[0]

        self.assertEqual(loaded.model, "deepseek-v4-flash")
        self.assertTrue(all(preset.default_model != "deepseek-chat" for preset in PRESETS))
        with self.assertRaisesRegex(ValueError, "已停用"):
            validate_config(ProviderConfig(
                "openai",
                "https://api.deepseek.com/v1",
                "deepseek-chat",
                "secret",
            ))


class ModelCatalogTest(unittest.TestCase):
    def test_lists_ollama_models_and_uses_official_tags_endpoint(self):
        response = FakeResponse({"models": [{"name": "qwen3:8b"}, {"model": "gemma3"}]})
        session = FakeSession(get_response=response)
        models = ModelCatalogClient(session).list_models(
            ProviderConfig("ollama", "http://127.0.0.1:11434/api", "unused")
        )
        self.assertEqual(models, ["gemma3", "qwen3:8b"])
        self.assertEqual(session.calls[0][1], "http://127.0.0.1:11434/api/tags")

    def test_pull_reports_progress_closes_response_and_can_cancel(self):
        response = FakeResponse([
            {"status": "pulling", "completed": 5, "total": 10},
            {"status": "success", "completed": 10, "total": 10},
        ])
        session = FakeSession(post_response=response)
        progress = []
        ModelCatalogClient(session).pull_ollama_model(
            ProviderConfig("ollama", "http://127.0.0.1:11434/api", "qwen3:8b"),
            "qwen3:8b",
            progress=progress.append,
        )
        self.assertEqual([one.percent for one in progress], [50, 100])
        self.assertTrue(response.closed)
        self.assertEqual(session.calls[0][2]["json"], {"model": "qwen3:8b", "stream": True})

        cancel = threading.Event()
        cancel.set()
        canceled_response = FakeResponse([{"status": "pulling"}])
        with self.assertRaises(ModelOperationCancelled):
            ModelCatalogClient(FakeSession(post_response=canceled_response)).pull_ollama_model(
                ProviderConfig("ollama", "http://localhost:11434/api", "qwen3:8b"),
                "qwen3:8b",
                cancel_event=cancel,
            )
        self.assertTrue(canceled_response.closed)

    def test_remote_protocol_cannot_download(self):
        with self.assertRaisesRegex(ValueError, "Ollama"):
            ModelCatalogClient(FakeSession()).pull_ollama_model(
                ProviderConfig("openai", "https://example.test/v1", "model", "key"),
                "model",
            )


class WebSearchTest(unittest.TestCase):
    def test_search_catalog_has_approved_ids_labels_and_builtin_boundary(self):
        self.assertEqual(
            SEARCH_ENGINES,
            (
                ("auto", "自动（直连推荐）"),
                ("baidu", "百度（直连）"),
                ("bing", "Bing（直连）"),
                ("google", "Google（需要代理）"),
                ("sogou", "搜狗（直连）"),
                ("so360", "360 搜索（直连）"),
                ("shenma", "神马（直连）"),
                ("duckduckgo", "DuckDuckGo（需要代理）"),
                ("searxng", "SearXNG（自托管）"),
            ),
        )
        self.assertEqual(
            BUILTIN_ENGINES,
            {"auto", "baidu", "bing", "google", "sogou", "so360", "shenma", "duckduckgo"},
        )
        for engine, _label in SEARCH_ENGINES[:-1]:
            self.assertEqual(
                validate_search_settings(engine, "https://ignored.example"),
                (engine, ""),
            )

    def test_custom_searxng_is_bounded_and_parsed(self):
        response = FakeResponse(
            {"results": [
                {"title": "Result", "url": "https://source.test/a", "content": "Snippet"},
                {"title": "Bad", "url": "javascript:alert(1)", "content": "x"},
            ]},
            url="https://search.example/search?q=x&format=json",
        )
        session = FakeSession(get_response=response)
        results = WebSearchClient(session).search(
            "x", engine="searxng", endpoint="https://search.example", limit=5
        )
        self.assertEqual([(one.title, one.url) for one in results], [("Result", "https://source.test/a")])
        self.assertEqual(session.calls[0][1], "https://search.example/search")
        self.assertFalse(session.calls[0][2]["allow_redirects"])
        self.assertTrue(session.calls[0][2]["stream"])
        self.assertTrue(response.closed)

    def test_rejects_credentials_insecure_remote_and_cross_host_redirect(self):
        invalid = [
            "http://remote.example/search",
            "https://user:secret@search.example/search",
            "https://search.example/search?q=preset",
        ]
        for endpoint in invalid:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_search_settings("searxng", endpoint)
        response = FakeResponse({"results": []}, url="https://evil.example/search")
        with self.assertRaisesRegex(RuntimeError, "未配置"):
            WebSearchClient(FakeSession(get_response=response)).search(
                "x", engine="searxng", endpoint="https://search.example"
            )

    def test_timeout_redirect_and_oversized_stream_are_safe(self):
        with self.assertRaisesRegex(RuntimeError, "超时"):
            WebSearchClient(FakeSession(get_response=requests.Timeout())).search(
                "x", engine="searxng", endpoint="https://search.example"
            )
        redirect = FakeResponse({}, status=302, url="https://search.example/search")
        with self.assertRaisesRegex(RuntimeError, "重定向"):
            WebSearchClient(FakeSession(get_response=redirect)).search(
                "x", engine="searxng", endpoint="https://search.example"
            )
        oversized = FakeResponse({}, url="https://search.example/search")
        oversized.content = b"x" * (2 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(RuntimeError, "2 MiB"):
            WebSearchClient(FakeSession(get_response=oversized)).search(
                "x", engine="searxng", endpoint="https://search.example"
            )
        self.assertTrue(oversized.closed)

    def test_result_filter_rejects_unsafe_duplicates_and_empty_titles(self):
        rows = [
            ("ok", "https://source.test/a", "one"),
            ("duplicate", "https://source.test/a", "two"),
            ("", "https://source.test/b", "empty"),
            ("bad", "https://user:secret@source.test/c", "credentials"),
            ("bad", "http:///missing-host", "host"),
            ("bad", "javascript:alert(1)", "scheme"),
        ]
        self.assertEqual(
            _normalize_results(rows),
            [SearchResult("ok", "https://source.test/a", "one")],
        )

    def test_bing_and_duckduckgo_distinguish_empty_from_unknown_structure(self):
        explicit_empty = (
            (
                "bing",
                BING_ENDPOINT,
                '<html><ol id="b_results"><li class="b_no">没有与此相关的结果</li></ol></html>',
            ),
            (
                "duckduckgo",
                DUCKDUCKGO_ENDPOINT,
                '<html><div class="no-results">No results.</div></html>',
            ),
        )
        for engine, endpoint, content in explicit_empty:
            with self.subTest(engine=engine):
                response = FakeResponse(text=content, url=endpoint)
                self.assertEqual(
                    WebSearchClient(FakeSession(get_response=response)).search(
                        "x", engine=engine
                    ),
                    [],
                )
                self.assertTrue(response.closed)

        for engine, endpoint in (
            ("bing", BING_ENDPOINT),
            ("duckduckgo", DUCKDUCKGO_ENDPOINT),
        ):
            with self.subTest(engine=f"{engine}-unknown"):
                response = FakeResponse(
                    text="<html><body>changed layout</body></html>",
                    url=endpoint,
                )
                with self.assertRaisesRegex(RuntimeError, "页面结构无法解析"):
                    WebSearchClient(FakeSession(get_response=response)).search(
                        "x", engine=engine
                    )
                self.assertTrue(response.closed)

    def test_baidu_resolves_only_selected_results_with_safe_head_requests(self):
        page = FakeResponse(
            text=(FIXTURE_DIR / "baidu_organic_result.html").read_text(encoding="utf-8"),
            url=f"{BAIDU_ENDPOINT}?wd=xjtu",
        )
        resolved = FakeResponse(
            status=302,
            url="https://www.baidu.com/link?url=fixture-token",
            headers={"Location": "https://www.xjtu.edu.cn/"},
        )
        session = FakeSession(get_response=page, head_response=resolved)

        results = WebSearchClient(session).search("xjtu", engine="baidu", limit=1)

        self.assertEqual(
            results,
            [
                SearchResult(
                    "西安交通大学",
                    "https://www.xjtu.edu.cn/",
                    "西安交通大学是教育部直属重点大学。",
                )
            ],
        )
        self.assertEqual(session.calls[0][1], BAIDU_ENDPOINT)
        self.assertEqual(session.calls[0][2]["params"], {"wd": "xjtu"})
        method, url, kwargs = session.calls[1]
        self.assertEqual((method, url), ("HEAD", "https://www.baidu.com/link?url=fixture-token"))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (10, 20))
        self.assertTrue(page.closed)
        self.assertTrue(resolved.closed)

    def test_baidu_drops_unsafe_and_failed_redirect_targets_with_limit_bound(self):
        links = "".join(
            '<div class="c-container"><h3><a href="https://www.baidu.com/link?url=%d">R%d</a></h3></div>'
            % (index, index)
            for index in range(6)
        )
        page = FakeResponse(
            text=f"<html><body>{links}</body></html>",
            url=f"{BAIDU_ENDPOINT}?wd=x",
        )
        javascript = FakeResponse(
            status=302,
            headers={"Location": "javascript:alert(1)"},
        )
        credentials = FakeResponse(
            status=302,
            headers={"Location": "https://user:secret@source.test/"},
        )
        missing = FakeResponse(status=302)
        wrong_status = FakeResponse(
            status=200,
            headers={"Location": "https://source.test/ignored"},
        )
        session = FakeSession(
            get_response=page,
            head_response=[
                javascript,
                credentials,
                missing,
                wrong_status,
                requests.Timeout(),
            ],
        )

        self.assertEqual(
            WebSearchClient(session).search("x", engine="baidu", limit=5),
            [],
        )
        self.assertEqual(
            [call[0] for call in session.calls],
            ["GET", "HEAD", "HEAD", "HEAD", "HEAD", "HEAD"],
        )
        self.assertTrue(
            all(
                response.closed
                for response in (page, javascript, credentials, missing, wrong_status)
            )
        )

    def test_baidu_distinguishes_challenge_empty_and_unknown_structure(self):
        cases = (
            (
                '<html><div id="verify-form">安全验证</div></html>',
                SearchHumanVerificationRequired,
                "人机验证",
            ),
            (
                '<html><div class="op_sp_realtime_n_result">没有找到相关结果</div></html>',
                None,
                "",
            ),
            ("<html><body>changed layout</body></html>", RuntimeError, "页面结构无法解析"),
        )
        for content, exception_type, message in cases:
            with self.subTest(message=message or "empty"):
                response = FakeResponse(text=content, url=BAIDU_ENDPOINT)
                client = WebSearchClient(FakeSession(get_response=response))
                if exception_type is None:
                    self.assertEqual(client.search("x", engine="baidu"), [])
                else:
                    with self.assertRaisesRegex(exception_type, message):
                        client.search("x", engine="baidu")
                self.assertTrue(response.closed)

    def test_so360_uses_data_mdurl_without_requesting_tracking_link(self):
        response = FakeResponse(
            text=(FIXTURE_DIR / "so360_organic_result.html").read_text(encoding="utf-8"),
            url=f"{SO360_ENDPOINT}?q=xjtu",
        )
        session = FakeSession(get_response=response)

        results = WebSearchClient(session).search("xjtu", engine="so360", limit=3)

        self.assertEqual(
            results,
            [
                SearchResult(
                    "西安交通大学新闻网",
                    "http://news.xjtu.edu.cn/",
                    "西安交通大学新闻门户。",
                )
            ],
        )
        self.assertEqual([call[1] for call in session.calls], [SO360_ENDPOINT])
        self.assertEqual(session.calls[0][2]["params"], {"q": "xjtu"})

    def test_sogou_decodes_url_parameter_without_requesting_tracking_link(self):
        response = FakeResponse(
            text=(FIXTURE_DIR / "sogou_organic_result.html").read_text(encoding="utf-8"),
            url=f"{SOGOU_ENDPOINT}?keyword=xjtu",
        )
        session = FakeSession(get_response=response)

        results = WebSearchClient(session).search("xjtu", engine="sogou", limit=3)

        self.assertEqual(
            results,
            [
                SearchResult(
                    "Welcome to Xi'an Jiaotong University!",
                    "http://men.xjtu.edu.cn/",
                    "XJTU teaching and research news.",
                )
            ],
        )
        self.assertEqual([call[1] for call in session.calls], [SOGOU_ENDPOINT])
        self.assertEqual(session.calls[0][2]["params"], {"keyword": "xjtu"})

    def test_so360_and_sogou_drop_opaque_or_unsafe_tracking_targets(self):
        cases = (
            (
                "so360",
                SO360_ENDPOINT,
                '<li class="res-list"><h3><a href="https://www.so.com/link?m=opaque">Opaque</a></h3></li>',
            ),
            (
                "so360",
                SO360_ENDPOINT,
                '<li class="res-list"><h3><a href="https://www.so.com/link?m=x" data-mdurl="javascript:alert(1)">Unsafe</a></h3></li>',
            ),
            (
                "sogou",
                SOGOU_ENDPOINT,
                '<div class="vrResult"><h3><a class="resultLink" href="./tc?url=https%3A%2F%2Fuser%3Asecret%40source.test%2F">Unsafe</a></h3></div>',
            ),
        )
        for engine, endpoint, content in cases:
            with self.subTest(engine=engine, content=content):
                response = FakeResponse(text=content, url=endpoint)
                with self.assertRaisesRegex(RuntimeError, "页面结构无法解析"):
                    WebSearchClient(FakeSession(get_response=response)).search(
                        "x", engine=engine
                    )

    def test_so360_and_sogou_distinguish_challenge_empty_and_unknown_structure(self):
        cases = (
            ("so360", SO360_ENDPOINT, '<div id="verify">captcha</div>', SearchHumanVerificationRequired),
            ("so360", SO360_ENDPOINT, '<div class="no-result">没有找到相关结果</div>', None),
            ("so360", SO360_ENDPOINT, '<div>changed layout</div>', RuntimeError),
            ("sogou", SOGOU_ENDPOINT, '<div class="verify">captcha</div>', SearchHumanVerificationRequired),
            ("sogou", SOGOU_ENDPOINT, '<div class="no-result">未找到相关结果</div>', None),
            ("sogou", SOGOU_ENDPOINT, '<div>changed layout</div>', RuntimeError),
        )
        for engine, endpoint, content, exception_type in cases:
            with self.subTest(engine=engine, exception=exception_type):
                response = FakeResponse(text=content, url=endpoint)
                client = WebSearchClient(FakeSession(get_response=response))
                if exception_type is None:
                    self.assertEqual(client.search("x", engine=engine), [])
                else:
                    with self.assertRaises(exception_type):
                        client.search("x", engine=engine)

    def test_shenma_parses_html_and_hydration_json_through_one_filter(self):
        response = FakeResponse(
            text=(FIXTURE_DIR / "shenma_organic_result.html").read_text(encoding="utf-8"),
            url=f"{SHENMA_ENDPOINT}?q=xjtu",
        )
        session = FakeSession(get_response=response)

        results = WebSearchClient(session).search("xjtu", engine="shenma", limit=5)

        self.assertEqual(
            results,
            [
                SearchResult(
                    "西安交通大学新闻网",
                    "https://news.xjtu.edu.cn/",
                    "西安交通大学新闻门户。",
                ),
                SearchResult(
                    "西安交通大学",
                    "http://www.xjtu.edu.cn",
                    "教育部直属重点大学。",
                ),
            ],
        )
        self.assertEqual([call[1] for call in session.calls], [SHENMA_ENDPOINT])
        self.assertEqual(session.calls[0][2]["params"], {"q": "xjtu"})

    def test_shenma_ignores_malformed_json_and_filters_unsafe_destinations(self):
        malformed_with_html = """
            <section class="sc">
              <a href="https://safe.test/" class="qk-link-wrapper">
                <div class="qk-title-text">Safe HTML</div>
              </a>
              <div class="qk-paragraph-text">Snippet</div>
            </section>
            <script type="application/json" data-used-by="hydrate">{broken</script>
        """
        response = FakeResponse(text=malformed_with_html, url=SHENMA_ENDPOINT)
        self.assertEqual(
            WebSearchClient(FakeSession(get_response=response)).search(
                "x", engine="shenma"
            ),
            [SearchResult("Safe HTML", "https://safe.test/", "Snippet")],
        )

        unsafe = """
            <section class="sc"><a href="javascript:alert(1)" class="qk-link-wrapper">
              <div class="qk-title-text">Unsafe HTML</div></a></section>
            <script type="application/json" data-used-by="hydrate">
              {"data":{"titleProps":{"content":"Unsafe JSON","dest_url":"https://user:secret@source.test/"}}}
            </script>
        """
        with self.assertRaisesRegex(RuntimeError, "页面结构无法解析"):
            WebSearchClient(
                FakeSession(get_response=FakeResponse(text=unsafe, url=SHENMA_ENDPOINT))
            ).search("x", engine="shenma")

    def test_shenma_distinguishes_challenge_empty_and_unknown_structure(self):
        cases = (
            ('<div class="captcha">captcha</div>', SearchHumanVerificationRequired),
            ('<div class="no-result">没有找到相关结果</div>', None),
            ('<div>changed layout</div>', RuntimeError),
        )
        for content, exception_type in cases:
            with self.subTest(exception=exception_type):
                response = FakeResponse(text=content, url=SHENMA_ENDPOINT)
                client = WebSearchClient(FakeSession(get_response=response))
                if exception_type is None:
                    self.assertEqual(client.search("x", engine="shenma"), [])
                else:
                    with self.assertRaises(exception_type):
                        client.search("x", engine="shenma")

    def test_google_enablejs_is_verification_not_empty_results(self):
        response = FakeResponse(
            text=(FIXTURE_DIR / "google_enablejs.html").read_text(encoding="utf-8"),
            url=f"{GOOGLE_ENDPOINT}?q=xjtu",
        )
        with self.assertRaisesRegex(SearchHumanVerificationRequired, "Google 要求人机验证"):
            WebSearchClient(FakeSession(get_response=response)).search(
                "xjtu", engine="google"
            )
        self.assertTrue(response.closed)

    def test_google_parser_accepts_standard_structure_without_live_fixture_claim(self):
        content = """
            <div id="search"><div class="result-contract">
              <a href="https://www.xjtu.edu.cn/"><h3>西安交通大学</h3></a>
              <div data-sncf="1">教育部直属重点大学。</div>
            </div></div>
        """
        self.assertEqual(
            WebSearchClient._parse_google(content),
            [
                SearchResult(
                    "西安交通大学",
                    "https://www.xjtu.edu.cn/",
                    "教育部直属重点大学。",
                )
            ],
        )

    def test_google_fixed_endpoint_uses_system_network_without_proxy_parameters(self):
        content = """
            <div id="search"><div>
              <a href="/url?q=https%3A%2F%2Fwww.xjtu.edu.cn%2F"><h3>西安交通大学</h3></a>
              <div class="VwiC3b">XJTU</div>
            </div></div>
        """
        response = FakeResponse(text=content, url=f"{GOOGLE_ENDPOINT}?q=xjtu")
        session = FakeSession(get_response=response)

        results = WebSearchClient(session).search("xjtu", engine="google", limit=3)

        self.assertEqual(results[0].url, "https://www.xjtu.edu.cn/")
        method, endpoint, kwargs = session.calls[0]
        self.assertEqual((method, endpoint), ("GET", GOOGLE_ENDPOINT))
        self.assertEqual(kwargs["params"], {"q": "xjtu"})
        self.assertNotIn("proxies", kwargs)
        self.assertFalse(kwargs["allow_redirects"])

    def test_google_distinguishes_empty_unknown_and_unsafe_results(self):
        empty = FakeResponse(
            text="<div>找不到和您的查询相符的内容</div>",
            url=GOOGLE_ENDPOINT,
        )
        self.assertEqual(
            WebSearchClient(FakeSession(get_response=empty)).search(
                "x", engine="google"
            ),
            [],
        )

        for content in (
            "<div>changed layout</div>",
            '<div id="search"><div><a href="javascript:alert(1)"><h3>Unsafe</h3></a></div></div>',
        ):
            with self.subTest(content=content):
                response = FakeResponse(text=content, url=GOOGLE_ENDPOINT)
                with self.assertRaisesRegex(RuntimeError, "页面结构无法解析"):
                    WebSearchClient(FakeSession(get_response=response)).search(
                        "x", engine="google"
                    )

    def test_duckduckgo_human_challenge_is_not_misreported_as_empty_results(self):
        challenged = FakeResponse(
            text='<html><form id="challenge-form"><div class="anomaly-modal">captcha</div></form></html>',
            url="https://html.duckduckgo.com/html/",
        )
        with self.assertRaisesRegex(SearchHumanVerificationRequired, "人机验证"):
            WebSearchClient(FakeSession(get_response=challenged)).search(
                "x", engine="duckduckgo", endpoint="https://ignored.example", limit=3
            )
        self.assertTrue(challenged.closed)

    def test_bing_direct_search_parses_organic_results_and_filters_unsafe_urls(self):
        real_fixture = (FIXTURE_DIR / "bing_organic_result.txt").read_text(
            encoding="utf-8"
        )
        response = FakeResponse(
            text=real_fixture.replace(
                "</ol>",
                '<li class="b_algo"><h2><a href="javascript:alert(1)">'
                "Unsafe result</a></h2></li></ol>",
            ),
            url="https://www.bing.com/search?q=x",
        )
        session = FakeSession(get_response=response)

        results = WebSearchClient(session).search(
            "x", engine="bing", endpoint="https://ignored.example", limit=3
        )

        self.assertEqual(
            [(one.title, one.url, one.snippet) for one in results],
            [
                (
                    "Xi'an Jiaotong University",
                    "https://en.xjtu.edu.cn/",
                    "Teaching and learning news from XJTU.",
                )
            ],
        )
        self.assertEqual(session.calls[0][1], BING_ENDPOINT)
        self.assertNotIn("engines", session.calls[0][2]["params"])

    def test_bing_tracking_links_are_unwrapped_without_following_redirects(self):
        response = FakeResponse(
            text="""
                <li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u=a1aHR0cHM6Ly93d3cueGp0dS5lZHUuY24v">
                  西安交通大学
                </a></h2></li>
            """,
            url="https://www.bing.com/search?q=x",
        )
        session = FakeSession(get_response=response)

        results = WebSearchClient(session).search("x", engine="bing", limit=3)

        self.assertEqual(results[0].url, "https://www.xjtu.edu.cn/")
        self.assertEqual(len(session.calls), 1)

    def test_auto_search_returns_bing_without_second_request(self):
        bing = FakeResponse(
            text=(FIXTURE_DIR / "bing_organic_result.txt").read_text(encoding="utf-8"),
            url=BING_ENDPOINT,
        )
        session = FakeSession(get_response=[bing])

        results = WebSearchClient(session).search("x", engine="auto", limit=3)

        self.assertEqual([one.title for one in results], ["Xi'an Jiaotong University"])
        self.assertEqual([call[1] for call in session.calls], [BING_ENDPOINT])

    def test_auto_search_falls_back_from_bing_challenge_to_duckduckgo(self):
        bing = FakeResponse(
            text='<form id="b_captcha"></form>',
            url=BING_ENDPOINT,
        )
        duckduckgo = FakeResponse(
            text=(FIXTURE_DIR / "duckduckgo_organic_result.txt").read_text(
                encoding="utf-8"
            ),
            url=DUCKDUCKGO_ENDPOINT,
        )
        session = FakeSession(get_response=[bing, duckduckgo])

        results = WebSearchClient(session).search("x", engine="auto", limit=3)

        self.assertEqual([one.title for one in results], ["Welcome to XJTU"])
        self.assertEqual(
            [call[1] for call in session.calls],
            [BING_ENDPOINT, DUCKDUCKGO_ENDPOINT],
        )

    def test_auto_search_falls_back_after_empty_results(self):
        bing = FakeResponse(
            text="<html></html>",
            url=BING_ENDPOINT,
        )
        duckduckgo = FakeResponse(
            text=(FIXTURE_DIR / "duckduckgo_organic_result.txt").read_text(
                encoding="utf-8"
            ),
            url=DUCKDUCKGO_ENDPOINT,
        )
        session = FakeSession(get_response=[bing, duckduckgo])

        results = WebSearchClient(session).search("x", engine="auto", limit=3)

        self.assertEqual([one.title for one in results], ["Welcome to XJTU"])
        self.assertEqual(len(session.calls), 2)

    def test_auto_search_reports_all_challenges_and_mixed_failures(self):
        duckduckgo_challenge = FakeResponse(
            text=(FIXTURE_DIR / "duckduckgo_challenge.txt").read_text(encoding="utf-8"),
            url=DUCKDUCKGO_ENDPOINT,
        )
        bing_challenge = FakeResponse(
            text='<form id="b_captcha"></form>',
            url="https://www.bing.com/search?q=x",
        )
        with self.assertRaisesRegex(SearchHumanVerificationRequired, "均要求人机验证"):
            WebSearchClient(FakeSession(get_response=[
                bing_challenge,
                duckduckgo_challenge,
            ])).search("x", engine="auto", limit=3)

        with self.assertRaisesRegex(RuntimeError, "内置联网搜索暂时不可用"):
            WebSearchClient(FakeSession(get_response=[
                requests.Timeout(),
                FakeResponse(text="<html></html>", url=DUCKDUCKGO_ENDPOINT),
            ])).search("x", engine="auto", limit=3)

    def test_one_challenge_and_one_empty_source_do_not_disable_search(self):
        bing_challenge = FakeResponse(
            text='<form id="b_captcha"></form>',
            url=BING_ENDPOINT,
        )
        duckduckgo_empty = FakeResponse(
            text="<html></html>",
            url=DUCKDUCKGO_ENDPOINT,
        )

        with self.assertRaisesRegex(RuntimeError, "内置联网搜索暂时不可用"):
            WebSearchClient(FakeSession(get_response=[
                bing_challenge,
                duckduckgo_empty,
            ])).search("x", engine="auto", limit=3)


class CapabilityAndMarkdownTest(unittest.TestCase):
    def test_only_explicit_local_capabilities_are_included_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notice = root / "notification.json"
            notice.write_text(json.dumps([{
                "title": "公开通知", "date": "2026-08-04", "source": "dean/jxtz",
                "link": "https://example.test/private?token=secret",
            }]), encoding="utf-8")
            (root / "score.json").write_text(json.dumps({"scores": [{
                "courseName": "高等数学", "score": 95, "coursePoint": 5, "gpa": 4.1,
                "studentId": "sensitive-id",
            }]}), encoding="utf-8")
            context = collect_local_context(
                ("public_notices",), notification_path=notice, account_directory=root
            )
            self.assertIn("公开通知", context.text)
            self.assertNotIn("高等数学", context.text)
            self.assertNotIn("token=", context.text)

            score_context = collect_local_context(("scores",), account_directory=root)
            self.assertIn("高等数学", score_context.text)
            self.assertNotIn("sensitive-id", score_context.text)

    def test_schedule_database_is_opened_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.db"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE courseinstance (name, day_of_week, start_time, end_time, week_number, location, teacher)"
            )
            connection.execute(
                "INSERT INTO courseinstance VALUES ('线性代数', 1, 1, 2, 3, '主楼', '张老师')"
            )
            connection.commit()
            connection.close()
            context = collect_local_context(("schedule",), account_directory=directory)
            self.assertIn("线性代数", context.text)
            self.assertIn("主楼", context.text)

    def test_selected_valid_empty_caches_are_sent_as_zero_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notice = root / "notification.json"
            notice.write_text("[]", encoding="utf-8")
            (root / "score.json").write_text(
                json.dumps({"scores": [], "terms": []}), encoding="utf-8"
            )
            (root / "attendance_flow.json").write_text("[]", encoding="utf-8")
            connection = sqlite3.connect(root / "schedule.db")
            connection.execute(
                "CREATE TABLE courseinstance "
                "(name, day_of_week, start_time, end_time, week_number, location, teacher)"
            )
            connection.commit()
            connection.close()

            context = collect_local_context(
                ("public_notices", "schedule", "scores", "attendance"),
                notification_path=notice,
                account_directory=root,
            )

            self.assertEqual(
                set(context.included),
                {"public_notices", "schedule", "scores", "attendance"},
            )
            self.assertFalse(context.unavailable)
            self.assertEqual(dict(context.counts), {
                "public_notices": 0,
                "schedule": 0,
                "scores": 0,
                "attendance": 0,
            })
            for heading in ("公开通知", "我的课表", "我的成绩", "我的考勤"):
                self.assertIn(f"## {heading}", context.text)
            self.assertGreaterEqual(context.text.count("暂无记录"), 4)
            self.assertIn("不得据此猜测", context.text)

    def test_missing_or_damaged_cache_is_reported_without_claiming_zero_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "score.json").write_text("not-json", encoding="utf-8")
            (root / "attendance_flow.json").write_text("{}", encoding="utf-8")
            (root / "schedule.db").write_text("not-sqlite", encoding="utf-8")

            context = collect_local_context(
                ("public_notices", "schedule", "scores", "attendance"),
                notification_path=root / "missing-notices.json",
                account_directory=root,
            )

            self.assertFalse(context.included)
            self.assertEqual(
                set(context.unavailable),
                {"public_notices", "schedule", "scores", "attendance"},
            )
            self.assertFalse(context.counts)
            for heading in ("公开通知", "我的课表", "我的成绩", "我的考勤"):
                self.assertIn(f"## {heading}（本机缓存不可用）", context.text)
            self.assertNotIn("已查询，0 条", context.text)
            self.assertIn("不存在、损坏或读取失败", context.text)
            self.assertIn("不得据此猜测", context.text)

    def test_available_empty_and_missing_cache_statuses_share_one_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notice = root / "notification.json"
            notice.write_text("[]", encoding="utf-8")

            context = collect_local_context(
                ("public_notices", "schedule"),
                notification_path=notice,
                account_directory=root,
            )

            self.assertEqual(context.included, ("public_notices",))
            self.assertEqual(context.unavailable, ("schedule",))
            self.assertEqual(dict(context.counts), {"public_notices": 0})
            self.assertIn("## 公开通知（已查询，0 条）", context.text)
            self.assertIn("## 我的课表（本机缓存不可用）", context.text)
            self.assertIn("暂无记录", context.text)
            self.assertIn("不存在、损坏或读取失败", context.text)

    def test_all_real_cache_shapes_share_the_budget_and_reach_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notice = root / "notification.json"
            notice.write_text(json.dumps([{
                "title": "创新竞赛通知", "date": "2026-08-05", "source": "dean/jxtz",
                "link": "https://secret.invalid/?token=not-sent",
            }]), encoding="utf-8")
            (root / "score.json").write_text(json.dumps({"scores": [
                {"courseName": "高等数学", "score": 90, "coursePoint": 4, "gpa": 4.0,
                 "studentId": "must-not-reach-provider"},
                {"courseName": "大学物理", "score": 80, "coursePoint": 2, "gpa": 3.0},
            ]}), encoding="utf-8")
            (root / "attendance_flow.json").write_text(json.dumps([{
                "sBh": "private-record-id", "eqno": "主楼A101",
                "watertime": "2026-08-05 08:01:00", "isdone": 1,
            }]), encoding="utf-8")
            connection = sqlite3.connect(root / "schedule.db")
            connection.execute(
                "CREATE TABLE courseinstance (name, day_of_week, start_time, end_time, week_number, location, teacher)"
            )
            connection.execute(
                "INSERT INTO courseinstance VALUES ('线性代数', 2, 3, 4, 1, '西一楼', '李老师')"
            )
            connection.commit()
            connection.close()

            context = collect_local_context(
                ("public_notices", "schedule", "scores", "attendance"),
                notification_path=notice,
                account_directory=root,
                max_characters=2400,
            )
            for heading in ("公开通知", "我的课表", "我的成绩", "我的考勤"):
                self.assertIn(f"## {heading}", context.text)
            for value in ("创新竞赛通知", "线性代数", "高等数学", "主楼A101"):
                self.assertIn(value, context.text)
            self.assertIn("成绩页统计", context.text)
            self.assertIn("86.667", context.text)
            self.assertNotIn("token=", context.text)
            self.assertNotIn("must-not-reach-provider", context.text)
            self.assertNotIn("private-record-id", context.text)
            self.assertEqual(set(context.included), {"public_notices", "schedule", "scores", "attendance"})
            self.assertFalse(context.unavailable)

    def test_markdown_renders_structure_but_removes_active_content(self):
        rendered = render_markdown_fragment(
            "# 标题\n\n**粗体** [安全](https://example.test) "
            "[危险](javascript:alert(1)) <script>alert(1)</script> ![跟踪](https://evil.test/x.png)"
        )
        self.assertIn("<h1>标题</h1>", rendered)
        self.assertIn("<strong>粗体</strong>", rendered)
        self.assertIn('href="https://example.test"', rendered)
        self.assertNotIn("javascript:", rendered)
        self.assertNotIn("<script", rendered)
        self.assertNotIn("<img", rendered)

    def test_code_characters_highlight_once_and_user_styles_remain_untrusted(self):
        rendered = render_markdown_fragment(
            "```cpp\n#include <iostream>\nif (a < b && b > 0) return 1;\n```"
        )
        self.assertIn("&lt;iostream&gt;", rendered)
        self.assertNotIn("&amp;lt;iostream", rendered)
        self.assertIn("<span style=", rendered)
        entity = render_markdown_fragment("```cpp\n#include &lt;iostream&gt;\n```")
        self.assertIn("&lt;iostream&gt;", entity)
        self.assertNotIn("&amp;lt;iostream", entity)

        hostile = render_markdown_fragment(
            '<script>alert(1)</script>x '
            '<span class="x\\" onload=alert(1)" style="color:red" onclick="bad()">safe</span> '
            '![pixel](https://evil.test/a.png) [bad](javascript:alert(1))'
        )
        self.assertNotIn("script", hostile.lower())
        self.assertNotIn("onload", hostile.lower())
        self.assertNotIn("onclick", hostile.lower())
        self.assertNotIn("style=", hostile.lower())
        self.assertNotIn("<img", hostile.lower())
        self.assertNotIn("javascript:", hostile.lower())

    def test_unknown_and_oversized_code_fall_back_safely(self):
        unknown = render_markdown_fragment("```not-a-real-language\n<a & b>\n```")
        self.assertIn("&lt;a &amp; b&gt;", unknown)
        oversized = render_markdown_fragment(
            "```python\n" + "x < y & z\n" * 12000 + "```"
        )
        self.assertNotIn("<span style=", oversized)
        self.assertIn("x &lt; y &amp; z", oversized)


class SharedScoreStatisticsTest(unittest.TestCase):
    def test_score_page_and_wenzhou_share_one_weighted_result(self):
        result = calculate_score_statistics([
            {"coursePoint": 4, "score": 90, "gpa": 4.0},
            {"coursePoint": 2, "score": 80, "gpa": 3.0},
            {"coursePoint": 1, "score": "A", "gpa": None},
        ])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.weighted_average, 86.666666, places=5)
        self.assertAlmostEqual(result.weighted_gpa, 3.666666, places=5)
        self.assertEqual(result.incomplete_count, 1)


class ConversationStoreTest(unittest.TestCase):
    def test_atomic_round_trip_active_session_and_corruption_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.json"
            store = ConversationStore(path)
            state = store.new_state()
            first = state.active
            first.title = "第一段"
            first.messages.extend([ChatMessage("user", "问题"), ChatMessage("assistant", "回答")])
            first.assistant_meta.append({"model": "fixture", "secret": "drop-me"})
            second = store.new_session("第二段")
            state.sessions.append(second)
            state.active_session_id = second.id
            store.save(state)
            loaded = store.load()
            self.assertEqual(loaded.active.title, "第二段")
            self.assertEqual(loaded.sessions[0].messages[-1].content, "回答")
            self.assertNotIn("secret", loaded.sessions[0].assistant_meta[0])
            self.assertFalse(any(path.parent.glob(f".{path.name}.*.tmp")))

            path.write_text("{broken", encoding="utf-8")
            fallback = store.load()
            self.assertEqual(len(fallback.sessions), 1)
            self.assertTrue(store.last_error)

    def test_rejects_oversized_single_message(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "conversations.json")
            state = store.new_state()
            state.active.messages.append(ChatMessage("user", "x" * (MAX_CONTENT_CHARACTERS + 1)))
            with self.assertRaisesRegex(ValueError, "超过限制"):
                store.save(state)


if __name__ == "__main__":
    unittest.main()

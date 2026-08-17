# Mainstream Zero-Cost Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不引入付费 API、应用内代理或凭据存储的前提下，为问舟提供自动、百度、Bing、Google、搜狗、360 搜索、神马、DuckDuckGo 和 SearXNG 九项搜索目录。

**Architecture:** `ai_assistant.web_search` 继续作为唯一搜索目录和普通 HTTP 适配层，各来源只解析自身公开结果页，最后统一执行 URL 安全过滤。自动模式严格串行执行 Bing、百度、360 搜索，只有三个来源全部要求人工验证时抛出聚合异常；UI 只消费稳定引擎 ID 和标签，不推导网络行为。

**Tech Stack:** Python 3、`requests.Session`、`lxml.html`、PyQt5/qfluentwidgets、标准库 `unittest`

## Global Constraints

- 搜索结果页固定使用 `requests`，不调用外部 `curl`，不增加依赖、API Key、付费 API 或项目托管代理。
- Google 和 DuckDuckGo 只标注“需要代理”；应用不新增代理 UI、代理探测、代理参数或代理凭据存储。
- 搜索词规范化后为 1–500 字符，结果数为 1–10；响应解压正文上限保持 2 MiB。
- 所有 GET/HEAD 均使用现有超时且 `allow_redirects=False`；结果页响应必须保持同源并在所有路径关闭。
- 自动模式顺序固定为 `bing -> baidu -> so360`，首个非空安全结果集立即返回，不合并、不重试。
- 百度只对最终入选的最多 `limit` 条结果做 HEAD 解链，最坏请求数为 3 个结果页加 10 个 HEAD。
- 只有明确无结果标记返回空列表；验证页抛人工验证异常，未知结构抛解析失败。
- schema 保持 v3；v2 `duckduckgo` 保持原 ID，v2 `bing`/`baidu`/`google`/`searxng` 迁移为 `searxng` 并保留端点。
- 自动化测试只读脱敏离线 fixture，不访问实时搜索站点；Google 正常页只接受真实响应或可追溯开源样本。
- 搜索专项不得包含 CI gate 文件；`cooperate.md` 只作本地评审通信，不进入产品 PR。
- commit 使用英文 Conventional Commit 类型、ASCII 冒号和中文正文。

---

## File Structure

- Modify: `ai_assistant/web_search.py` — 权威引擎目录、共享有界请求、七个公开页适配器、自动回退和 URL 过滤。
- Modify: `ai_assistant/config.py` — 按序列化版本执行 v2 搜索路由迁移。
- Modify: `ai_assistant/__init__.py` — 导出自动模式聚合验证异常。
- Modify: `app/AIInterface.py` — 让显式验证失败走普通请求错误，仅聚合验证关闭联网搜索。
- Modify: `test/ai_assistant/test_ai_features.py` — 搜索适配器、配置迁移、请求安全和自动回退回归。
- Modify: `test/app/test_notice_search_ui.py` — 九项目录、端点显隐、持久化和线程异常分流回归。
- Create: `test/ai_assistant/fixtures/baidu_organic_result.html` — 从保留的百度真实响应提取并脱敏的结果结构。
- Create: `test/ai_assistant/fixtures/google_enablejs.html` — 从保留的 Google 香港站响应提取的 `enablejs` 验证结构。
- Create: `test/ai_assistant/fixtures/sogou_organic_result.html` — 从搜狗移动真实响应提取并脱敏的 `vrResult` 结构。
- Create: `test/ai_assistant/fixtures/so360_organic_result.html` — 从 360 真实响应提取并脱敏的 `res-list`/`data-mdurl` 结构。
- Create: `test/ai_assistant/fixtures/shenma_organic_result.html` — 从神马真实响应提取并脱敏的普通链接和 hydration JSON 结构。

### Task 1: 固定引擎目录和配置迁移

**Files:**
- Modify: `ai_assistant/web_search.py:14-53`
- Modify: `ai_assistant/config.py:230-237`
- Test: `test/ai_assistant/test_ai_features.py:101-157`

**Interfaces:**
- Produces: `SEARCH_ENGINES: tuple[tuple[str, str], ...]`、`BUILTIN_ENGINES: frozenset[str]`。
- Produces: `_profile_from_payload(data: dict, version: int) -> AIProfile` 的 v2 路由迁移。

- [ ] **Step 1: 写目录、校验和迁移失败测试**

```python
def test_search_catalog_has_approved_ids_labels_and_builtin_boundary(self):
    self.assertEqual(SEARCH_ENGINES, (
        ("auto", "自动（直连推荐）"),
        ("baidu", "百度（直连）"),
        ("bing", "Bing（直连）"),
        ("google", "Google（需要代理）"),
        ("sogou", "搜狗（直连）"),
        ("so360", "360 搜索（直连）"),
        ("shenma", "神马（直连）"),
        ("duckduckgo", "DuckDuckGo（需要代理）"),
        ("searxng", "SearXNG（自托管）"),
    ))
    for engine, _label in SEARCH_ENGINES[:-1]:
        self.assertEqual(validate_search_settings(engine, "https://ignored.example"), (engine, ""))

def test_v2_search_engines_migrate_without_changing_network_route(self):
    expected = {
        "duckduckgo": ("duckduckgo", ""),
        "bing": ("searxng", "https://search.example/"),
        "baidu": ("searxng", "https://search.example/"),
        "google": ("searxng", "https://search.example/"),
        "searxng": ("searxng", "https://search.example/"),
    }
    with tempfile.TemporaryDirectory() as directory:
        for legacy_engine, expected_settings in expected.items():
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
                self.assertEqual(
                    (loaded.search_engine, loaded.search_endpoint),
                    expected_settings,
                )
```

- [ ] **Step 2: 运行测试确认旧目录和 duckduckgo 迁移失败**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.IndependentProtocolConfigTest.test_v2_search_engines_migrate_without_changing_network_route test.ai_assistant.test_ai_features.WebSearchTest.test_search_catalog_has_approved_ids_labels_and_builtin_boundary -v`

Expected: FAIL，旧目录只有四项，且 v2 `duckduckgo` 被迁移为 `auto`。

- [ ] **Step 3: 写入固定目录、端点和迁移规则**

```python
BAIDU_ENDPOINT = "https://www.baidu.com/s"
BING_ENDPOINT = "https://www.bing.com/search"
GOOGLE_ENDPOINT = "https://www.google.com.hk/search"
SOGOU_ENDPOINT = "https://m.sogou.com/web/searchList.jsp"
SO360_ENDPOINT = "https://www.so.com/s"
SHENMA_ENDPOINT = "https://m.sm.cn/s"
DUCKDUCKGO_ENDPOINT = "https://html.duckduckgo.com/html/"

SEARCH_ENGINES = (
    ("auto", "自动（直连推荐）"),
    ("baidu", "百度（直连）"),
    ("bing", "Bing（直连）"),
    ("google", "Google（需要代理）"),
    ("sogou", "搜狗（直连）"),
    ("so360", "360 搜索（直连）"),
    ("shenma", "神马（直连）"),
    ("duckduckgo", "DuckDuckGo（需要代理）"),
    ("searxng", "SearXNG（自托管）"),
)
BUILTIN_ENGINES = frozenset(engine for engine, _label in SEARCH_ENGINES if engine != "searxng")
```

```python
if version < 3 and profile.search_engine in {"bing", "baidu", "google", "searxng"}:
    profile = replace(profile, search_engine="searxng")
elif version < 3 and profile.search_engine == "duckduckgo":
    profile = replace(profile, search_endpoint="")
```

- [ ] **Step 4: 运行迁移和当前 schema 往返测试**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.IndependentProtocolConfigTest -v`

Expected: PASS，schema 仍为 3，所有内置引擎保存后端点为空。

- [ ] **Step 5: 提交目录和迁移**

```bash
git commit --only ai_assistant/web_search.py ai_assistant/config.py test/ai_assistant/test_ai_features.py -m "feat(search): 增加主流搜索引擎目录"
```

### Task 2: 强化共享解析状态和请求测试替身

**Files:**
- Modify: `ai_assistant/web_search.py:61-190`
- Modify: `test/ai_assistant/test_ai_features.py:25-68,224-284`

**Interfaces:**
- Produces: `SearchAllSourcesVerificationRequired(SearchHumanVerificationRequired)`。
- Produces: `_parse_html(content: str, engine_name: str) -> HtmlElement`、`_require_results_or_empty(document, results, *, no_result_xpath, engine_name)`。
- Produces: `FakeSession.head(url, **kwargs)`，记录 `("HEAD", url, kwargs)`。

- [ ] **Step 1: 写响应关闭、HEAD 记录、未知结构和结果过滤测试**

```python
def test_result_filter_rejects_missing_host_credentials_duplicates_and_empty_title(self):
    rows = [
        ("ok", "https://source.test/a", "one"),
        ("duplicate", "https://source.test/a", "two"),
        ("", "https://source.test/b", "empty"),
        ("bad", "https://user:secret@source.test/c", "credentials"),
        ("bad", "http:///missing-host", "host"),
    ]
    self.assertEqual(_normalize_results(rows), [SearchResult("ok", "https://source.test/a", "one")])

def test_bing_unknown_structure_is_not_reported_as_empty(self):
    response = FakeResponse(text="<html><body>changed layout</body></html>", url=BING_ENDPOINT)
    with self.assertRaisesRegex(RuntimeError, "Bing 页面结构无法解析"):
        WebSearchClient(FakeSession(get_response=response)).search("x", engine="bing")
    self.assertTrue(response.closed)
```

- [ ] **Step 2: 运行测试确认空结构仍被当成零结果**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest.test_result_filter_rejects_missing_host_credentials_duplicates_and_empty_title test.ai_assistant.test_ai_features.WebSearchTest.test_bing_unknown_structure_is_not_reported_as_empty -v`

Expected: 第二项 FAIL，当前 Bing 解析器返回空列表。

- [ ] **Step 3: 增加聚合异常、解析状态 helper 和 HEAD 测试替身**

```python
class SearchAllSourcesVerificationRequired(SearchHumanVerificationRequired):
    """Every source in automatic mode returned a challenge page."""


def _parse_html(content: str, engine_name: str):
    try:
        return html.fromstring(content)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{engine_name} 返回了无法解析的页面") from error


def _require_results_or_empty(document, results, *, no_result_xpath: str, engine_name: str):
    if results or document.xpath(no_result_xpath):
        return results
    raise RuntimeError(f"{engine_name} 页面结构无法解析")
```

```python
class FakeSession:
    def __init__(self, get_response=None, post_response=None, head_response=None):
        self.get_response = get_response
        self.post_response = post_response
        self.head_response = head_response
        self.calls = []

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", url, kwargs))
        response = self.head_response
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")
```

- [ ] **Step 4: 运行共享请求与解析状态测试**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest.test_custom_searxng_is_bounded_and_parsed test.ai_assistant.test_ai_features.WebSearchTest.test_timeout_redirect_and_oversized_stream_are_safe test.ai_assistant.test_ai_features.WebSearchTest.test_bing_unknown_structure_is_not_reported_as_empty test.ai_assistant.test_ai_features.WebSearchTest.test_result_filter_rejects_missing_host_credentials_duplicates_and_empty_title -v`

Expected: PASS。

- [ ] **Step 5: 提交共享失败语义**

```bash
git commit --only ai_assistant/web_search.py test/ai_assistant/test_ai_features.py -m "fix(search): 区分空结果与页面解析失败"
```

### Task 3: 增加百度解析和安全 HEAD 解链

**Files:**
- Create: `test/ai_assistant/fixtures/baidu_organic_result.html`
- Modify: `ai_assistant/web_search.py`
- Modify: `test/ai_assistant/test_ai_features.py`

**Interfaces:**
- Produces: `_parse_baidu(content: str) -> list[SearchResult]`，结果 URL 在解链前可为百度 link URL。
- Produces: `_resolve_baidu_results(results: list[SearchResult], limit: int) -> list[SearchResult]`。
- Consumes: `self.session.head(..., allow_redirects=False, timeout=self.timeout)`。

- [ ] **Step 1: 从真实响应建立最小脱敏 fixture**

```html
<!-- Source: /tmp/xjtu-probe-baidu-compressed-xjtu.body, captured 2026-08-17.
     Redacted: request ids, account state, scripts, styles, tracking fields and unrelated results. -->
<html><body>
  <div class="result c-container xpath-log new-pmd" srcid="1599">
    <h3 class="t"><a href="https://www.baidu.com/link?url=fixture-token">西安交通大学</a></h3>
    <div class="c-abstract">西安交通大学是教育部直属重点大学。</div>
  </div>
</body></html>
```

- [ ] **Step 2: 写有界解链和 challenge/空页测试**

```python
def test_baidu_resolves_only_selected_results_with_safe_head_requests(self):
    page = FakeResponse(text=fixture("baidu_organic_result.html"), url=BAIDU_ENDPOINT)
    resolved = FakeResponse(status=302, url="https://www.baidu.com/link?url=fixture-token")
    resolved.headers = {"Location": "https://www.xjtu.edu.cn/"}
    session = FakeSession(get_response=page, head_response=resolved)
    results = WebSearchClient(session).search("xjtu", engine="baidu", limit=1)
    self.assertEqual(results, [SearchResult("西安交通大学", "https://www.xjtu.edu.cn/", "西安交通大学是教育部直属重点大学。")])
    method, _url, kwargs = session.calls[1]
    self.assertEqual(method, "HEAD")
    self.assertFalse(kwargs["allow_redirects"])
    self.assertEqual(kwargs["timeout"], (10, 20))
    self.assertTrue(resolved.closed)

def test_baidu_drops_unsafe_and_failed_redirect_targets(self):
    links = "".join(
        '<div class="c-container"><h3><a href="https://www.baidu.com/link?url=%d">R%d</a></h3></div>' % (index, index)
        for index in range(5)
    )
    page = FakeResponse(text=f"<html><body>{links}</body></html>", url=BAIDU_ENDPOINT)
    javascript = FakeResponse(status=302, url="https://www.baidu.com/link?url=0")
    javascript.headers = {"Location": "javascript:alert(1)"}
    credentials = FakeResponse(status=302, url="https://www.baidu.com/link?url=1")
    credentials.headers = {"Location": "https://user:secret@source.test/"}
    missing = FakeResponse(status=302, url="https://www.baidu.com/link?url=2")
    missing.headers = {}
    wrong_status = FakeResponse(status=200, url="https://www.baidu.com/link?url=3")
    wrong_status.headers = {"Location": "https://source.test/ignored"}
    session = FakeSession(
        get_response=page,
        head_response=[javascript, credentials, missing, wrong_status, requests.Timeout()],
    )
    self.assertEqual(WebSearchClient(session).search("x", engine="baidu", limit=5), [])
    self.assertEqual([call[0] for call in session.calls], ["GET", "HEAD", "HEAD", "HEAD", "HEAD", "HEAD"])
    self.assertTrue(all(response.closed for response in (page, javascript, credentials, missing, wrong_status)))
```

- [ ] **Step 3: 运行测试确认百度尚未受支持**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest.test_baidu_resolves_only_selected_results_with_safe_head_requests test.ai_assistant.test_ai_features.WebSearchTest.test_baidu_drops_unsafe_and_failed_redirect_targets -v`

Expected: FAIL，`_search_one` 尚无百度分支。

- [ ] **Step 4: 实现百度解析和逐条安全解链**

```python
def _parse_baidu(content: str) -> list[SearchResult]:
    document = _parse_html(content, "百度")
    if document.xpath("//*[contains(@class, 'verify') or @id='verify-form']"):
        raise SearchHumanVerificationRequired("百度要求人机验证")
    rows = []
    for result in document.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' c-container ')]"):
        links = result.xpath(".//h3//a[@href]")
        if not links:
            continue
        snippets = result.xpath(".//*[contains(concat(' ', normalize-space(@class), ' '), ' c-abstract ')]")
        rows.append((links[0].text_content(), links[0].get("href"), snippets[0].text_content() if snippets else ""))
    results = _normalize_results(rows)
    return _require_results_or_empty(
        document, results,
        no_result_xpath="//*[contains(@class, 'op_sp_realtime_n_result') or contains(text(), '没有找到')]",
        engine_name="百度",
    )
```

```python
def _resolve_baidu_results(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
    resolved = []
    for result in results[:limit]:
        parsed = urlparse(result.url)
        if not _is_host(parsed, "baidu.com") or not parsed.path.startswith("/link"):
            resolved.extend(_normalize_results([(result.title, result.url, result.snippet)]))
            continue
        response = None
        try:
            response = self.session.head(result.url, headers=_BROWSER_HEADERS, timeout=self.timeout, allow_redirects=False)
            location = str(response.headers.get("Location", "")) if 300 <= int(response.status_code) < 400 else ""
            resolved.extend(_normalize_results([(result.title, urljoin(result.url, location), result.snippet)]))
        except requests.RequestException:
            continue
        finally:
            if response is not None:
                response.close()
    return resolved
```

- [ ] **Step 5: 运行百度与共享安全测试**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest -k baidu -v`

Expected: PASS，未发生实时网络请求。

- [ ] **Step 6: 提交百度适配器**

```bash
git commit --only ai_assistant/web_search.py test/ai_assistant/test_ai_features.py test/ai_assistant/fixtures/baidu_organic_result.html -m "feat(search): 增加百度搜索和安全解链"
```

### Task 4: 增加 360 和搜狗适配器

**Files:**
- Create: `test/ai_assistant/fixtures/so360_organic_result.html`
- Create: `test/ai_assistant/fixtures/sogou_organic_result.html`
- Modify: `ai_assistant/web_search.py`
- Modify: `test/ai_assistant/test_ai_features.py`

**Interfaces:**
- Produces: `_parse_so360(content: str) -> list[SearchResult]`，优先消费 `data-mdurl`。
- Produces: `_parse_sogou(content: str) -> list[SearchResult]`，只本地解码 `url` 参数。

- [ ] **Step 1: 建立真实结构 fixture**

```html
<!-- Source: /tmp/xjtu-probe-so360.body, captured 2026-08-17; tracking ids and unrelated nodes removed. -->
<html><body><ul class="result"><li class="res-list">
  <h3 class="res-title"><a href="https://www.so.com/link?m=redacted" data-mdurl="https://news.xjtu.edu.cn/">西安交通大学新闻网</a></h3>
  <p class="res-desc">西安交通大学新闻门户。</p>
</li></ul></body></html>
```

```html
<!-- Source: /tmp/xjtu-probe-sogou-mobile.body, captured 2026-08-17; scripts, styles and tracking ids removed. -->
<html><body><div class="vrResult">
  <h3><a class="resultLink" href="/web/searchList.jsp?url=https%3A%2F%2Fwww.xjtu.edu.cn%2F">西安交通大学</a></h3>
  <p class="str_info">西安交通大学官方网站。</p>
</div></body></html>
```

- [ ] **Step 2: 写端点参数、正常解析、危险目标、明确空页和未知结构测试**

```python
def test_so360_uses_data_mdurl_without_requesting_tracking_link(self):
    response = FakeResponse(text=fixture("so360_organic_result.html"), url=SO360_ENDPOINT)
    session = FakeSession(get_response=response)
    results = WebSearchClient(session).search("xjtu", engine="so360", limit=3)
    self.assertEqual(results[0].url, "https://news.xjtu.edu.cn/")
    self.assertEqual([call[1] for call in session.calls], [SO360_ENDPOINT])
    self.assertEqual(session.calls[0][2]["params"], {"q": "xjtu"})

def test_sogou_decodes_url_parameter_without_following_redirect(self):
    response = FakeResponse(text=fixture("sogou_organic_result.html"), url=SOGOU_ENDPOINT)
    session = FakeSession(get_response=response)
    results = WebSearchClient(session).search("xjtu", engine="sogou", limit=3)
    self.assertEqual(results[0].url, "https://www.xjtu.edu.cn/")
    self.assertEqual(len(session.calls), 1)
    self.assertEqual(session.calls[0][2]["params"], {"keyword": "xjtu"})
```

- [ ] **Step 3: 运行测试确认两个适配器缺失**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest.test_so360_uses_data_mdurl_without_requesting_tracking_link test.ai_assistant.test_ai_features.WebSearchTest.test_sogou_decodes_url_parameter_without_following_redirect -v`

Expected: FAIL。

- [ ] **Step 4: 实现两个纯本地解码解析器**

```python
def _parse_so360(content: str) -> list[SearchResult]:
    document = _parse_html(content, "360 搜索")
    if document.xpath("//*[@id='verify' or contains(@class, 'captcha')]"):
        raise SearchHumanVerificationRequired("360 搜索要求人机验证")
    rows = []
    for result in document.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' res-list ')]"):
        links = result.xpath(".//h3//a[@href]")
        if links:
            link = links[0]
            snippets = result.xpath(".//*[contains(concat(' ', normalize-space(@class), ' '), ' res-desc ')]")
            rows.append((link.text_content(), link.get("data-mdurl") or link.get("href"), snippets[0].text_content() if snippets else ""))
    return _require_results_or_empty(document, _normalize_results(rows), no_result_xpath="//*[contains(text(), '没有找到相关结果')]", engine_name="360 搜索")
```

```python
def _parse_sogou(content: str) -> list[SearchResult]:
    document = _parse_html(content, "搜狗")
    if document.xpath("//*[contains(@class, 'verify') or contains(@class, 'captcha')]"):
        raise SearchHumanVerificationRequired("搜狗要求人机验证")
    rows = []
    for result in document.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' vrResult ')]"):
        links = result.xpath(".//h3//a[@href] | .//a[contains(@class, 'resultLink')][@href]")
        if links:
            parsed = urlparse(urljoin(SOGOU_ENDPOINT, links[0].get("href")))
            target = parse_qs(parsed.query).get("url", [links[0].get("href")])[0]
            snippets = result.xpath(".//p | .//*[contains(@class, 'str_info')]")
            rows.append((links[0].text_content(), unquote(target), snippets[0].text_content() if snippets else ""))
    return _require_results_or_empty(document, _normalize_results(rows), no_result_xpath="//*[contains(text(), '未找到相关结果')]", engine_name="搜狗")
```

- [ ] **Step 5: 运行 360、搜狗和危险 URL 测试**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest -k so360 -v && uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest -k sogou -v`

Expected: PASS。

- [ ] **Step 6: 提交 360 和搜狗适配器**

```bash
git commit --only ai_assistant/web_search.py test/ai_assistant/test_ai_features.py test/ai_assistant/fixtures/so360_organic_result.html test/ai_assistant/fixtures/sogou_organic_result.html -m "feat(search): 增加360和搜狗搜索"
```

### Task 5: 增加神马普通链接和 hydration JSON 适配器

**Files:**
- Create: `test/ai_assistant/fixtures/shenma_organic_result.html`
- Modify: `ai_assistant/web_search.py`
- Modify: `test/ai_assistant/test_ai_features.py`

**Interfaces:**
- Produces: `_parse_shenma(content: str) -> list[SearchResult]`。
- Produces: `_walk_dicts(value)`，只遍历 `dict`/`list` 并产出字典节点。

- [ ] **Step 1: 建立含 HTML 和真实字段名的脱敏 fixture**

```html
<!-- Source: /tmp/xjtu-probe-shenma.body, captured 2026-08-17; request ids and unrelated payload removed. -->
<html><body>
  <a class="c-header-inner" href="https://news.xjtu.edu.cn/"><h2>西安交通大学新闻网</h2></a>
  <p class="c-line-clamp3">西安交通大学新闻门户。</p>
  <script type="application/json">{"titleProps":{"content":"西安交通大学","dest_url":"http://www.xjtu.edu.cn"},"summaryProps":{"content":"教育部直属重点大学。","dest_url":"http://www.xjtu.edu.cn"}}</script>
</body></html>
```

- [ ] **Step 2: 写 HTML/JSON 去重、危险 `dest_url` 和异常 JSON 测试**

```python
def test_shenma_parses_links_and_hydration_json_through_one_filter(self):
    response = FakeResponse(text=fixture("shenma_organic_result.html"), url=SHENMA_ENDPOINT)
    session = FakeSession(get_response=response)
    results = WebSearchClient(session).search("xjtu", engine="shenma", limit=5)
    self.assertEqual([one.url for one in results], ["https://news.xjtu.edu.cn/", "http://www.xjtu.edu.cn"])
    self.assertEqual(session.calls[0][2]["params"], {"q": "xjtu"})
```

- [ ] **Step 3: 运行测试确认神马分支缺失**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest.test_shenma_parses_links_and_hydration_json_through_one_filter -v`

Expected: FAIL。

- [ ] **Step 4: 实现受限 JSON 遍历和统一过滤**

```python
def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _parse_shenma(content: str) -> list[SearchResult]:
    document = _parse_html(content, "神马")
    if document.xpath("//*[contains(@class, 'captcha') or contains(@class, 'verify')]"):
        raise SearchHumanVerificationRequired("神马要求人机验证")
    rows = []
    for link in document.xpath("//a[@href][.//h2 or contains(@class, 'c-header-inner')]"):
        snippets = link.xpath("following::*[contains(@class, 'c-line-clamp')][1]")
        rows.append((link.text_content(), link.get("href"), snippets[0].text_content() if snippets else ""))
    for node in document.xpath("//script[@type='application/json']/text()"):
        try:
            payload = json.loads(node)
        except ValueError:
            continue
        for item in _walk_dicts(payload):
            title = item.get("titleProps")
            summary = item.get("summaryProps", {})
            if isinstance(title, dict):
                rows.append((title.get("content"), title.get("dest_url"), summary.get("content") if isinstance(summary, dict) else ""))
    return _require_results_or_empty(document, _normalize_results(rows), no_result_xpath="//*[contains(text(), '没有找到相关结果')]", engine_name="神马")
```

- [ ] **Step 5: 运行神马测试**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest -k shenma -v`

Expected: PASS。

- [ ] **Step 6: 提交神马适配器**

```bash
git commit --only ai_assistant/web_search.py test/ai_assistant/test_ai_features.py test/ai_assistant/fixtures/shenma_organic_result.html -m "feat(search): 增加神马搜索"
```

### Task 6: 增加 Google 正常结构解析和验证页识别

**Files:**
- Create: `test/ai_assistant/fixtures/google_enablejs.html`
- Modify: `ai_assistant/web_search.py`
- Modify: `test/ai_assistant/test_ai_features.py`

**Interfaces:**
- Produces: `_parse_google(content: str) -> list[SearchResult]`。
- Constraint: 没有真实或可追溯正常页 fixture 时，不新增伪造的“正常页已验证”fixture；解析器用结构测试覆盖，验收说明保留证据限制。

- [ ] **Step 1: 从已观测响应提取 `enablejs` fixture**

```html
<!-- Source: /tmp/xjtu-probe-google-hk.body, captured 2026-08-17; nonce and request id removed. -->
<html><body><noscript>
  <meta content="0;url=/httpservice/retry/enablejs" http-equiv="refresh">
  <a href="/httpservice/retry/enablejs">此处</a>
</noscript></body></html>
```

- [ ] **Step 2: 写 challenge、端点参数、结构解析和危险链接测试**

```python
def test_google_enablejs_is_verification_not_empty_results(self):
    response = FakeResponse(text=fixture("google_enablejs.html"), url=GOOGLE_ENDPOINT)
    with self.assertRaisesRegex(SearchHumanVerificationRequired, "Google 要求人机验证"):
        WebSearchClient(FakeSession(get_response=response)).search("xjtu", engine="google")

def test_google_parser_accepts_standard_result_shape_without_claiming_live_fixture(self):
    content = '<div id="search"><div><a href="https://www.xjtu.edu.cn/"><h3>西安交通大学</h3></a><div data-sncf="1">教育部直属重点大学。</div></div></div>'
    results = WebSearchClient._parse_google(content)
    self.assertEqual(results, [SearchResult("西安交通大学", "https://www.xjtu.edu.cn/", "教育部直属重点大学。")])
```

- [ ] **Step 3: 运行测试确认 Google 尚未受支持**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest.test_google_enablejs_is_verification_not_empty_results test.ai_assistant.test_ai_features.WebSearchTest.test_google_parser_accepts_standard_result_shape_without_claiming_live_fixture -v`

Expected: FAIL。

- [ ] **Step 4: 实现标准结构解析并优先识别 challenge**

```python
def _parse_google(content: str) -> list[SearchResult]:
    document = _parse_html(content, "Google")
    if document.xpath("//a[contains(@href, '/httpservice/retry/enablejs')] | //form[contains(@action, '/sorry/')]") or "unusual traffic" in content.lower():
        raise SearchHumanVerificationRequired("Google 要求人机验证；请检查桌面代理后重试")
    rows = []
    for heading in document.xpath("//*[@id='search']//a[@href]/h3"):
        link = heading.getparent()
        container = link.getparent()
        snippets = container.xpath(".//*[@data-sncf or contains(@class, 'VwiC3b')]")
        rows.append((heading.text_content(), link.get("href"), snippets[0].text_content() if snippets else ""))
    return _require_results_or_empty(document, _normalize_results(rows), no_result_xpath="//*[contains(text(), '找不到和您的查询相符的内容') or contains(text(), 'did not match any documents')]", engine_name="Google")
```

- [ ] **Step 5: 运行 Google 测试并确认请求未携带应用代理参数**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest -k google -v`

Expected: PASS，GET 调用参数中不存在 `proxies`。

- [ ] **Step 6: 提交 Google 适配器**

```bash
git commit --only ai_assistant/web_search.py test/ai_assistant/test_ai_features.py test/ai_assistant/fixtures/google_enablejs.html -m "feat(search): 增加Google搜索验证页识别"
```

### Task 7: 接入所有显式来源并重写自动回退

**Files:**
- Modify: `ai_assistant/web_search.py`
- Modify: `ai_assistant/__init__.py`
- Modify: `test/ai_assistant/test_ai_features.py`

**Interfaces:**
- Consumes: 六个固定 HTML 端点、DuckDuckGo HTML、SearXNG JSON。
- Produces: `_search_one(query: str, engine: str, endpoint: str, limit: int) -> list[SearchResult]` 的完整分派。
- Produces: `_search_auto(query: str, limit: int) -> list[SearchResult]`，只尝试 `("bing", "baidu", "so360")`。

- [ ] **Step 1: 写分派表、自动顺序、早停、混合失败和三源全验证测试**

```python
def test_auto_falls_back_bing_baidu_so360_and_stops_at_first_results(self):
    session = FakeSession(
        get_response=[
            FakeResponse(text='<form id="b_captcha"></form>', url=BING_ENDPOINT),
            FakeResponse(text='<div class="op_sp_realtime_n_result">没有找到</div>', url=BAIDU_ENDPOINT),
            FakeResponse(text=fixture("so360_organic_result.html"), url=SO360_ENDPOINT),
        ]
    )
    results = WebSearchClient(session).search("xjtu", engine="auto", limit=3)
    self.assertEqual(results[0].url, "https://news.xjtu.edu.cn/")
    self.assertEqual([call[1] for call in session.calls], [BING_ENDPOINT, BAIDU_ENDPOINT, SO360_ENDPOINT])

def test_auto_raises_aggregate_verification_only_when_all_three_challenge(self):
    challenged = [
        FakeResponse(text='<form id="b_captcha"></form>', url=BING_ENDPOINT),
        FakeResponse(text='<div id="verify-form"></div>', url=BAIDU_ENDPOINT),
        FakeResponse(text='<div id="verify"></div>', url=SO360_ENDPOINT),
    ]
    with self.assertRaises(SearchAllSourcesVerificationRequired):
        WebSearchClient(FakeSession(get_response=challenged)).search("x", engine="auto")
```

- [ ] **Step 2: 运行测试确认旧自动模式仍请求 DuckDuckGo**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest.test_auto_falls_back_bing_baidu_so360_and_stops_at_first_results test.ai_assistant.test_ai_features.WebSearchTest.test_auto_raises_aggregate_verification_only_when_all_three_challenge -v`

Expected: FAIL。

- [ ] **Step 3: 用固定描述表分派显式来源并实现三源自动模式**

```python
def _search_auto(self, query: str, limit: int) -> list[SearchResult]:
    challenges = 0
    for engine in ("bing", "baidu", "so360"):
        try:
            results = self._search_one(query, engine, "", limit)
        except SearchHumanVerificationRequired:
            challenges += 1
            continue
        except RuntimeError:
            continue
        if results:
            return results
    if challenges == 3:
        raise SearchAllSourcesVerificationRequired("自动模式的公开搜索服务均要求人机验证；已自动关闭联网搜索")
    raise RuntimeError("内置联网搜索暂时不可用，请稍后重试")
```

```python
adapters = {
    "baidu": (BAIDU_ENDPOINT, {"wd": query}, self._parse_baidu),
    "bing": (BING_ENDPOINT, {"q": query, "setlang": "zh-Hans"}, self._parse_bing),
    "google": (GOOGLE_ENDPOINT, {"q": query}, self._parse_google),
    "sogou": (SOGOU_ENDPOINT, {"keyword": query}, self._parse_sogou),
    "so360": (SO360_ENDPOINT, {"q": query}, self._parse_so360),
    "shenma": (SHENMA_ENDPOINT, {"q": query}, self._parse_shenma),
    "duckduckgo": (DUCKDUCKGO_ENDPOINT, {"q": query}, self._parse_duckduckgo),
}
endpoint_url, params, parser = adapters[engine]
content, encoding = self._fetch(
    endpoint_url,
    params=params,
    accept="text/html,application/xhtml+xml",
)
results = parser(content.decode(encoding, errors="replace"))
if engine == "baidu":
    return self._resolve_baidu_results(results, limit)
return results[:limit]
```

- [ ] **Step 4: 运行全部搜索核心测试**

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest -v`

Expected: PASS；断言自动模式调用列表不含 Google、DuckDuckGo、搜狗、神马和 SearXNG。

- [ ] **Step 5: 提交完整路由和自动回退**

```bash
git commit --only ai_assistant/web_search.py ai_assistant/__init__.py test/ai_assistant/test_ai_features.py -m "feat(search): 实现三源自动回退"
```

### Task 8: 区分显式验证失败和自动聚合验证

**Files:**
- Modify: `app/AIInterface.py:58-69,136-149`
- Modify: `test/app/test_notice_search_ui.py:620-675`

**Interfaces:**
- Consumes: `SearchAllSourcesVerificationRequired`。
- Produces: `_AIRequestThread.webSearchDisabled` 只承载聚合验证；显式 `SearchHumanVerificationRequired` 进入 `_AIRequestThread.failed`。

- [ ] **Step 1: 写两条信号分流测试**

```python
def test_explicit_verification_fails_request_without_disabling_saved_search(self):
    class RaisingSearch:
        session = SimpleNamespace(close=lambda: None)

        def search(self, _query, **_settings):
            raise SearchHumanVerificationRequired("Google 要求人机验证")

    thread = _AIRequestThread(
        [ChatMessage("user", "查询")],
        ProviderConfig("openai", "https://api.example/v1", "fixture", "key"),
        search_query="xjtu",
        search_settings={"engine": "google", "endpoint": "", "limit": 3},
    )
    thread.search_client = RaisingSearch()
    failed, disabled = [], []
    thread.failed.connect(failed.append)
    thread.webSearchDisabled.connect(disabled.append)
    thread.run()
    self.assertEqual(failed[0].message, "Google 要求人机验证")
    self.assertFalse(disabled)

def test_aggregate_verification_emits_disable_signal(self):
    class RaisingSearch:
        session = SimpleNamespace(close=lambda: None)

        def search(self, _query, **_settings):
            raise SearchAllSourcesVerificationRequired("均要求人机验证")

    thread = _AIRequestThread(
        [ChatMessage("user", "查询")],
        ProviderConfig("openai", "https://api.example/v1", "fixture", "key"),
        search_query="xjtu",
        search_settings={"engine": "auto", "endpoint": "", "limit": 3},
    )
    thread.search_client = RaisingSearch()
    failed, disabled = [], []
    thread.failed.connect(failed.append)
    thread.webSearchDisabled.connect(disabled.append)
    thread.run()
    self.assertFalse(failed)
    self.assertEqual(disabled[0].message, "均要求人机验证")
```

- [ ] **Step 2: 运行测试确认显式验证仍会关闭开关**

Run: `QT_QPA_PLATFORM=offscreen uv run --frozen python -m unittest test.app.test_notice_search_ui.AIInterfaceTest.test_explicit_verification_fails_request_without_disabling_saved_search test.app.test_notice_search_ui.AIInterfaceTest.test_aggregate_verification_emits_disable_signal -v`

Expected: FAIL，当前基类异常全部发往 `webSearchDisabled`。

- [ ] **Step 3: 按子类优先级拆分异常处理**

```python
except SearchAllSourcesVerificationRequired as error:
    if self.cancel_event.is_set():
        self.cancelled.emit()
    else:
        self.webSearchDisabled.emit(_AIRequestFailure(str(error), self.session_id))
except SearchHumanVerificationRequired as error:
    if self.cancel_event.is_set():
        self.cancelled.emit()
    else:
        self.failed.emit(_AIRequestFailure(str(error), self.session_id))
```

- [ ] **Step 4: 运行线程和 UI 保存回归**

Run: `QT_QPA_PLATFORM=offscreen uv run --frozen python -m unittest test.app.test_notice_search_ui.AIInterfaceTest.test_explicit_verification_fails_request_without_disabling_saved_search test.app.test_notice_search_ui.AIInterfaceTest.test_aggregate_verification_emits_disable_signal test.app.test_notice_search_ui.AIInterfaceTest.test_direct_search_choice_persists_and_captcha_turns_capability_off -v`

Expected: PASS；将旧的“显式 CAPTCHA 关闭开关”测试改为只验证聚合异常关闭并持久化。

- [ ] **Step 5: 提交验证异常分流**

```bash
git commit --only app/AIInterface.py test/app/test_notice_search_ui.py -m "fix(search): 仅在自动全源验证时关闭搜索"
```

### Task 9: 更新九项 UI 目录和端点持久化回归

**Files:**
- Modify: `app/AIInterface.py:374-405,749-868`
- Modify: `test/app/test_notice_search_ui.py:520-620`

**Interfaces:**
- Consumes: `SEARCH_ENGINES` 的稳定顺序。
- Preserves: `_currentSearchEngine() -> str`、`_persistSearchPreferences()`、`_profileFromForm() -> AIProfile`。

- [ ] **Step 1: 写九项标签顺序、默认值、显隐和往返测试**

```python
def test_search_catalog_labels_order_and_endpoint_visibility(self):
    widget = self.create_interfaces()
    labels = [widget.searchEngineCombo.itemText(index) for index in range(widget.searchEngineCombo.count())]
    self.assertEqual(labels, [
        "自动（直连推荐）", "百度（直连）", "Bing（直连）", "Google（需要代理）",
        "搜狗（直连）", "360 搜索（直连）", "神马（直连）",
        "DuckDuckGo（需要代理）", "SearXNG（自托管）",
    ])
    widget.capabilityChecks["web_search"].setChecked(True)
    for index in range(widget.searchEngineCombo.count() - 1):
        widget.searchEngineCombo.setCurrentIndex(index)
        self.assertTrue(widget.searchEndpointEdit.isHidden())
        self.assertFalse(widget.searchEndpointEdit.isEnabled())
    widget.searchEngineCombo.setCurrentIndex(widget.searchEngineCombo.count() - 1)
    self.assertFalse(widget.searchEndpointEdit.isHidden())
    self.assertTrue(widget.searchEndpointEdit.isEnabled())
```

- [ ] **Step 2: 运行测试确认目录数量和标签失败**

Run: `QT_QPA_PLATFORM=offscreen uv run --frozen python -m unittest test.app.test_notice_search_ui.AIInterfaceTest.test_search_catalog_labels_order_and_endpoint_visibility -v`

Expected: 在 Task 1 前 FAIL；Task 1 后标签通过，端点显隐仍需覆盖全部内置项。

- [ ] **Step 3: 保持控件结构，只收紧 SearXNG 专属端点语义**

```python
needs_searxng = self._currentSearchEngine() == "searxng"
self.searchEndpointEdit.setVisible(needs_searxng)
self.searchEndpointEdit.setEnabled(enabled and needs_searxng)
endpoint = self.searchEndpointEdit.text().strip() if needs_searxng else ""
```

在持久化测试中依次保存 `google`、`duckduckgo`、`searxng`，重建界面后断言引擎 ID 原样恢复，前两者 endpoint 为空，SearXNG endpoint 规范化保留。

- [ ] **Step 4: 运行 AI UI 模块测试**

Run: `QT_QPA_PLATFORM=offscreen uv run --frozen python -m unittest test.app.test_notice_search_ui.AIInterfaceTest -v`

Expected: PASS，无需代理控件或说明面板。

- [ ] **Step 5: 提交 UI 目录和持久化**

```bash
git commit --only app/AIInterface.py test/app/test_notice_search_ui.py -m "feat(search): 展示九项搜索来源"
```

### Task 10: 全量验证、候选隔离和双 Agent 评审

**Files:**
- Verify: `ai_assistant/web_search.py`
- Verify: `ai_assistant/config.py`
- Verify: `ai_assistant/__init__.py`
- Verify: `app/AIInterface.py`
- Verify: `test/ai_assistant/fixtures/*.html`
- Verify: `test/ai_assistant/test_ai_features.py`
- Verify: `test/app/test_notice_search_ui.py`
- Local only: `cooperate.md`

**Interfaces:**
- Produces: 可从 `origin/main` 重建的搜索专项 commit 集合和冻结候选哈希。
- Produces: Claude 可独立复验的 handoff 证据。

- [ ] **Step 1: 扫描应用代理、付费 API、占位符和不安全请求参数**

Run: `rg -n "proxies|proxy[_-]|api[_-]?key|TO[D]O|TB[D]|allow_redirects=True" ai_assistant/web_search.py app/AIInterface.py test/ai_assistant test/app/test_notice_search_ui.py`

Expected: 没有新增应用代理、搜索 API Key、占位符或自动跟随重定向；测试文字命中需逐条说明。

- [ ] **Step 2: 运行格式和聚焦测试**

Run: `git diff --check`

Expected: PASS。

Run: `uv run --frozen python -m unittest test.ai_assistant.test_ai_features.WebSearchTest test.ai_assistant.test_ai_features.IndependentProtocolConfigTest -v`

Expected: PASS。

Run: `QT_QPA_PLATFORM=offscreen uv run --frozen python -m unittest test.app.test_notice_search_ui.AIInterfaceTest -v`

Expected: PASS。

- [ ] **Step 3: 运行完整测试发现**

Run: `QT_QPA_PLATFORM=offscreen uv run --frozen python -m unittest discover -s test -t . -p "*.py" -v`

Expected: PASS；如存在与搜索无关的既有失败，记录完整测试名和错误，不修改无关模块掩盖失败。

- [ ] **Step 4: 审核每个搜索 commit 的文件边界**

Run: `git log --oneline --decorate origin/main..HEAD`

Expected: 每个搜索 commit 使用中文正文，设计、计划和各逻辑功能独立。

Run: `git diff --name-only origin/main...HEAD`

Expected: 搜索候选不含 `.github/workflows/pr-tests.yml`、`.github/pull_request_template.md`、`docs/development/testing.md`、`pyproject.toml`、`uv.lock`、`schedule/lesson.py` 或 `test/schedule/test_lesson.py`。

- [ ] **Step 5: 从 `origin/main` 重建专属分支并冻结候选**

在不改动当前用户 staged 内容的前提下，新建独立 worktree/分支并只 cherry-pick 以下搜索 commit：设计文档、实施计划、Task 1–9 的搜索 commit。运行同一聚焦和全量测试后，记录 `git rev-parse HEAD` 与 `git diff origin/main...HEAD` 的 SHA-256。

- [ ] **Step 6: 通过 `cooperate.md` handoff Claude 独立评审**

handoff 必须登记候选分支、HEAD、完整搜索 diff 哈希、fixture 来源、聚焦/全量测试命令和结果、安全不变量，以及 `cooperate.md` 不进入候选的证明。Claude 所有 finding 终态、终态 decision 和双方 closing ACK 完成前，不宣称评审闭环。

- [ ] **Step 7: 向用户展示 PR 候选并单独申请外部操作确认**

展示目标上游仓库、fork 账号、搜索分支、完整文件列表、完整 diff、精确 PR 标题和正文。获得用户对 fork、push、创建该 PR 的单独确认后才执行；不自动合并，等待他人 review。

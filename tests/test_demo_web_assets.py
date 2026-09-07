from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "apps/demo_web/index.html"
APP = ROOT / "apps/demo_web/app.js"
STYLES = ROOT / "apps/demo_web/styles.css"
API_APP = ROOT / "services/api/app.py"
API_ROUTES = ROOT / "services/api/routes.py"
DOCKERFILE = ROOT / "Dockerfile"


# ADD 2026-09-07: Demo Web가 combined inspection과 Live Monitor 진입점을 제공하게 한다.
def test_demo_web_declares_combined_inspection_workspace() -> None:
    html = INDEX.read_text(encoding="utf-8")
    assert 'id="image-input"' in html
    assert 'id="inspect-button"' in html
    assert 'id="decision-disposition"' in html
    assert 'id="defect-list"' in html
    assert 'id="history-list"' in html
    assert 'href="/live/"' in html


# ADD 2026-09-07: Browser upload가 backend multipart contract를 사용하게 한다.
def test_demo_web_uses_combined_inspection_api_contract() -> None:
    app = APP.read_text(encoding="utf-8")
    routes = API_ROUTES.read_text(encoding="utf-8")
    assert 'combined:"/v1/combined-inspections"' in app
    assert 'body.append("image",selectedFile)' in app
    assert 'method:"POST"' in app
    assert '@router.post("/v1/combined-inspections"' in routes


# ADD 2026-09-07: Demo Web가 compact YOLO box/mask summary만 사용하게 한다.
def test_demo_web_renders_compact_known_defect_contract() -> None:
    app = APP.read_text(encoding="utf-8")
    assert "instance.box.x_min" in app
    assert "instance.mask.area_ratio" in app
    assert "renderBoxes(payload)" in app
    assert "raw_mask" not in app


# ADD 2026-09-07: API image가 Demo Web을 포함하고 same-origin path로 mount하게 한다.
def test_demo_web_is_packaged_and_mounted_same_origin() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    api_app = API_APP.read_text(encoding="utf-8")
    assert "COPY apps/demo_web ./apps/demo_web" in dockerfile
    assert "DEFAULT_DEMO_WEB_DIR" in api_app
    assert '"/demo"' in api_app
    assert 'name="demo-web"' in api_app


# ADD 2026-09-07: responsive decision UI → MODIFY 2026-09-07: hidden empty-state 계약 추가.
def test_demo_web_styles_include_responsive_decision_ui() -> None:
    styles = STYLES.read_text(encoding="utf-8")
    assert ".workspace" in styles
    assert ".pill.accept" in styles
    assert ".pill.reject" in styles
    assert ".empty[hidden]{display:none}" in styles
    assert "@media(max-width:980px)" in styles

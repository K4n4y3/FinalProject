import pytest
from fastapi.testclient import TestClient

from menu.api.app import create_app
from menu.service import MenuService
from menu.settings import Settings
from tests.conftest import HAS_LLM

MOJITO = ["2 oz white rum", "1 lime, cut into wedges", "10 fresh mint leaves", "2 tsp sugar", "club soda"]


def _client(bundle_dir, **kw) -> TestClient:
    settings = Settings(artifacts_dir=bundle_dir, enable_dense=False, log_json=False, **kw)
    service = MenuService(settings).load(background_llm=False)
    return TestClient(create_app(settings, service))


@pytest.fixture(scope="module")
def client(bundle_dir):
    with _client(bundle_dir, enable_llm=HAS_LLM) as c:
        yield c


@pytest.fixture(scope="module")
def client_no_llm(bundle_dir):
    with _client(bundle_dir, enable_llm=False, mount_ui=False) as c:
        yield c


def test_health_and_model_info(client):
    h = client.get("/health").json()
    assert h["components"]["classifier"] and h["bundle_version"] == "test-0.0.1"
    info = client.get("/v1/model").json()
    assert info["bundle_version"] == "test-0.0.1"
    assert "ingredients" in info["input_schema"]["properties"]
    assert info["components"]["data"]["leakage_check"]["group"] == 0


def test_classify_contract(client):
    r = client.post("/v1/classify", json={"ingredients": MOJITO, "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dish_type"] == "Drink"
    assert len(body["top"]) == 3 and abs(sum(x["probability"] for x in body["top"])) <= 1.0001
    assert body["meta"]["model_version"] == "test-0.0.1" and body["meta"]["latency_ms"] >= 0
    assert r.headers["X-Request-ID"]


def test_classify_accepts_free_text(client):
    r = client.post("/v1/classify", json={"ingredients": "gin; lime juice\nclub soda"})
    assert r.status_code == 200
    assert r.json()["normalized_ingredients"] == ["gin", "lime juice", "club soda"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"ingredients": []},
        {"ingredients": " ; ; "},
        {"ingredients": ["123", "!!"]},
        {"ingredients": ["salt"], "top_k": 0},
    ],
)
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/v1/classify", json=payload).status_code == 422


def test_unknown_words_produce_a_warning(client):
    body = client.post("/v1/classify", json={"ingredients": ["мята", "лайм", "ром"]}).json()
    assert any("unknown" in w for w in body["meta"]["warnings"])


@pytest.mark.skipif(not HAS_LLM, reason="needs the llm extra")
@pytest.mark.parametrize("mode", ["auto", "lora", "lora_rag", "zero_shot", "zero_shot_rag"])
def test_generate_every_mode(client, mode):
    r = client.post("/v1/generate", json={"ingredients": MOJITO, "mode": mode})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["degraded"] is False
    assert isinstance(body["title"], str)
    expected_ctx = 3 if mode in ("auto", "lora_rag", "zero_shot_rag") else 0
    assert len(body["context"]) == expected_ctx


def test_menu_search_chat(client):
    m = client.post("/v1/menu", json={"ingredients": MOJITO}).json()
    assert m["classification"]["dish_type"] == "Drink" and "title" in m["generation"]
    s = client.post("/v1/search", json={"query": "gin lime soda", "k": 4}).json()
    assert len(s["hits"]) == 4 and s["hits"][0]["score"] >= s["hits"][-1]["score"]
    c = client.post("/v1/chat", json={"message": "I have gin and lime, what can I make?"}).json()
    assert c["sources"] and isinstance(c["answer"], str) and c["suggested_dish_type"]


def test_metrics_and_monitoring(client):
    client.post("/v1/classify", json={"ingredients": MOJITO})
    text = client.get("/metrics").text
    assert "menu_requests_total" in text and "menu_prediction_confidence_bucket" in text
    assert 'menu_predicted_class_total{label="Drink"}' in text
    summary = client.get("/v1/monitoring").json()
    assert summary["window_requests"] >= 1 and "drift_alert" in summary


def test_ui_is_mounted(client):
    assert client.get("/", follow_redirects=False).headers["location"] == "/ui"
    assert client.get("/ui/").status_code == 200


def test_degraded_mode_without_llm(client_no_llm):
    h = client_no_llm.get("/health").json()
    assert h["status"] == "degraded" and "disabled" in h["errors"]["generator_llm"]
    g = client_no_llm.post("/v1/generate", json={"ingredients": MOJITO}).json()
    assert g["degraded"] is True and g["mode"] == "retrieval_fallback" and g["title"]
    c = client_no_llm.post("/v1/chat", json={"message": "gin and lime"}).json()
    assert c["degraded"] is True and c["grounded"] is True and "Try" in c["answer"]


def test_api_key(bundle_dir):
    with _client(bundle_dir, enable_llm=False, mount_ui=False, api_key="secret") as c:
        assert c.post("/v1/classify", json={"ingredients": MOJITO}).status_code == 401
        ok = c.post("/v1/classify", json={"ingredients": MOJITO}, headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
        assert c.get("/health").status_code == 200  # probes stay open

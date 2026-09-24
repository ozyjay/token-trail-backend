from contextlib import contextmanager

from fastapi.testclient import TestClient

from token_trail_backend.api import create_app
from token_trail_backend.config import MAX_PROMPT_CHARS, Settings


class FakeEngine:
    ready = True

    def close(self):
        pass

    def trace(self, request, cancel=None):
        return {"prompt": request.prompt, "model": request.model, "steps": []}


@contextmanager
def client(engine=None):
    with TestClient(
        create_app(Settings(rate_limit_per_minute=100), engine or FakeEngine(), load_model=False)
    ) as test_client:
        yield test_client


def test_valid_trace():
    with client() as api:
        response = api.post("/v1/traces", json={"prompt": "Hello"})
        assert response.status_code == 200
        assert response.json()["prompt"] == "Hello"


def test_request_bounds():
    with client() as api:
        for payload in (
            {"prompt": "x" * (MAX_PROMPT_CHARS + 1)},
            {"prompt": "x", "max_new_tokens": 33},
            {"prompt": "x", "top_k": 11},
            {"prompt": "x", "model": "other"},
            {"prompt": "x", "unexpected": 1},
            {"prompt": "x", "max_new_tokens": "2"},
        ):
            assert api.post("/v1/traces", json=payload).status_code in (413, 422)


def test_payload_limit_and_malformed_json():
    with client() as api:
        assert api.post("/v1/traces", content=b"x" * 4097).status_code == 413
        assert (
            api.post(
                "/v1/traces",
                content=b'{"prompt":"\\ud800"}',
                headers={"Content-Type": "application/json"},
            ).status_code
            == 422
        )
        assert (
            api.post(
                "/v1/traces", content=b"{", headers={"Content-Type": "application/json"}
            ).status_code
            == 422
        )
        invalid_utf8 = api.post(
            "/v1/traces", content=b"\xff", headers={"Content-Type": "application/json"}
        )
        assert invalid_utf8.status_code == 400
        assert invalid_utf8.json()["error"]["code"] == "invalid_request"


def test_health_and_model_allowlist():
    with client() as api:
        assert api.get("/health").json() == {"status": "ready", "ready": True}
        models = api.get("/v1/models").json()["models"]
        assert len(models) == 1 and models[0]["alias"] == "qwen2.5-0.5b"
    engine = FakeEngine()
    engine.ready = False
    with client(engine) as api:
        assert api.get("/health").json()["ready"] is False
        assert api.post("/v1/traces", json={"prompt": "x"}).status_code == 503


def test_model_load_failure_does_not_fall_back():
    class BrokenEngine(FakeEngine):
        ready = False

        def load(self):
            raise RuntimeError("pinned model unavailable")

    try:
        with TestClient(create_app(engine=BrokenEngine())):
            pass
    except RuntimeError as exc:
        assert "pinned model unavailable" in str(exc)
    else:
        raise AssertionError("startup silently substituted a model")


def test_rate_limit():
    with TestClient(
        create_app(Settings(rate_limit_per_minute=1), FakeEngine(), load_model=False)
    ) as api:
        assert api.post("/v1/traces", json={"prompt": "a"}).status_code == 200
        assert api.post("/v1/traces", json={"prompt": "b"}).status_code == 429


def test_cors():
    with client() as api:
        allowed = api.options(
            "/v1/traces",
            headers={"Origin": "https://crunchycodes.net", "Access-Control-Request-Method": "POST"},
        )
        denied = api.options(
            "/v1/traces",
            headers={"Origin": "https://other.example", "Access-Control-Request-Method": "POST"},
        )
        assert allowed.headers.get("access-control-allow-origin") == "https://crunchycodes.net"
        assert "access-control-allow-origin" not in denied.headers

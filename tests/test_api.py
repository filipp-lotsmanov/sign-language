"""
API-level tests.

These run without model weights: the app is designed to start degraded, and that
degraded behaviour is itself worth pinning.
"""

import pytest
from fastapi.testclient import TestClient

from src.backend.api import routes


@pytest.fixture
def client():
    with TestClient(routes.app, raise_server_exceptions=False) as test_client:
        yield test_client
    routes.session_manager.sessions.clear()


class TestHealth:
    def test_health_reports_component_state(self, client) -> None:
        body = client.get("/health").json()
        assert set(body) >= {"status", "static_model", "dynamic_model", "hand_capture"}

    def test_status_reflects_missing_components(self, client) -> None:
        response = client.get("/health")
        body = response.json()
        if not body["hand_capture"]:
            # Without hand detection nothing can be served, so this must not
            # report healthy to a load balancer.
            assert body["status"] == "unhealthy"
            assert response.status_code == 503
        elif not (body["static_model"] and body["dynamic_model"]):
            assert body["status"] == "degraded"
        else:
            assert body["status"] == "healthy"


class TestSessionEndpoints:
    def test_create_session_returns_a_server_generated_id(self, client) -> None:
        body = client.post("/api/session/new").json()
        assert body["session_id"]
        assert body["mode"] == "sequential"

    def test_create_session_rejects_an_unknown_mode(self, client) -> None:
        response = client.post("/api/session/new", params={"mode": "banana"})
        assert response.status_code == 400

    @pytest.mark.parametrize("mode", ["sequential", "random", "sentence"])
    def test_create_session_accepts_valid_modes(self, client, mode) -> None:
        assert client.post("/api/session/new", params={"mode": mode}).status_code == 200

    def test_sessions_are_independent(self, client) -> None:
        first = client.post("/api/session/new").json()["session_id"]
        second = client.post("/api/session/new").json()["session_id"]
        assert first != second

    def test_get_unknown_session_is_404(self, client) -> None:
        assert client.get("/api/session/no-such-session").status_code == 404

    def test_get_known_session_round_trips(self, client) -> None:
        session_id = client.post("/api/session/new").json()["session_id"]
        body = client.get(f"/api/session/{session_id}").json()
        assert body["session_id"] == session_id

    def test_delete_removes_the_session(self, client) -> None:
        session_id = client.post("/api/session/new").json()["session_id"]
        assert client.delete(f"/api/session/{session_id}").status_code == 200
        assert client.get(f"/api/session/{session_id}").status_code == 404

    def test_delete_unknown_session_is_404(self, client) -> None:
        assert client.delete("/api/session/no-such-session").status_code == 404

    def test_mode_change_rejects_an_invalid_mode(self, client) -> None:
        session_id = client.post("/api/session/new").json()["session_id"]
        response = client.post(f"/api/session/{session_id}/mode", json={"mode": "banana"})
        assert response.status_code == 400

    def test_mode_change_applies(self, client) -> None:
        session_id = client.post("/api/session/new").json()["session_id"]
        response = client.post(f"/api/session/{session_id}/mode", json={"mode": "random"})
        assert response.status_code == 200
        assert response.json()["mode"] == "random"

    def test_mode_change_on_unknown_session_is_404(self, client) -> None:
        response = client.post("/api/session/no-such-session/mode", json={"mode": "random"})
        assert response.status_code == 404


class TestStaticRoutes:
    def test_index_is_served(self, client) -> None:
        assert client.get("/").status_code == 200

    def test_favicon_does_not_error(self, client) -> None:
        # Regression guard: the missing-file branch referenced an unimported
        # `Response` and raised NameError.
        assert client.get("/favicon.ico").status_code in (200, 204)


class TestFrameDecoding:
    def test_garbage_payload_returns_none_instead_of_raising(self) -> None:
        import base64

        assert routes.decode_frame(base64.b64encode(b"not-an-image").decode()) is None

    def test_data_url_prefix_is_stripped(self) -> None:
        import base64

        import cv2
        import numpy as np

        buffer = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))[1]
        encoded = base64.b64encode(buffer.tobytes()).decode()
        frame = routes.decode_frame(f"data:image/jpeg;base64,{encoded}")
        assert frame is not None and frame.shape == (16, 16, 3)


class TestDynamicLetterRouting:
    @pytest.mark.parametrize("letter", ["J", "Z", "j", "z"])
    def test_dynamic_letters_are_detected(self, letter) -> None:
        assert routes.is_dynamic_letter(letter) is True

    @pytest.mark.parametrize("letter", ["A", "M", "Y", None, ""])
    def test_everything_else_is_static(self, letter) -> None:
        assert routes.is_dynamic_letter(letter) is False


class TestWebSocketProtocol:
    """
    Validate live WebSocket responses against the declared schemas.

    The schemas used to describe a protocol the server did not speak. Checking
    real responses against them here means they cannot drift again silently.
    """

    def test_response_matches_the_declared_schema(self, client) -> None:
        from src.backend.api.schemas import DetectionResponse

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start_recording"})
            DetectionResponse.model_validate(ws.receive_json())

    def test_every_displayed_message_carries_a_key(self, client) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start_recording"})
            assert ws.receive_json()["message_key"] == "recording_started"

            ws.send_json({"type": "skip"})
            skipped = ws.receive_json()
            assert skipped["skipped"] is True
            assert skipped["message_key"] == "skipped_letter"
            assert "letter" in skipped["message_args"]

    def test_sentence_flow_reports_keys(self, client) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "set_sentence", "sentence": "AB"})
            response = ws.receive_json()
            assert response["message_key"] == "sentence_set"
            assert response["target_sentence"] == "AB"

            ws.send_json({"type": "clear_sentence"})
            assert ws.receive_json()["message_key"] == "sentence_cleared"

    def test_empty_sentence_enters_free_mode(self, client) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "set_sentence", "sentence": ""})
            assert ws.receive_json()["message_key"] == "free_mode_on"

    def test_non_object_message_is_ignored(self, client) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json(["not", "an", "object"])
            ws.send_json({"type": "start_recording"})
            assert ws.receive_json()["message_key"] == "recording_started"

    def test_oversized_frame_closes_the_socket(self, client) -> None:
        from starlette.websockets import WebSocketDisconnect

        from src.backend.core.config import MAX_FRAME_BYTES

        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "frame", "frame": "A" * (MAX_FRAME_BYTES + 1)})
                ws.receive_json()
        assert excinfo.value.code == 1009

    def test_flooding_trips_the_rate_limiter(self, client) -> None:
        from starlette.websockets import WebSocketDisconnect

        from src.backend.core.config import MAX_MESSAGES_PER_SECOND

        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/ws") as ws:
                for _ in range(int(MAX_MESSAGES_PER_SECOND) * 10):
                    ws.send_json({"type": "start_recording"})
                    ws.receive_json()
        assert excinfo.value.code == 1008


class TestDynamicClasses:
    def test_classes_fall_back_to_the_dynamic_letters(self, tmp_path) -> None:
        # models/README.md advertises dynamic/classes.npy; it is now read.
        import numpy as np

        from src.backend.core.config import DYNAMIC_LETTERS
        from src.backend.detection.dynamic_detector import DynamicSignPredictor

        predictor = DynamicSignPredictor(str(tmp_path / "missing.pth"))
        assert predictor.classes == list(DYNAMIC_LETTERS)

        np.save(tmp_path / "classes.npy", np.array(["Z", "J"]))
        relabelled = DynamicSignPredictor(str(tmp_path / "missing.pth"))
        assert relabelled.classes == ["Z", "J"]

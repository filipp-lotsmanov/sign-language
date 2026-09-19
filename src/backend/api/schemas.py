"""
Pydantic schemas for the API and the WebSocket protocol.

These previously described a protocol the server did not speak: DetectionResponse
required `consecutive_matches` and `matches_needed`, which nothing in the
codebase ever produced, and no WebSocket response was ever built from the model.
The definitions below match what the server actually sends, and
tests/test_api.py validates live WebSocket responses against them so they cannot
drift again.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

VALID_MODES = ("sequential", "random", "sentence")


class PredictionResult(BaseModel):
    """A single model prediction."""

    predicted_class: str
    confidence: float
    # Only the REST/debug paths carry the full distribution; the WebSocket omits
    # it to keep per-frame payloads small.
    all_probabilities: Optional[Dict[str, float]] = None


class Progress(BaseModel):
    """The progress block attached to every WebSocket response."""

    model_config = ConfigDict(extra="allow")

    current_letter: Optional[str] = None
    total_correct: int
    total_attempts: int
    accuracy: float
    completed_letters: List[str]
    attempt_count: int
    time_remaining: float
    tutorial_url: Optional[str] = None
    is_recording: bool
    mode: str
    target_sentence: str
    recognized_sentence: str


class ClientMessage(BaseModel):
    """
    Inbound WebSocket message.

    `session_id` is only ever used to resume an existing session. The server
    never adopts a client-supplied ID for a new session.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "frame"
    session_id: Optional[str] = None
    frame: Optional[str] = None  # base64 JPEG, optionally a data: URL
    sentence: Optional[str] = None  # for type="set_sentence"


class DetectionResponse(BaseModel):
    """
    Outbound WebSocket message.

    Every field beyond `session_id` and `progress` is optional: the server sends
    different subsets depending on whether it is previewing, recording, or
    reporting a finished attempt.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    progress: Progress

    hand_detected: Optional[bool] = None
    recording: Optional[bool] = None
    current_letter: Optional[str] = None
    is_dynamic: Optional[bool] = None

    prediction: Optional[PredictionResult] = None
    current_prediction: Optional[str] = None
    confidence: Optional[float] = None
    buffer_progress: Optional[float] = None

    match: Optional[bool] = None
    success: Optional[bool] = None
    timeout: Optional[bool] = None
    skipped: Optional[bool] = None

    show_hint: Optional[bool] = None
    hint_message: Optional[str] = None
    hint_key: Optional[str] = None

    message: Optional[str] = None
    # Stable identifier for `message`, so the client can localise it instead of
    # displaying the server's English string.
    message_key: Optional[str] = None
    message_args: Optional[Dict[str, Any]] = None

    mode: Optional[str] = None
    target_sentence: Optional[str] = None
    recognized_sentence: Optional[str] = None


class SessionInfo(BaseModel):
    """Session information returned by the REST endpoints."""

    session_id: str
    current_letter: Optional[str] = None
    total_correct: int
    total_attempts: int
    accuracy: float
    completed_letters: List[str]
    mode: str = "sequential"
    target_sentence: str = ""
    recognized_sentence: str = ""


class ModeChangeRequest(BaseModel):
    """Request to change letter sequence mode. One of VALID_MODES."""

    mode: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None

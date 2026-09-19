"""
FastAPI routes for sign language detection API - Recording-based workflow.
"""

import asyncio
import base64
import contextlib
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.schemas import SessionInfo, ModeChangeRequest
from src.backend.core.session_manager import SessionManager
from src.backend.core.config import (
    CORS_ALLOWED_ORIGINS,
    DYNAMIC_LETTERS,
    DYNAMIC_MODEL_PATH,
    MAX_FRAME_BYTES,
    MAX_MESSAGES_PER_SECOND,
    SESSION_SWEEP_INTERVAL,
    STATIC_MODEL_PATH,
)
from src.backend.detection.hand_capture import HandCapture
from src.backend.detection.static_detector import StaticSignPredictor
from src.backend.detection.dynamic_detector import DynamicSignPredictor

logger = logging.getLogger(__name__)

# Module-level state. This is per-process, so the app must run with a single
# worker; see main.py for why and what it would take to lift that.
session_manager = SessionManager()
static_predictor = None
dynamic_predictor = None
hand_capture = None

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ASSETS_DIR = PROJECT_ROOT / "src" / "assets"


def _load_models() -> None:
    """Load the hand landmarker and both predictors. Failures degrade, not crash."""
    global static_predictor, dynamic_predictor, hand_capture

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting server on %s", device)

    try:
        hand_capture = HandCapture()
        logger.info("Hand capture initialized")
    except Exception as e:
        hand_capture = None
        logger.error("Failed to initialize hand capture: %s", e)

    if STATIC_MODEL_PATH.exists():
        try:
            static_predictor = StaticSignPredictor(str(STATIC_MODEL_PATH), device=device)
            logger.info("Static model loaded: %s", STATIC_MODEL_PATH.name)
        except Exception as e:
            logger.error("Failed to load static model: %s", e)
    else:
        logger.warning("Static model not found: %s", STATIC_MODEL_PATH)

    if DYNAMIC_MODEL_PATH.exists():
        try:
            dynamic_predictor = DynamicSignPredictor(str(DYNAMIC_MODEL_PATH), device=device)
            logger.info("Dynamic model loaded: %s", DYNAMIC_MODEL_PATH.name)
        except Exception as e:
            logger.error("Failed to load dynamic model: %s", e)
    else:
        logger.warning("Dynamic model not found: %s", DYNAMIC_MODEL_PATH)


async def _sweep_sessions_forever() -> None:
    """Periodically drop idle sessions so memory does not grow without bound."""
    while True:
        await asyncio.sleep(SESSION_SWEEP_INTERVAL)
        try:
            session_manager.cleanup_expired()
        except Exception:
            logger.exception("Session sweep failed")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown. Replaces the deprecated @app.on_event hooks."""
    _load_models()

    sweeper = asyncio.create_task(_sweep_sessions_forever())
    try:
        yield
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper
        if hand_capture is not None:
            with contextlib.suppress(Exception):
                hand_capture.close()


app = FastAPI(title="Sign Language Learning API", lifespan=lifespan)

# A wildcard origin combined with allow_credentials=True is rejected by browsers
# and unsafe in production. Origins come from CORS_ALLOWED_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Sign Language Learning</h1><p>Frontend not found</p>")


@app.get("/health")
async def health_check(response: Response):
    """
    Health check endpoint.

    Reports unhealthy when hand detection is unavailable, since no prediction
    can be served without it, and degraded when a model failed to load.
    """
    components = {
        "static_model": static_predictor is not None,
        "dynamic_model": dynamic_predictor is not None,
        "hand_capture": hand_capture is not None,
    }

    if not components["hand_capture"]:
        status = "unhealthy"
    elif not all(components.values()):
        status = "degraded"
    else:
        status = "healthy"

    if status == "unhealthy":
        response.status_code = 503

    return {"status": status, **components}


VALID_MODES = ("sequential", "random", "sentence")


@app.post("/api/session/new", response_model=SessionInfo)
async def create_session(mode: str = "sequential"):
    """Create a new learning session. The ID is always generated server-side."""
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: {', '.join(VALID_MODES)}",
        )

    session = session_manager.create_session(mode=mode)
    progress = session.get_progress()

    return SessionInfo(
        session_id=session.id,
        current_letter=session.current_letter,
        total_correct=session.total_correct,
        total_attempts=session.total_attempts,
        accuracy=progress["accuracy"],
        completed_letters=session.completed_letters,
        mode=session.mode,
        target_sentence=session.target_sentence,
        recognized_sentence=session.recognized_sentence,
    )


@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    progress = session.get_progress()
    return SessionInfo(
        session_id=session.id,
        current_letter=session.current_letter,
        total_correct=session.total_correct,
        total_attempts=session.total_attempts,
        accuracy=progress["accuracy"],
        completed_letters=session.completed_letters,
        mode=session.mode,
        target_sentence=session.target_sentence,
        recognized_sentence=session.recognized_sentence,
    )


@app.post("/api/session/{session_id}/mode")
async def change_mode(session_id: str, request: ModeChangeRequest):
    """Change letter sequence mode for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session.set_mode(request.mode)
        return {
            "status": "success",
            "mode": session.mode,
            "message": f"Mode changed to {request.mode}",
            "progress": session.get_progress(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    removed = session_manager.remove_session(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


def decode_frame(frame_base64: str) -> Optional[np.ndarray]:
    """Decode a base64 frame to a BGR array, or None if it is not a valid image."""
    # Remove data URL prefix if present
    if "," in frame_base64:
        frame_base64 = frame_base64.split(",")[1]

    img_bytes = base64.b64decode(frame_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    if nparr.size == 0:
        return None

    # imdecode returns None for corrupt or non-image payloads.
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def is_dynamic_letter(letter: Optional[str]) -> bool:
    """Check if letter requires dynamic detection (LSTM)."""
    if not letter:
        return False
    return letter.upper() in DYNAMIC_LETTERS


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sign language detection.

    New workflow:
    1. Client connects and gets session
    2. Show target letter and GIF
    3. Client starts camera (preview only)
    4. User clicks "Record" button
    5. Server receives 'start_recording' command
    6. Collect frames for 3-5 seconds
    7. Server processes recording and returns result
    8. Show success/failure, move to next letter

    Messages:
    - Client -> Server: {"type": "start_recording", "session_id": "..."}
    - Client -> Server: {"type": "frame", "frame": "base64...", "session_id": "..."}
    - Client -> Server: {"type": "stop_recording", "session_id": "..."}
    - Server -> Client: DetectionResponse
    """
    await websocket.accept()
    logger.info("WebSocket connected")

    session = None
    is_dynamic = False
    # Sliding window of recent message timestamps, for rate limiting.
    recent_messages: deque[float] = deque(maxlen=int(MAX_MESSAGES_PER_SECOND) * 2 or 2)

    try:
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error("Error receiving message: %s", e)
                break

            if not isinstance(data, dict):
                logger.warning("Ignoring non-object websocket message")
                continue

            # Rate limit: if the window is full and spans under a second, the
            # client is sending faster than MAX_MESSAGES_PER_SECOND.
            now = time.monotonic()
            if len(recent_messages) == recent_messages.maxlen:
                window = now - recent_messages[0]
                if window < len(recent_messages) / MAX_MESSAGES_PER_SECOND:
                    logger.warning("Rate limit exceeded; closing websocket")
                    await websocket.close(code=1008, reason="Rate limit exceeded")
                    return
            recent_messages.append(now)

            message_type = data.get("type", "frame")

            # Resume the named session if it exists, otherwise start a new one
            # with a server-generated ID. A client-supplied ID is only ever used
            # to look up an existing session, never to name a new one.
            session_id = data.get("session_id")
            if not session:
                session = session_manager.get_or_create(session_id)
                is_dynamic = is_dynamic_letter(session.current_letter)
            else:
                session.touch()

            # Handle different message types
            if message_type == "start_recording":
                # Start recording
                session.start_recording()

                # If dynamic, start collecting frames
                if is_dynamic and dynamic_predictor:
                    dynamic_predictor.start_collecting()

                response = {
                    "session_id": session.id,
                    "recording": True,
                    "current_letter": session.current_letter,
                    "is_dynamic": is_dynamic,
                    "message": "Recording started",
                    "message_key": "recording_started",
                    # Every response carries progress, so the client never has
                    # to guess and DetectionResponse can require it.
                    "progress": session.get_progress(),
                }
                await websocket.send_json(response)
                continue

            elif message_type == "stop_recording":
                # Force stop recording
                if session.is_recording:
                    result = session.finish_recording()

                    if is_dynamic and dynamic_predictor:
                        dynamic_predictor.stop_collecting()

                    response = {
                        "session_id": session.id,
                        "recording": False,
                        **result,
                        "progress": session.get_progress(),
                    }
                    await websocket.send_json(response)
                continue

            elif message_type == "set_sentence":
                target_sentence = data.get("sentence", "")
                session.set_mode("sentence")
                session.set_target_sentence(target_sentence)

                # The new sentence may start with J or Z, which must go to the LSTM.
                is_dynamic = is_dynamic_letter(session.current_letter)

                response = {
                    "session_id": session.id,
                    "mode": session.mode,
                    "target_sentence": session.target_sentence,
                    "recognized_sentence": session.recognized_sentence,
                    "message": "Target sentence set"
                    if target_sentence
                    else "Free sign mode activated",
                    "message_key": "sentence_set" if target_sentence else "free_mode_on",
                    "progress": session.get_progress(),
                }
                await websocket.send_json(response)
                continue

            elif message_type == "clear_sentence":
                session.clear_recognized()

                # Restarting the sentence resets the target letter.
                is_dynamic = is_dynamic_letter(session.current_letter)

                response = {
                    "session_id": session.id,
                    "recognized_sentence": session.recognized_sentence,
                    "message": "Recognized sentence cleared",
                    "message_key": "sentence_cleared",
                    "progress": session.get_progress(),
                }
                await websocket.send_json(response)
                continue

            elif message_type == "skip":
                # Skip current letter and move to next one
                logger.info("Skipping letter: %s", session.current_letter)

                # Stop recording if active
                if session.is_recording:
                    session.is_recording = False
                    if is_dynamic and dynamic_predictor:
                        dynamic_predictor.stop_collecting()
                        dynamic_predictor.clear_buffer()

                # Skip to next letter
                result = session.skip_letter()

                # Update is_dynamic flag for new letter
                is_dynamic = is_dynamic_letter(session.current_letter)

                # Send response with new letter - same structure as timeout/success
                response = {
                    "session_id": session.id,
                    "hand_detected": False,
                    "recording": False,
                    "match": result.get("match", False),
                    "success": result.get("success", False),
                    "timeout": result.get("timeout", False),
                    "skipped": result.get("skipped", True),
                    "show_hint": result.get("show_hint", False),
                    "hint_message": result.get("hint_message", ""),
                    "hint_key": result.get("hint_key", ""),
                    "message": result.get("message", "Letter skipped"),
                    "message_key": result.get("message_key", "skipped_letter"),
                    "message_args": result.get("message_args", {}),
                    "progress": session.get_progress(),
                }
                await websocket.send_json(response)
                continue

            elif message_type == "frame":
                # Process frame
                frame_base64 = data.get("frame")
                if not frame_base64 or not isinstance(frame_base64, str):
                    continue

                # Reject oversized payloads before allocating a decode buffer.
                if len(frame_base64) > MAX_FRAME_BYTES:
                    logger.warning(
                        "Frame payload of %d bytes exceeds the %d byte limit",
                        len(frame_base64),
                        MAX_FRAME_BYTES,
                    )
                    await websocket.close(code=1009, reason="Frame too large")
                    return

                # The hand landmarker failed to initialize at startup; without it
                # no frame can be processed, so report once and keep the socket open.
                if hand_capture is None:
                    await websocket.send_json(
                        {
                            "session_id": session.id,
                            "hand_detected": False,
                            "recording": False,
                            "message": "Hand detection unavailable on the server",
                            "message_key": "detection_unavailable",
                            "progress": session.get_progress(),
                        }
                    )
                    session.is_recording = False
                    continue

                # Decode frame
                try:
                    frame = decode_frame(frame_base64)
                except Exception as e:
                    logger.error("Frame decode error: %s", e)
                    continue

                if frame is None:
                    logger.warning("Received an undecodable frame; skipping")
                    continue

                # Extract landmarks (raw; the predictors normalize internally)
                landmarks = hand_capture.extract_landmarks(frame)

                if landmarks is None:
                    # No hand detected
                    if session.is_recording:
                        # During recording, add a "no hand" entry
                        response = {
                            "session_id": session.id,
                            "hand_detected": False,
                            "recording": True,
                            "message": "No hand detected",
                            "message_key": "no_hand",
                            "progress": session.get_progress(),
                        }
                    else:
                        # Just preview
                        response = {
                            "session_id": session.id,
                            "hand_detected": False,
                            "recording": False,
                            "current_letter": session.current_letter,
                            "progress": session.get_progress(),
                        }

                    await websocket.send_json(response)
                    continue

                # If not recording, just send preview status
                if not session.is_recording:
                    response = {
                        "session_id": session.id,
                        "hand_detected": True,
                        "recording": False,
                        "current_letter": session.current_letter,
                        "message": "Hand detected - Click Record to start",
                        "message_key": "hand_detected_ready",
                        "progress": session.get_progress(),
                    }
                    await websocket.send_json(response)
                    continue

                # Recording in progress
                if is_dynamic:
                    # Dynamic letter (J, Z) - collect frames for LSTM
                    if dynamic_predictor:
                        is_ready = dynamic_predictor.add_frame(landmarks)

                        if is_ready:
                            # Buffer full, make prediction
                            try:
                                pred_result = dynamic_predictor.predict()
                            except Exception as e:
                                logger.error("Dynamic prediction error: %s", e)
                                pred_result = None

                            if pred_result:
                                # Add to session
                                result = session.add_prediction(
                                    pred_result["predicted_class"], pred_result["confidence"]
                                )

                                # Finish recording if not already finished
                                if not result:
                                    result = session.finish_recording()

                                dynamic_predictor.clear_buffer()
                                is_dynamic = is_dynamic_letter(session.current_letter)

                                response = {
                                    "session_id": session.id,
                                    "hand_detected": True,
                                    "recording": False,
                                    "prediction": {
                                        "predicted_class": pred_result["predicted_class"],
                                        "confidence": pred_result["confidence"],
                                    },
                                    "match": result.get("match", False),
                                    "success": result.get("success", False),
                                    "timeout": result.get("timeout", False),
                                    "message": result.get("message", ""),
                                    "message_key": result.get("message_key", ""),
                                    "message_args": result.get("message_args", {}),
                                    "show_hint": result.get("show_hint", False),
                                    "hint_message": result.get("hint_message", ""),
                                    "hint_key": result.get("hint_key", ""),
                                    "progress": session.get_progress(),
                                }
                                await websocket.send_json(response)
                        else:
                            # Still collecting
                            progress_pct = dynamic_predictor.get_buffer_progress()
                            response = {
                                "session_id": session.id,
                                "hand_detected": True,
                                "recording": True,
                                "buffer_progress": progress_pct,
                                "message": f"Collecting frames... {int(progress_pct * 100)}%",
                                "message_key": "collecting_frames",
                                "message_args": {"percent": int(progress_pct * 100)},
                                "progress": session.get_progress(),
                            }
                            await websocket.send_json(response)
                    else:
                        # No dynamic model
                        response = {
                            "session_id": session.id,
                            "hand_detected": True,
                            "recording": False,
                            "message": "Dynamic model not available",
                            "message_key": "model_unavailable",
                            "progress": session.get_progress(),
                        }
                        await websocket.send_json(response)
                        session.is_recording = False

                else:
                    # Static letter - use CNN
                    if static_predictor:
                        try:
                            # Raw landmarks: predict() normalizes internally.
                            pred_result = static_predictor.predict(landmarks)
                        except Exception as e:
                            logger.error("Static prediction error: %s", e)
                            continue

                        # Add prediction to session
                        result = session.add_prediction(
                            pred_result["predicted_class"], pred_result["confidence"]
                        )

                        if result:
                            # Recording finished
                            is_dynamic = is_dynamic_letter(session.current_letter)

                            response = {
                                "session_id": session.id,
                                "hand_detected": True,
                                "recording": False,
                                "prediction": {
                                    "predicted_class": pred_result["predicted_class"],
                                    "confidence": pred_result["confidence"],
                                },
                                "match": result.get("match", False),
                                "success": result.get("success", False),
                                "timeout": result.get("timeout", False),
                                "message": result.get("message", ""),
                                "message_key": result.get("message_key", ""),
                                "message_args": result.get("message_args", {}),
                                "show_hint": result.get("show_hint", False),
                                "hint_message": result.get("hint_message", ""),
                                "hint_key": result.get("hint_key", ""),
                                "progress": session.get_progress(),
                            }
                            await websocket.send_json(response)
                        else:
                            # Still recording
                            response = {
                                "session_id": session.id,
                                "hand_detected": True,
                                "recording": True,
                                "current_prediction": pred_result["predicted_class"],
                                "confidence": pred_result["confidence"],
                                "message": "Recording in progress...",
                                "message_key": "recording_in_progress",
                                "progress": session.get_progress(),
                            }
                            await websocket.send_json(response)
                    else:
                        # No static model
                        response = {
                            "session_id": session.id,
                            "hand_detected": True,
                            "recording": False,
                            "message": "Static model not available",
                            "message_key": "model_unavailable",
                            "progress": session.get_progress(),
                        }
                        await websocket.send_json(response)
                        session.is_recording = False

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except Exception:
        logger.exception("WebSocket error")
        # Do not leak the exception text to the client; it can carry paths and
        # internal state. The detail is in the server log.
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "error": "internal_error",
                    "message": "Server error occurred",
                    "recording": False,
                }
            )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon from static files."""
    favicon_path = FRONTEND_DIR / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(str(favicon_path), media_type="image/svg+xml")
    return Response(status_code=204)


# Mount static files after routes to avoid conflicts
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    # Prefer `python main.py`, which configures logging and the bind address.
    from main import main as run_app

    run_app()

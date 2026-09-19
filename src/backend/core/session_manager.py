"""
User session management for sign language learning.
Updated for recording-based workflow.
"""

import logging
import time
import uuid
from collections import Counter
from typing import Dict, Optional

from src.backend.core.config import (
    CONFIDENCE_THRESHOLD,
    MAX_ATTEMPT_TIME,
    MAX_HINTS,
    MAX_SESSIONS,
    RECORDING_DURATION,
    SESSION_MAX_IDLE,
)
from src.backend.core.letter_sequence import LetterSequence
from src.backend.core.tutorial_manager import TutorialManager

logger = logging.getLogger(__name__)


class UserSession:
    """Manages state for a single user learning session."""

    def __init__(self, session_id: Optional[str] = None, mode: str = "sequential"):
        """Initialize new session."""
        self.id = session_id or str(uuid.uuid4())
        self.created_at = time.time()
        # Bumped on every client message; drives idle expiry in SessionManager.
        self.last_seen = self.created_at
        self.mode = mode  # "sequential" or "random"
        self.letter_sequence = LetterSequence(mode=mode, include_dynamic=True)
        self.tutorial_manager = TutorialManager()

        # Current state
        self.current_letter = self.letter_sequence.get_next_letter()
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0

        # Recording state
        self.is_recording = False
        self.recording_start_time = None
        self.recording_predictions = []  # Collect predictions during recording

        # Statistics
        self.total_correct = 0
        self.total_attempts = 0
        self.completed_letters = []

        # Flags
        self.is_active = True
        self.show_hint = False
        self.hint_message = ""

        # Sentence Mode state
        self.target_sentence = ""
        self.recognized_sentence = ""
        self.sentence_index = 0

    def touch(self):
        """Mark the session as active so the idle sweeper leaves it alone."""
        self.last_seen = time.time()

    def start_recording(self):
        """Start a new recording attempt."""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_predictions = []
        logger.info("Started recording for letter '%s'", self.current_letter)

    def add_prediction(self, predicted_letter: str, confidence: float) -> Optional[Dict]:
        """
        Add a prediction during recording.

        Args:
            predicted_letter: Letter predicted by model
            confidence: Confidence score (0-1)

        Returns:
            None during recording, dict with result when recording complete
        """
        if not self.is_recording:
            return None

        # Store prediction
        self.recording_predictions.append(
            {"letter": predicted_letter, "confidence": confidence, "timestamp": time.time()}
        )

        # Check if recording duration reached
        elapsed = time.time() - self.recording_start_time
        if elapsed >= RECORDING_DURATION:
            return self.finish_recording()

        return None

    def finish_recording(self) -> Dict:
        """
        Finish recording and determine result using majority vote.

        Returns:
            dict with match status and next actions
        """
        self.is_recording = False
        self.attempt_count += 1
        self.total_attempts += 1

        if not self.recording_predictions:
            return {
                "match": False,
                "success": False,
                "timeout": False,
                "message": "No hand detected during recording",
                "message_key": "no_hand_during_recording",
                "show_hint": False,
            }

        # Filter by confidence threshold
        valid_predictions = [
            p for p in self.recording_predictions if p["confidence"] >= CONFIDENCE_THRESHOLD
        ]

        if not valid_predictions:
            return self._handle_failed_attempt("Low confidence predictions", key="low_confidence")

        # Majority vote
        letters = [p["letter"] for p in valid_predictions]
        letter_counts = Counter(letters)
        most_common_letter, count = letter_counts.most_common(1)[0]

        # Calculate average confidence for the most common letter
        avg_confidence = (
            sum(p["confidence"] for p in valid_predictions if p["letter"] == most_common_letter)
            / count
        )

        logger.info(
            "Predictions: %s | Winner: %s (%d/%d)",
            dict(letter_counts),
            most_common_letter,
            count,
            len(valid_predictions),
        )

        # Check if it matches target
        if self.mode == "sentence":
            if self.target_sentence and self.current_letter:
                match = most_common_letter == self.current_letter
                if match:
                    return self._handle_sentence_success(most_common_letter, avg_confidence)
                else:
                    return self._handle_failed_attempt(
                        f"Detected '{most_common_letter}' instead of '{self.current_letter}'",
                        key="wrong_letter",
                        args={"detected": most_common_letter, "expected": self.current_letter},
                    )
            else:
                return self._handle_free_mode_success(most_common_letter, avg_confidence)
        else:
            match = most_common_letter == self.current_letter

            if match:
                return self._handle_success(most_common_letter, avg_confidence)
            else:
                return self._handle_failed_attempt(
                    f"Detected '{most_common_letter}' instead of '{self.current_letter}'",
                    key="wrong_letter",
                    args={"detected": most_common_letter, "expected": self.current_letter},
                )

    def _handle_success(self, predicted_letter: str, confidence: float) -> Dict:
        """Handle successful letter recognition."""
        self.total_correct += 1
        self.completed_letters.append(self.current_letter)
        self.letter_sequence.mark_completed(self.current_letter)

        logger.info(
            "Success! Correctly signed '%s' (confidence: %.2f)", self.current_letter, confidence
        )

        # Advance to next letter
        next_letter = self.letter_sequence.get_next_letter(self.current_letter)
        self.current_letter = next_letter
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.show_hint = False

        return {
            "match": True,
            "success": True,
            "timeout": False,
            "predicted_letter": predicted_letter,
            "confidence": confidence,
            "message": f"Correct! Moving to letter {next_letter}",
            "message_key": "correct_next",
            "message_args": {"letter": next_letter},
            "next_letter": next_letter,
            "show_hint": False,
        }

    def _handle_failed_attempt(
        self, reason: str, key: str = "attempt_failed", args: Optional[Dict] = None
    ) -> Dict:
        """Handle failed recognition attempt."""
        # Check timeout
        elapsed = time.time() - self.letter_start_time
        if elapsed >= MAX_ATTEMPT_TIME:
            logger.info("Timeout on letter '%s'", self.current_letter)
            return self._handle_timeout()

        # Check if hint should be shown
        show_hint = self.tutorial_manager.should_show_hint(
            self.attempt_count, self.hints_shown, MAX_HINTS
        )

        if show_hint:
            self.hints_shown += 1
            self.show_hint = True
            self.hint_message = self.tutorial_manager.get_hint_message(
                self.current_letter, self.hints_shown
            )
        else:
            self.show_hint = False

        logger.info("Failed attempt %d: %s", self.attempt_count, reason)

        return {
            "match": False,
            "success": False,
            "timeout": False,
            "message": reason,
            "message_key": key,
            "message_args": args or {},
            "show_hint": self.show_hint,
            "hint_message": self.hint_message if self.show_hint else "",
            "hint_key": f"hint_{self.hints_shown}" if self.show_hint else "",
        }

    def _handle_timeout(self) -> Dict:
        """Handle timeout - move to next letter."""
        next_letter = self.letter_sequence.get_next_letter(self.current_letter)
        self.current_letter = next_letter
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.show_hint = False

        return {
            "match": False,
            "success": False,
            "timeout": True,
            "message": f"Time up! Moving to letter {next_letter}",
            "message_key": "timeout_next",
            "message_args": {"letter": next_letter},
            "next_letter": next_letter,
            "show_hint": False,
        }

    def _handle_sentence_success(self, predicted_letter: str, confidence: float) -> Dict:
        self.total_correct += 1
        self.recognized_sentence += predicted_letter
        self.sentence_index += 1

        logger.info(
            "Success! Correctly signed '%s' (confidence: %.2f)", predicted_letter, confidence
        )

        # Advance to next letter
        self._advance_to_next_valid_char()
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.show_hint = False

        return {
            "match": True,
            "success": True,
            "timeout": False,
            "predicted_letter": predicted_letter,
            "confidence": confidence,
            "message": "Correct! Next letter...",
            "message_key": "correct_next_sentence",
            "next_letter": self.current_letter,
            "show_hint": False,
        }

    def _handle_free_mode_success(self, predicted_letter: str, confidence: float) -> Dict:
        self.recognized_sentence += predicted_letter
        self.letter_start_time = time.time()
        self.attempt_count = 0

        logger.info("Free Mode: signed '%s' (confidence: %.2f)", predicted_letter, confidence)

        return {
            "match": True,
            "success": True,
            "timeout": False,
            "predicted_letter": predicted_letter,
            "confidence": confidence,
            "message": f"Signed {predicted_letter}",
            "message_key": "free_signed",
            "message_args": {"letter": predicted_letter},
            "next_letter": None,
            "show_hint": False,
        }

    def get_tutorial_url(self) -> Optional[str]:
        """Get tutorial GIF URL for current letter."""
        return self.tutorial_manager.get_tutorial_url(self.current_letter)

    def get_time_remaining(self) -> float:
        """Get seconds remaining for current letter."""
        elapsed = time.time() - self.letter_start_time
        return max(0, MAX_ATTEMPT_TIME - elapsed)

    def get_progress(self) -> dict:
        """Get current session progress."""
        accuracy = (
            (self.total_correct / self.total_attempts * 100) if self.total_attempts > 0 else 0
        )

        return {
            "current_letter": self.current_letter,
            "total_correct": self.total_correct,
            "total_attempts": self.total_attempts,
            "accuracy": accuracy,
            "completed_letters": self.completed_letters,
            "attempt_count": self.attempt_count,
            "time_remaining": self.get_time_remaining(),
            "tutorial_url": self.get_tutorial_url(),
            "is_recording": self.is_recording,
            "mode": self.mode,
            "target_sentence": self.target_sentence,
            "recognized_sentence": self.recognized_sentence,
        }

    def set_mode(self, mode: str):
        """Change letter sequence mode (sequential, random, sentence)."""
        if mode not in ["sequential", "random", "sentence"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'sequential', 'random', or 'sentence'."
            )

        previous_mode = self.mode
        self.mode = mode

        if mode != "sentence":
            self.letter_sequence.mode = mode

            # Leaving sentence mode used to leave current_letter wherever the
            # sentence had ended, including None once the sentence completed,
            # which rendered as "undefined" in the UI and blocked all matching.
            if previous_mode == "sentence" or self.current_letter is None:
                self.target_sentence = ""
                self.recognized_sentence = ""
                self.sentence_index = 0
                self.current_letter = self.letter_sequence.get_next_letter()
                self.letter_start_time = time.time()
                self.attempt_count = 0
                self.hints_shown = 0
                self.show_hint = False

        logger.info("Mode changed: %s -> %s (letter %s)", previous_mode, mode, self.current_letter)

    def set_target_sentence(self, sentence: str):
        """Set a target sentence for sentence mode."""
        self.target_sentence = sentence.upper()
        self.recognized_sentence = ""
        self.sentence_index = 0
        if self.target_sentence:
            self._advance_to_next_valid_char()
            logger.info("Target sentence set: '%s'", self.target_sentence)
        else:
            self.current_letter = None
            logger.info("Free mode activated")

    def _advance_to_next_valid_char(self):
        """Advance index past spaces."""
        while (
            self.sentence_index < len(self.target_sentence)
            and self.target_sentence[self.sentence_index] == " "
        ):
            self.recognized_sentence += " "
            self.sentence_index += 1

        if self.sentence_index < len(self.target_sentence):
            self.current_letter = self.target_sentence[self.sentence_index]
        else:
            self.current_letter = None  # Sentence complete

    def skip_letter(self) -> Dict:
        """
        Skip the current letter and move to the next one.

        Returns:
            Dict with skip result information
        """
        skipped_letter = self.current_letter

        # Advance according to the active mode.
        if self.mode == "sentence" and self.target_sentence:
            self.sentence_index += 1
            self._advance_to_next_valid_char()
        else:
            self.current_letter = self.letter_sequence.get_next_letter(self.current_letter)
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.is_recording = False
        self.recording_predictions = []
        self.show_hint = False

        # Mark as attempt (not counted as correct)
        self.total_attempts += 1

        logger.info("Skipped letter '%s' -> '%s'", skipped_letter, self.current_letter)

        return {
            "match": False,
            "success": False,
            "timeout": False,
            "skipped": True,
            "message": f"Skipped {skipped_letter}, now showing {self.current_letter}",
            "message_key": "skipped_letter",
            "message_args": {"skipped": skipped_letter, "letter": self.current_letter},
            "next_letter": self.current_letter,
            "show_hint": False,
        }

    def clear_recognized(self):
        """Clear recognized letters in sentence mode."""
        if self.mode == "sentence":
            self.recognized_sentence = ""
            self.sentence_index = 0
            if self.target_sentence:
                self._advance_to_next_valid_char()
            else:
                self.current_letter = None

    def reset(self):
        """Reset session to beginning."""
        self.letter_sequence.reset()
        self.current_letter = self.letter_sequence.get_next_letter()
        self.letter_start_time = time.time()
        self.attempt_count = 0
        self.hints_shown = 0
        self.is_recording = False
        self.recording_predictions = []
        self.total_correct = 0
        self.total_attempts = 0
        self.completed_letters = []
        self.is_active = True
        self.target_sentence = ""
        self.recognized_sentence = ""
        self.sentence_index = 0


class SessionManager:
    """
    Holds in-memory user sessions.

    Session IDs are always generated server-side. Accepting a client-supplied ID
    would let any client claim or collide with another user's session.

    Sessions expire on idle time rather than on an is_active flag, which nothing
    ever cleared, so the previous sweeper could never remove anything and the
    dictionary grew for the lifetime of the process.

    This store is per-process. Running more than one uvicorn worker gives each
    worker its own sessions, so requests would land on a worker that has never
    seen the session. See main.py.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS, max_idle_seconds: int = SESSION_MAX_IDLE):
        self.sessions: Dict[str, UserSession] = {}
        self.max_sessions = max_sessions
        self.max_idle_seconds = max_idle_seconds

    def create_session(self, mode: str = "sequential") -> UserSession:
        """Create a new session with a server-generated ID."""
        self.cleanup_expired()

        if len(self.sessions) >= self.max_sessions:
            # Evict the least recently used session rather than growing without limit.
            oldest_id = min(self.sessions, key=lambda sid: self.sessions[sid].last_seen)
            logger.warning("Session cap %d reached, evicting %s", self.max_sessions, oldest_id)
            del self.sessions[oldest_id]

        session = UserSession(mode=mode)
        self.sessions[session.id] = session
        logger.info("Created session %s (%d active)", session.id, len(self.sessions))
        return session

    def get_session(self, session_id: Optional[str]) -> Optional[UserSession]:
        """Look up a session by ID, refreshing its idle timer."""
        if not session_id:
            return None
        session = self.sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def get_or_create(self, session_id: Optional[str], mode: str = "sequential") -> UserSession:
        """Resume the named session if it exists, otherwise start a fresh one."""
        session = self.get_session(session_id)
        if session is not None:
            return session
        return self.create_session(mode=mode)

    def remove_session(self, session_id: str) -> bool:
        """Remove a session. Returns True if one was removed."""
        return self.sessions.pop(session_id, None) is not None

    def cleanup_expired(self, max_idle: Optional[int] = None) -> int:
        """
        Drop sessions idle for longer than max_idle seconds.

        Returns:
            Number of sessions removed.
        """
        max_idle = self.max_idle_seconds if max_idle is None else max_idle
        cutoff = time.time() - max_idle

        expired = [sid for sid, s in self.sessions.items() if s.last_seen < cutoff]
        for sid in expired:
            del self.sessions[sid]

        if expired:
            logger.info(
                "Expired %d idle session(s), %d remaining", len(expired), len(self.sessions)
            )
        return len(expired)

    def __len__(self) -> int:
        return len(self.sessions)

"""
Tests for session lifecycle, letter sequencing and hints.

None of this logic was covered before, which is where most of the audited bugs
were found.
"""

import time

import pytest

from src.backend.core.config import (
    ALL_LETTERS,
    DYNAMIC_LETTERS,
    HINT_THRESHOLD_ATTEMPTS,
    HINT_THRESHOLDS,
    MAX_HINTS,
    STATIC_LETTERS,
)
from src.backend.core.letter_sequence import LetterSequence
from src.backend.core.session_manager import SessionManager, UserSession
from src.backend.core.tutorial_manager import TutorialManager


class TestSessionManagerLifecycle:
    def test_ids_are_server_generated_and_unique(self) -> None:
        manager = SessionManager()
        ids = {manager.create_session().id for _ in range(20)}
        assert len(ids) == 20

    def test_client_cannot_name_a_session(self) -> None:
        # Regression guard: create_session used to accept a client-supplied ID,
        # which let any client claim or collide with another user's session.
        manager = SessionManager()
        session = manager.get_or_create("attacker-chosen-id")
        assert session.id != "attacker-chosen-id"

    def test_get_or_create_resumes_a_known_session(self) -> None:
        manager = SessionManager()
        first = manager.create_session()
        assert manager.get_or_create(first.id) is first

    def test_get_or_create_starts_fresh_for_unknown_id(self) -> None:
        manager = SessionManager()
        assert manager.get_or_create("does-not-exist") is not None
        assert len(manager) == 1

    def test_get_session_handles_none(self) -> None:
        assert SessionManager().get_session(None) is None

    def test_remove_session_reports_whether_it_existed(self) -> None:
        manager = SessionManager()
        session = manager.create_session()
        assert manager.remove_session(session.id) is True
        assert manager.remove_session(session.id) is False


class TestSessionExpiry:
    def test_idle_sessions_are_swept(self) -> None:
        manager = SessionManager(max_idle_seconds=60)
        stale = manager.create_session()
        fresh = manager.create_session()
        stale.last_seen = time.time() - 3600

        assert manager.cleanup_expired() == 1
        assert manager.get_session(stale.id) is None
        assert manager.get_session(fresh.id) is fresh

    def test_activity_prevents_expiry(self) -> None:
        manager = SessionManager(max_idle_seconds=60)
        session = manager.create_session()
        session.last_seen = time.time() - 3600
        session.touch()
        assert manager.cleanup_expired() == 0

    def test_session_count_is_capped(self) -> None:
        manager = SessionManager(max_sessions=3)
        for _ in range(10):
            manager.create_session()
        assert len(manager) <= 3

    def test_cap_evicts_least_recently_used(self) -> None:
        manager = SessionManager(max_sessions=2)
        oldest = manager.create_session()
        keep = manager.create_session()
        oldest.last_seen = time.time() - 100
        keep.touch()

        manager.create_session()
        assert manager.get_session(oldest.id) is None
        assert manager.get_session(keep.id) is keep


class TestLetterSequence:
    def test_sequential_walks_the_alphabet_without_repeats(self) -> None:
        sequence = LetterSequence(mode="sequential", include_dynamic=True)
        letter = sequence.get_next_letter()
        seen = [letter]
        for _ in range(len(ALL_LETTERS) - 1):
            letter = sequence.get_next_letter(letter)
            seen.append(letter)
        assert seen == ALL_LETTERS

    def test_sequential_wraps_around(self) -> None:
        sequence = LetterSequence(mode="sequential", include_dynamic=True)
        assert sequence.get_next_letter(ALL_LETTERS[-1]) == ALL_LETTERS[0]

    def test_random_never_repeats_the_current_letter(self) -> None:
        sequence = LetterSequence(mode="random", include_dynamic=True)
        for _ in range(200):
            assert sequence.get_next_letter("M") != "M"

    def test_excluding_dynamic_drops_j_and_z(self) -> None:
        sequence = LetterSequence(include_dynamic=False)
        assert sequence.available_letters == STATIC_LETTERS
        assert not set(DYNAMIC_LETTERS) & set(sequence.available_letters)

    def test_mark_completed_is_idempotent(self) -> None:
        sequence = LetterSequence()
        sequence.mark_completed("A")
        sequence.mark_completed("A")
        assert sequence.completed_letters == ["A"]


class TestSentenceMode:
    def test_spaces_are_skipped_and_echoed(self) -> None:
        session = UserSession(mode="sentence")
        session.set_target_sentence("a b")
        assert session.current_letter == "A"

        session._handle_sentence_success("A", 0.9)
        # The space between the words is consumed automatically.
        assert session.current_letter == "B"
        assert session.recognized_sentence == "A "

    def test_sentence_completion_clears_the_target_letter(self) -> None:
        session = UserSession(mode="sentence")
        session.set_target_sentence("AB")
        session._handle_sentence_success("A", 0.9)
        session._handle_sentence_success("B", 0.9)
        assert session.current_letter is None

    def test_empty_sentence_enters_free_mode(self) -> None:
        session = UserSession(mode="sentence")
        session.set_target_sentence("")
        assert session.current_letter is None

    def test_clear_recognized_restarts_the_sentence(self) -> None:
        session = UserSession(mode="sentence")
        session.set_target_sentence("AB")
        session._handle_sentence_success("A", 0.9)
        session.clear_recognized()
        assert session.recognized_sentence == ""
        assert session.current_letter == "A"


class TestSessionProgress:
    def test_accuracy_is_zero_before_any_attempt(self) -> None:
        assert UserSession().get_progress()["accuracy"] == 0

    def test_accuracy_is_a_percentage(self) -> None:
        session = UserSession()
        session.total_correct, session.total_attempts = 3, 4
        assert session.get_progress()["accuracy"] == pytest.approx(75.0)

    def test_invalid_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            UserSession().set_mode("banana")

    @pytest.mark.parametrize("mode", ["sequential", "random", "sentence"])
    def test_valid_modes_are_accepted(self, mode) -> None:
        session = UserSession()
        session.set_mode(mode)
        assert session.mode == mode

    def test_skip_advances_and_counts_an_attempt(self) -> None:
        session = UserSession()
        first = session.current_letter
        result = session.skip_letter()
        assert result["skipped"] is True
        assert session.current_letter != first
        assert session.total_attempts == 1

    def test_recording_with_no_predictions_fails_cleanly(self) -> None:
        session = UserSession()
        session.start_recording()
        result = session.finish_recording()
        assert result["match"] is False
        assert "No hand detected" in result["message"]

    def test_add_prediction_ignored_when_not_recording(self) -> None:
        assert UserSession().add_prediction("A", 0.99) is None


class TestTutorialManager:
    def test_hints_stop_at_the_configured_maximum(self) -> None:
        manager = TutorialManager()
        assert manager.should_show_hint(attempt_count=5, hints_shown=2, max_hints=2) is False

    def test_hint_fires_on_a_threshold_attempt(self) -> None:
        manager = TutorialManager()
        assert (
            manager.should_show_hint(
                attempt_count=HINT_THRESHOLDS[0], hints_shown=0, max_hints=MAX_HINTS
            )
            is True
        )

    def test_no_hint_between_thresholds(self) -> None:
        manager = TutorialManager()
        between = HINT_THRESHOLDS[0] + 1
        assert between not in HINT_THRESHOLDS
        assert (
            manager.should_show_hint(attempt_count=between, hints_shown=0, max_hints=MAX_HINTS)
            is False
        )

    def test_thresholds_follow_the_configured_constant(self) -> None:
        # Regression guard: these were hardcoded [5, 10, 15] while the config
        # said 3, so HINT_THRESHOLD_ATTEMPTS was dead and the docs were wrong.
        assert HINT_THRESHOLDS == [HINT_THRESHOLD_ATTEMPTS * (i + 1) for i in range(MAX_HINTS)]
        assert len(HINT_THRESHOLDS) == MAX_HINTS

    def test_missing_letter_yields_no_tutorial(self) -> None:
        manager = TutorialManager()
        assert manager.get_tutorial_url(None) is None
        assert manager.has_tutorial(None) is False

    def test_every_letter_has_a_tutorial_gif(self) -> None:
        # The assets are committed, so a missing GIF is a real packaging bug.
        manager = TutorialManager()
        missing = [letter for letter in ALL_LETTERS if not manager.has_tutorial(letter)]
        assert missing == []


class TestModeSwitching:
    """Leaving sentence mode used to strand current_letter at None."""

    def test_leaving_sentence_mode_restores_a_letter(self) -> None:
        session = UserSession(mode="sentence")
        session.set_target_sentence("AB")
        session._handle_sentence_success("A", 0.9)
        session._handle_sentence_success("B", 0.9)
        assert session.current_letter is None  # sentence finished

        session.set_mode("sequential")
        assert session.current_letter in ALL_LETTERS

    def test_leaving_sentence_mode_clears_sentence_state(self) -> None:
        session = UserSession(mode="sentence")
        session.set_target_sentence("HELLO")
        session._handle_sentence_success("H", 0.9)

        session.set_mode("random")
        assert session.target_sentence == ""
        assert session.recognized_sentence == ""
        assert session.sentence_index == 0
        assert session.current_letter in ALL_LETTERS

    def test_switching_between_letter_modes_keeps_the_letter(self) -> None:
        session = UserSession(mode="sequential")
        letter = session.current_letter
        session.set_mode("random")
        assert session.current_letter == letter


class TestLocalizationKeys:
    """Every user-visible server message carries a key the client can translate."""

    def test_failed_attempt_carries_a_key(self) -> None:
        session = UserSession()
        session.start_recording()
        session.recording_predictions = [{"letter": "B", "confidence": 0.99}]
        session.current_letter = "A"
        result = session.finish_recording()
        assert result["message_key"] == "wrong_letter"
        assert result["message_args"] == {"detected": "B", "expected": "A"}

    def test_success_carries_a_key(self) -> None:
        session = UserSession()
        session.start_recording()
        target = session.current_letter
        session.recording_predictions = [{"letter": target, "confidence": 0.99}]
        result = session.finish_recording()
        assert result["message_key"] == "correct_next"
        assert "letter" in result["message_args"]

    def test_skip_carries_a_key(self) -> None:
        assert UserSession().skip_letter()["message_key"] == "skipped_letter"

    def test_no_hand_carries_a_key(self) -> None:
        session = UserSession()
        session.start_recording()
        assert session.finish_recording()["message_key"] == "no_hand_during_recording"

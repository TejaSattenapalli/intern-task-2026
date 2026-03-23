"""Integration tests -- require ANTHROPIC_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import os

import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar", "spelling", "word_choice", "punctuation", "word_order",
    "missing_word", "extra_word", "conjugation", "gender_agreement",
    "number_agreement", "tone_register", "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


def assert_valid_response(result):
    """Assert that a FeedbackResponse is structurally valid."""
    assert result.difficulty in VALID_DIFFICULTIES
    assert isinstance(result.is_correct, bool)
    assert isinstance(result.errors, list)
    assert isinstance(result.corrected_sentence, str)
    assert len(result.corrected_sentence) > 0
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0
        assert len(error.original) > 0
        assert error.correction is not None


@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    result = await get_feedback(FeedbackRequest(
        sentence="Yo soy fue al mercado ayer.",
        target_language="Spanish",
        native_language="English",
    ))
    assert_valid_response(result)
    assert result.is_correct is False
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_correct_german_sentence():
    result = await get_feedback(FeedbackRequest(
        sentence="Ich habe gestern einen interessanten Film gesehen.",
        target_language="German",
        native_language="English",
    ))
    assert_valid_response(result)
    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == "Ich habe gestern einen interessanten Film gesehen."


@pytest.mark.asyncio
async def test_french_gender_agreement_errors():
    result = await get_feedback(FeedbackRequest(
        sentence="La chat noir est sur le table.",
        target_language="French",
        native_language="English",
    ))
    assert_valid_response(result)
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert any(e.error_type == "gender_agreement" for e in result.errors)


@pytest.mark.asyncio
async def test_japanese_particle_non_latin():
    """Tests non-Latin script handling and particle error."""
    result = await get_feedback(FeedbackRequest(
        sentence="私は東京を住んでいます。",
        target_language="Japanese",
        native_language="English",
    ))
    assert_valid_response(result)
    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_portuguese_spelling_and_grammar():
    result = await get_feedback(FeedbackRequest(
        sentence="Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
        target_language="Portuguese",
        native_language="English",
    ))
    assert_valid_response(result)
    assert result.is_correct is False
    error_types = {e.error_type for e in result.errors}
    assert "spelling" in error_types


@pytest.mark.asyncio
async def test_correct_simple_spanish():
    """A simple, fully-correct Spanish sentence should return is_correct=True."""
    result = await get_feedback(FeedbackRequest(
        sentence="El niño come una manzana.",
        target_language="Spanish",
        native_language="English",
    ))
    assert_valid_response(result)
    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty in {"A1", "A2"}


@pytest.mark.asyncio
async def test_arabic_non_latin_conjugation():
    """Tests Arabic (right-to-left, non-Latin) script with a conjugation error."""
    result = await get_feedback(FeedbackRequest(
        sentence="أنا تدرس اللغة العربية.",
        target_language="Arabic",
        native_language="English",
    ))
    assert_valid_response(result)
    assert result.is_correct is False
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_native_language_explanations():
    """Explanations should be in the native language (French), not the target (Spanish)."""
    result = await get_feedback(FeedbackRequest(
        sentence="Yo soy fue al mercado ayer.",
        target_language="Spanish",
        native_language="French",
    ))
    assert_valid_response(result)
    # Can't fully auto-verify the language of text, but explanation must be non-empty
    for error in result.errors:
        assert len(error.explanation) > 5

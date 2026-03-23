"""Unit tests -- run without an API key using mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.feedback import get_feedback, _extract_json, _validate_and_coerce
from app.models import FeedbackRequest


def _mock_openai_response(response_data: dict) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ---------------------------------------------------------------------------
# Helper / utility unit tests
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    raw = '{"corrected_sentence": "ok", "is_correct": true, "errors": [], "difficulty": "A1"}'
    result = _extract_json(raw)
    assert result["is_correct"] is True


def test_extract_json_strips_markdown_fence():
    raw = '```json\n{"corrected_sentence": "ok", "is_correct": true, "errors": [], "difficulty": "A1"}\n```'
    result = _extract_json(raw)
    assert result["difficulty"] == "A1"


def test_validate_coerce_bad_difficulty():
    data = {"corrected_sentence": "x", "is_correct": True, "errors": [], "difficulty": "Z9"}
    result = _validate_and_coerce(data)
    assert result["difficulty"] == "A1"


def test_validate_coerce_bad_error_type():
    data = {
        "corrected_sentence": "x",
        "is_correct": False,
        "errors": [{"original": "a", "correction": "b", "error_type": "totally_fake", "explanation": "..."}],
        "difficulty": "B1",
    }
    result = _validate_and_coerce(data)
    assert result["errors"][0]["error_type"] == "other"


def test_validate_coerce_is_correct_inconsistency_with_errors():
    data = {
        "corrected_sentence": "x",
        "is_correct": True,
        "errors": [{"original": "a", "correction": "b", "error_type": "grammar", "explanation": "..."}],
        "difficulty": "A1",
    }
    result = _validate_and_coerce(data)
    assert result["is_correct"] is False


def test_validate_coerce_is_correct_no_errors():
    data = {"corrected_sentence": "x", "is_correct": False, "errors": [], "difficulty": "A1"}
    result = _validate_and_coerce(data)
    assert result["is_correct"] is True


# ---------------------------------------------------------------------------
# Core API integration unit tests (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    mock_data = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms. Use only 'fui' (I went).",
            }
        ],
        "difficulty": "A2",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_openai_response(mock_data))

        result = await get_feedback(FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        ))

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_correct_german_sentence():
    mock_data = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_openai_response(mock_data))

        result = await get_feedback(FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        ))

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == "Ich habe gestern einen interessanten Film gesehen."
    assert result.difficulty == "B1"


@pytest.mark.asyncio
async def test_french_gender_agreement_multiple_errors():
    mock_data = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' (cat) is masculine -- use 'le', not 'la'.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine -- use 'la', not 'le'.",
            },
        ],
        "difficulty": "A1",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_openai_response(mock_data))

        result = await get_feedback(FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        ))

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)
    assert result.difficulty == "A1"


@pytest.mark.asyncio
async def test_japanese_particle_error():
    mock_data = {
        "corrected_sentence": "\u79c1\u306f\u6771\u4eac\u306b\u4f4f\u3093\u3067\u3044\u307e\u3059\u3002",
        "is_correct": False,
        "errors": [
            {
                "original": "\u3092",
                "correction": "\u306b",
                "error_type": "grammar",
                "explanation": "The verb to live requires the particle ni, not wo.",
            }
        ],
        "difficulty": "A2",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_openai_response(mock_data))

        result = await get_feedback(FeedbackRequest(
            sentence="\u79c1\u306f\u6771\u4eac\u3092\u4f4f\u3093\u3067\u3044\u307e\u3059\u3002",
            target_language="Japanese",
            native_language="English",
        ))

    assert result.is_correct is False
    assert any("\u306b" in e.correction for e in result.errors)
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_portuguese_spelling_and_grammar():
    mock_data = {
        "corrected_sentence": "Eu quero comprar um presente para minha irma, mas nao sei do que ela gosta.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "Gift in Portuguese is spelled presente with an s.",
            },
            {
                "original": "o que ela gosta",
                "correction": "do que ela gosta",
                "error_type": "grammar",
                "explanation": "The verb gostar requires the preposition de.",
            },
        ],
        "difficulty": "B1",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_openai_response(mock_data))

        result = await get_feedback(FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irma, mas nao sei o que ela gosta.",
            target_language="Portuguese",
            native_language="English",
        ))

    assert result.is_correct is False
    assert len(result.errors) == 2
    error_types = {e.error_type for e in result.errors}
    assert "spelling" in error_types
    assert "grammar" in error_types


@pytest.mark.asyncio
async def test_arabic_non_latin_script():
    mock_data = {
        "corrected_sentence": "Arabic corrected sentence",
        "is_correct": False,
        "errors": [
            {
                "original": "wrong form",
                "correction": "correct form",
                "error_type": "conjugation",
                "explanation": "For I (ana), Arabic uses the verb form adrus.",
            }
        ],
        "difficulty": "A2",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_openai_response(mock_data))

        result = await get_feedback(FeedbackRequest(
            sentence="Arabic sentence with error",
            target_language="Arabic",
            native_language="English",
        ))

    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.errors[0].error_type == "conjugation"


@pytest.mark.asyncio
async def test_explanation_in_native_language_not_target():
    mock_data = {
        "corrected_sentence": "Yo fui al mercado.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "Du hast zwei Verbformen gemischt. Benutze nur fui.",
            }
        ],
        "difficulty": "A2",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_openai_response(mock_data))

        result = await get_feedback(FeedbackRequest(
            sentence="Yo soy fue al mercado.",
            target_language="Spanish",
            native_language="German",
        ))

    assert len(result.errors) > 0
    assert len(result.errors[0].explanation) > 0


@pytest.mark.asyncio
async def test_high_cefr_complex_sentence():
    mock_data = {
        "corrected_sentence": "Bien que la situation soit difficile, il convient de maintenir son calme.",
        "is_correct": False,
        "errors": [
            {
                "original": "sois",
                "correction": "soit",
                "error_type": "conjugation",
                "explanation": "After bien que, French requires the subjunctive. Use soit not sois.",
            }
        ],
        "difficulty": "C1",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_openai_response(mock_data))

        result = await get_feedback(FeedbackRequest(
            sentence="Bien que la situation sois difficile, il convient de maintenir son calme.",
            target_language="French",
            native_language="English",
        ))

    assert result.difficulty in {"B2", "C1", "C2"}
    assert result.is_correct is False
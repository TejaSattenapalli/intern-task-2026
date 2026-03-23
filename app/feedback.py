"""System prompt and LLM interaction for language feedback."""

import json
import re
import asyncio
from openai import AsyncOpenAI

from app.models import FeedbackRequest, FeedbackResponse

SYSTEM_PROMPT = """\
You are an expert language tutor specializing in error correction for language learners.
Your task is to analyze a learner's sentence in their target language, identify errors, and return structured feedback.

ANALYSIS GUIDELINES:
1. Carefully examine the sentence for ALL errors: grammar, spelling, conjugation, gender/number agreement, word order, punctuation, missing/extra words, word choice, and tone/register issues.
2. Preserve the learner's intended meaning - make only the minimal corrections necessary.
3. If the sentence is already fully correct, set is_correct=true, return an empty errors array, and set corrected_sentence to the original sentence verbatim.
4. Explanations MUST be written in the learner's native language (not the target language).
5. Keep explanations concise (1-3 sentences), warm, and educational.
6. For the CEFR difficulty rating, assess sentence complexity based on vocabulary and grammar structures used - NOT based on whether it has errors.

ERROR TYPE RULES - use exactly one of these values:
grammar, spelling, word_choice, punctuation, word_order, missing_word, extra_word,
conjugation, gender_agreement, number_agreement, tone_register, other

CEFR DIFFICULTY:
A1: Very simple sentences, basic vocabulary, present tense only
A2: Simple connected sentences, common vocabulary, basic past/future
B1: More complex sentences, varied tenses, some idiomatic language
B2: Complex structures, abstract topics, nuanced vocabulary
C1: Sophisticated structures, idiomatic, extended discourse
C2: Near-native complexity, subtle distinctions, rare vocabulary

CRITICAL: Respond with ONLY a valid JSON object - no markdown, no code fences, no preamble.
Schema:
{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {
      "original": "string",
      "correction": "string",
      "error_type": "one of the allowed types",
      "explanation": "string (written in the learner's NATIVE language)"
    }
  ],
  "difficulty": "A1|A2|B1|B2|C1|C2"
}
"""

VALID_ERROR_TYPES = {
    "grammar", "spelling", "word_choice", "punctuation", "word_order",
    "missing_word", "extra_word", "conjugation", "gender_agreement",
    "number_agreement", "tone_register", "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def _validate_and_coerce(data: dict) -> dict:
    if data.get("difficulty") not in VALID_DIFFICULTIES:
        data["difficulty"] = "A1"
    if not isinstance(data.get("errors"), list):
        data["errors"] = []
    for error in data.get("errors", []):
        if error.get("error_type") not in VALID_ERROR_TYPES:
            error["error_type"] = "other"
    has_errors = len(data.get("errors", [])) > 0
    if data.get("is_correct") is True and has_errors:
        data["is_correct"] = False
    elif data.get("is_correct") is False and not has_errors:
        data["is_correct"] = True
    return data


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    client = AsyncOpenAI()

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Learner's sentence: {request.sentence}"
    )

    last_error = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.choices[0].message.content
            data = _extract_json(content)
            data = _validate_and_coerce(data)
            return FeedbackResponse(**data)

        except json.JSONDecodeError as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(1)
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(1)

    raise RuntimeError(f"Failed to get valid feedback after 3 attempts: {last_error}")
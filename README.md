# Language Feedback API

An LLM-powered language feedback API built with FastAPI and Claude (Anthropic). Given a learner's sentence in a target language and their native language, it returns structured correction feedback including a corrected sentence, categorized errors with learner-friendly explanations, and a CEFR difficulty rating.

## Design Decisions

### LLM Provider: Anthropic Claude (claude-sonnet-4-20250514)
I chose Claude Sonnet over GPT-4o-mini for this task because:
- Claude excels at nuanced linguistic analysis and following complex structured output instructions
- Claude's JSON output is highly reliable, reducing parse failures
- The model handles non-Latin scripts (Japanese, Arabic, Chinese) particularly well
- `temperature=0.1` ensures deterministic, consistent corrections

### Prompt Engineering Strategy
The system prompt is designed around three principles:

1. **Explicit constraint enumeration**: All valid `error_type` values and CEFR levels are listed in the prompt. This eliminates hallucinated categories and keeps output schema-compliant.

2. **Separation of concerns**: The CEFR difficulty rating is explicitly decoupled from correctness — a simple error-filled sentence is still A1. This prevents the model from rating difficulty based on error count.

3. **Native language enforcement**: The prompt explicitly states explanations must be in the native language, not the target. This is critical for multilingual learners and is emphasized with a direct instruction.

4. **Minimal correction principle**: The prompt emphasizes preserving the learner's voice and intended meaning, avoiding over-correction.

### Robustness Features
- **Retry logic (3 attempts)**: Transient JSON parse failures or API errors are retried with a 1-second backoff
- **JSON extraction helper**: Strips markdown code fences that some models occasionally emit
- **Post-processing validation**: `_validate_and_coerce` fixes invalid `difficulty` values, bad `error_type` strings, and `is_correct`/errors inconsistencies
- **Structured error responses**: HTTP 502 for upstream LLM failures, 500 for unexpected errors

## How to Run

### Prerequisites
- Python 3.11+
- An Anthropic API key

### Local Development
```bash
git clone <your-fork-url>
cd intern-task-2026

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

uvicorn app.main:app --reload
```

Test it:
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

Health check:
```bash
curl http://localhost:8000/health
```

### Docker
```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
docker compose up --build
```

### Running Tests
```bash
# Unit tests (no API key needed — uses mocked responses)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires ANTHROPIC_API_KEY in .env)
pytest tests/test_feedback_integration.py -v

# All tests
pytest -v
```

## Test Suite Coverage

The test suite covers:
| Test | Description |
|------|-------------|
| Spanish conjugation error | `soy fue` → `fui` |
| Correct German sentence | No errors, `is_correct=True` |
| French double gender agreement | Two `gender_agreement` errors |
| Japanese particle (non-Latin) | `を` → `に` with non-Latin script |
| Portuguese spelling + grammar | Mixed error types |
| Arabic conjugation (RTL script) | Non-Latin right-to-left script |
| Native language explanations | Explanations in learner's native language |
| High CEFR complexity | C1-level French subjunctive |
| JSON extraction edge cases | Handles markdown fences in output |
| Schema validation inconsistencies | `is_correct`/error list coercion |

## API Reference

### `POST /feedback`
**Request:**
```json
{
  "sentence": "string (min length 1)",
  "target_language": "string",
  "native_language": "string"
}
```

**Response:**
```json
{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {
      "original": "string",
      "correction": "string",
      "error_type": "grammar|spelling|word_choice|punctuation|word_order|missing_word|extra_word|conjugation|gender_agreement|number_agreement|tone_register|other",
      "explanation": "string (in native language)"
    }
  ],
  "difficulty": "A1|A2|B1|B2|C1|C2"
}
```

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

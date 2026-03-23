"""FastAPI application -- language feedback endpoint."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.feedback import get_feedback
from app.models import FeedbackRequest, FeedbackResponse

load_dotenv()

app = FastAPI(
    title="Language Feedback API",
    description="Analyzes learner-written sentences and provides structured language feedback.",
    version="2.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        return await get_feedback(request)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

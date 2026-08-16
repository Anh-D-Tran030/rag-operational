"""POST /feedback route."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/feedback")


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1)
    score: int = Field(ge=-1, le=1)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    trace_id: str
    status: str


@router.post("/", response_model=FeedbackResponse)
async def feedback(request: Request, body: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback score for an existing trace."""
    try:
        request.app.state.tracer.log_score(body.trace_id, "user_feedback", body.score)
    except Exception as exc:
        detail = {"error": "feedback_failed", "detail": str(exc)}
        raise HTTPException(status_code=500, detail=detail) from exc
    return FeedbackResponse(trace_id=body.trace_id, status="ok")

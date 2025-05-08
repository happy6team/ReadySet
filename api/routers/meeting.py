from fastapi import APIRouter, Depends, HTTPException, Form, Request
from ..schemas.meeting_dto import MeetingSummaryRequest
import copy
from meeting.text_summarizer import summarize_meeting_text

router = APIRouter()

# 텍스트 요약 API
@router.post("/meeting/summarize")
async def summarize_text(text: str = Form(...)):
    try:
        summarized_text = summarize_meeting_text(text)
        
        return {
            "status": "success",
            "original_text": text,
            "summary": summarized_text
        }
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


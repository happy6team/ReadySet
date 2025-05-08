from fastapi import APIRouter, Depends, HTTPException, Form, Request,UploadFile, File
import copy
from meeting.text_summarizer import summarize_meeting_text
from meeting.model import RtzrAPI
import os
import uuid
import time

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

# 음성 파일 업로드 API
@router.post("/meeting/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        UPLOAD_DIR= "meeting/resource"
        # 고유한 파일 이름 생성
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 파일 저장
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        return {
            "status": "success",
            "message": "파일이 성공적으로 업로드되었습니다.",
            "filename": file.filename,
            "file_id": unique_filename.split('.')[0]  # UUID 부분만 반환
        }
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

# 음성 파일 요약 API
@router.post("/meeting/summarize/{file_id}")
async def summarize_audio(file_id: str):
    try:
        # 파일 찾기
        UPLOAD_DIR= "meeting/resource"
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(file_id):
                file_path = os.path.join(UPLOAD_DIR, filename)
                break
        else:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

        file_dict = {"file": (file_path, open(file_path, "rb"))}

        api = RtzrAPI(
            file_path=file_dict,
            speaker_num=2,
            domain="일반",
            profanity_filter=True,
            keyword=["홍대", "성수", "신제품", "SKALA", "광고"],
            dev=False,  # dev API 쓸 경우 True
        )

        print("⏳ 변환 중... (최대 수 분 소요)")

        # polling
        while api.raw_data is None:
            time.sleep(5)
            api.api_get()
            print("🔁 재요청 중...")

        print("✅ 텍스트 변환 완료:")
        print(api.voice_data)

        print("🧠 요약 중...")
        api.summary_inference()

        print("\n📌 요약 결과:")
        print(api.summary_data)
        
        return {
            "status": "success",
            "transcription": api.summary_data,
            "file_id": file_id
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        return HTTPException(status_code=500, detail=str(e))
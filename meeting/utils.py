# meeting_summarization/utils.py

import os
from pathlib import Path
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# 모델과 토크나이저 로딩 (최초 1회만 로딩되도록 캐싱)
_model = None
_tokenizer = None

def load_model():
    """
    KoBART 요약 모델과 토크나이저를 로드합니다.
    여러 번 호출 시 재사용됩니다.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("🔄 모델 로딩 중...")
        _model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")
        _tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
        print("✅ 모델 로딩 완료.")
    return _model, _tokenizer

def file_upload_save(dir_path: str, upload_file) -> str:
    """
    업로드된 파일을 저장하고 경로 반환
    - dir_path: 저장 디렉토리 경로
    - upload_file: UploadFile 객체 또는 파일 객체
    """
    os.makedirs(dir_path, exist_ok=True)

    save_path = Path(dir_path) / upload_file.filename
    with open(save_path, "wb") as f:
        f.write(upload_file.file.read())  # FastAPI UploadFile 기준
    return str(save_path)

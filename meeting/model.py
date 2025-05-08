# meeting_summarization/model.py

import os
import json
import requests
from dotenv import load_dotenv
from utils import load_model


# .env에서 키 로드
load_dotenv()

VITO_CLIENT_ID = os.getenv("VITO_CLIENT_ID")
VITO_CLIENT_SECRET = os.getenv("VITO_CLIENT_SECRET")

class RtzrAPI:
    def __init__(
        self,
        file_path: dict,
        speaker_num: int = 2,
        domain: str = "일반",
        profanity_filter: bool = False,   # ✅ 비속어 필터링 제거
        keyword: list = None,
        dev: bool = False,
    ) -> None:
        """VITO API 사용을 위한 초기 설정"""
        self.dev = "dev-" if dev else ""
        self.file_path = file_path
        self.speaker_num = speaker_num
        self.config = {"domain": "GENERAL"} if domain == "일반" else {"domain": "CALL"}

        # 화자 수 명확히 설정
        if speaker_num > 0:
            self.config["use_diarization"] = True
            self.config["diarization"] = {"spk_count": speaker_num}

        # 비속어 필터링 여부 설정
        if profanity_filter:
            self.config["use_profanity_filter"] = True

        # 키워드 부스팅
        if keyword:
            self.config["keyword"] = keyword
        else:
            self.config["keyword"] = []

        self.raw_data = None
        self.voice_data = None
        self.summary_data = None

        from dotenv import load_dotenv
        load_dotenv()
        import os
        client_id = os.getenv("VITO_CLIENT_ID")
        client_secret = os.getenv("VITO_CLIENT_SECRET")

        self.access_token = self.auth_check(client_id, client_secret)
        self.transcribe_id = self.api_post(self.access_token)

    def auth_check(self, client_id: str, client_secret: str) -> str:
        """API 인증"""
        resp = requests.post(
            f"https://{self.dev}openapi.vito.ai/v1/authenticate",
            data={"client_id": client_id, "client_secret": client_secret},
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def api_post(self, access_token: str) -> str:
        """오디오 파일 등록 및 텍스트 전환 요청"""
        resp = requests.post(
            f"https://{self.dev}openapi.vito.ai/v1/transcribe",
            headers={"Authorization": f"Bearer {access_token}"},
            files=self.file_path,
            data={"config": json.dumps(self.config)},
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def api_get(self) -> None:
        """텍스트 변환 결과 요청"""
        resp = requests.get(
            f"https://{self.dev}openapi.vito.ai/v1/transcribe/" + self.transcribe_id,
            headers={"Authorization": "Bearer " + self.access_token},
        )
        resp.raise_for_status()

        if resp.json()["status"] == "transcribing":
            self.raw_data = None
        else:
            self.raw_data = resp.json()
            self.voice_data = self.preprocessing(self.raw_data)

    def preprocessing(self, raw_data: dict) -> str:
        """다화자 포함 시 텍스트 포맷 조정"""
        speakers = set([x["spk"] for x in raw_data["results"]["utterances"]])
        if len(speakers) == 1:
            return " ".join([data["msg"] for data in raw_data["results"]["utterances"]])
        else:
            return "\n".join(
                [f"화자{text_data['spk']}] {text_data['msg']}" for text_data in raw_data["results"]["utterances"]]
            )
            
    def summary_inference(self) -> None:
        """
        KoBART 요약 모델을 사용해 voice_data 텍스트를 요약합니다.
        """
        if not self.voice_data or len(self.voice_data) < 40:
            self.summary_data = "Text too short!"
            return

        model, tokenizer = load_model()

        # 입력 인코딩
        inputs = tokenizer(
            self.voice_data,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1026,
        )
        input_length = len(self.voice_data)

        # 동적으로 길이 조정
        if input_length < 300:
            max_len = 200
            min_len = 50
        elif input_length < 1000:
            max_len = 400
            min_len = 80
        else:
            max_len = 600
            min_len = 100


        # 요약 생성
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            repetition_penalty=1.3,
            no_repeat_ngram_size=5,
            length_penalty=1.0,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
        )


        self.summary_data = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


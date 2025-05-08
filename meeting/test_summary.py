# meeting_summarization/test_summary.py

import time
from model import RtzrAPI
from text_summarizer import summarize_meeting_text


# 테스트용 오디오 파일 경로
AUDIO_FILE_PATH = "./resource/test_audio2.m4a"

text = """
1. 도입부 논의
김다은 책임

현재 스마트팜 환경 내 주요 작물(파프리카, 토마토 등)에 대한 생장 데이터는 수집되고 있으나, 환경변수와 수확량 간의 직접 상관성 모델이 미흡합니다. 특히 CO₂ 농도와 관수 주기 간 인과 추정이 요구됩니다.

박시현 매니저

맞습니다. 현장에서는 '하우스별 품질 차이'에 대한 피드백이 지속적으로 들어오고 있으며, 이는 단순 수치 기반이 아닌 맥락 기반 조정 로직이 필요하다는 의미입니다.

2. 주요 논의 사항
이진우 수석

현재 설치된 센서는 1분 간격으로 토양 수분, 온도, CO₂, 조도 데이터를 전송합니다. 기존 API Latency가 평균 3.4초로 측정되므로, 실시간 분석보다는 시간대별 배치 기반 모델 예측으로 전략을 바꿀 것을 제안드립니다.

김다은 책임

동의합니다. 모델 구조는 기존 LightGBM 기반에서 Attention 구조를 도입한 LSTM-Hybrid로 테스트 중이며, 조도 편차가 클 경우 수확량 예측 정확도가 12% 상승한 결과를 얻었습니다.

박시현 매니저

다만 농가 단에서는 모델 해석 가능성이 중요하므로, Explainable AI 접근도 고려해 주셨으면 합니다. 예: SHAP 값 기반 주요 변수 시각화 제공.

3. 결정 사항 및 Action Item
[결정] 향후 2주간 LSTM-Attention 기반 모델의 베타 버전을 하우스 2, 5, 7호에 우선 적용한다.

[결정] SHAP 기반 인사이트 리포트는 월 1회 제공하며, 농업인 교육자료로도 활용 예정.

[Action] 이진우 수석 → 하우스별 실시간 데이터 수신률 점검 (5월 12일까지)

[Action] 김다은 책임 → 베타 모델 학습 완료 및 예측 결과 정리 (5월 16일까지)

[Action] 박시현 매니저 → 농가 대상 설명회 일정 조율 (5월 20일까지)
"""
def run_test():
    print("📤 API 호출 시작")

    # FastAPI UploadFile 형식 흉내 → (filename, file_obj) 형식
    file_dict = {"file": (AUDIO_FILE_PATH, open(AUDIO_FILE_PATH, "rb"))}

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

if __name__ == "__main__":
    # run_test()
    print(summarize_meeting_text("이 회의는 스마트팜 센서 개선 전략을 논의했습니다."))
    summary = summarize_meeting_text(text)
    print(summary)


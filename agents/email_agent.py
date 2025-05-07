from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 환경변수 로드
load_dotenv()

# 전역에서 LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 사용자 입력
email_type = input("이메일 목적은 무엇인가요? ")
recipient = input("받는 사람은 누구인가요? ")
tone = input("어떤 말투를 원하나요? (예: 정중하게, 간결하게, 친근하게): ")
main_content = input("전하고 싶은 내용을 입력해주세요: ")

# 이메일 생성 함수
def generate_email() -> str:
    prompt = f"""
너는 신입사원을 위한 이메일 작성 보조 에이전트야. 아래 정보를 바탕으로 상황에 맞는 이메일을 작성해줘.

이메일 목적: {email_type}
받는 사람: {recipient}
말투: {tone}
주요 내용: {main_content}

한국의 기업 문화에 맞춰 이메일을 다음 형식으로 작성해줘:
1. 제목 (이메일 목적을 분명히 알 수 있게, 예: [요청], [보고], [안내], [사과])
2. 인사말 (예: 안녕하세요, 김다은입니다)
3. 본문 (간결하고 목적 위주로, 문장은 정중하면서도 실용적으로 작성. 핵심 목적을 먼저 밝힘 → 필요한 정보/배경 순서로 작성)
4. 마무리 인사 (예: 감사합니다. 김다은 드림)

문장은 자연스럽고, 기업 문화에 맞는 형식으로 작성해줘.
"""

    response = llm.invoke(prompt)
    return response.content



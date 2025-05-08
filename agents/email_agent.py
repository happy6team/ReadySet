from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 환경변수 로드
load_dotenv()

# 전역에서 LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 이메일 생성 함수
def generate_email(email_input: str) -> str:
    """이메일 생성 함수"""
    
    # 프롬프트 구성
    prompt = f"""
First, please extract the email purpose, recipient, tone, and main content from the user input below.
If certain information is missing, assume the email purpose is 'request', the recipient is 'manager', and the tone is 'respectful'.

User input: {email_input}

Based on the extracted information, please write an email for a new employee.
Please write the email in the following format according to Korean corporate culture:

Subject: (clearly indicating the email purpose, e.g.: [Request], [Report], [Notice], [Apology])

To: [extracted recipient name] 

Greeting: (e.g.: Hello, I am writing to request...)

Body: (concise and purpose-oriented, with sentences that are respectful yet practical. Reveal the core purpose first → then provide necessary information/background)

Closing remarks: (e.g.: Thank you for your consideration.)

From: (Your name e.g.: Kim Da-eun)

Please write the sentences naturally and in a format appropriate for corporate culture.

IMPORTANT: 
1. The recipient is the person mentioned in the user input (in this case: {email_input})
2. The sender is assumed to be "me" (not the recipient)
3. Please make sure to generate the final email in Korean language.
"""
    response = llm.invoke(prompt)
    return response.content


# LangGraph Supervisor용 invoke 함수 
def invoke(state: dict, config) -> dict:
    """LangGraph Supervisor용 invoke 함수"""
    # 입력 쿼리 가져오기
    email_input = state.get("input_query", "")

    # 설정에서 thread_id 가져오기
    thread_id = (
        getattr(config, "configurable", {}).get("thread_id")
        if hasattr(config, "configurable")
        else config.get("thread_id", "default")
    )

    # 이메일 생성 함수 호출
    generated_email = generate_email(email_input)  # self 제거하고 직접 함수 호출
    print(f"생성된 이메일:\n{generated_email}")

    # messages 누적
    new_messages = list(state.get("messages", []))  # 기존 메시지 유지
    new_messages.append(f"📧 생성된 이메일:\n{generated_email}")

    return {
        **state,
        "messages": new_messages,
        "thread_id": thread_id,
        "generated_email": generated_email  # 생성된 이메일 추가
    }


# 실행 예시 (main 부분)
if __name__ == "__main__":
    # 사용자 입력
    user_input = input("이메일 목적, 받는사람, 말투, 전하고 싶은 내용을 입력해주세요: ")
    
    # 이메일 생성
    email = generate_email(user_input)
    
    # 결과 출력
    print("\n===== 생성된 이메일 =====\n")
    print(email)
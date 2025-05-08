from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

class EmailAssistant:
    """이메일 작성을 도와주는 에이전트 클래스"""
    
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        """이메일 어시스턴트 초기화"""
        # 환경변수 로드
        load_dotenv()
        
        # LLM 모델 초기화
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
    def get_user_input(self):
        """사용자 입력 받기"""
        return input(
    "이메일 작성을 위한 내용을 반드시 입력해주세요.\n"
    "이메일 목적, 받는 사람, 말투, 전하고 싶은 내용: "
)
    
    def generate_email(self, email_input=None):
        """이메일 생성 함수"""
        # 입력이 없으면 사용자에게 요청
        if email_input is None:
            email_input = self.get_user_input()
        
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

        # LLM 호출 및 결과 반환
        response = self.llm.invoke(prompt)
        return response.content
    
    def run(self):
        """에이전트 실행"""
        # 이메일 생성
        email_content = self.generate_email()
        
        # 결과 출력
        print("\n===== 생성된 이메일 =====\n")
        print(email_content)
        
        return email_content

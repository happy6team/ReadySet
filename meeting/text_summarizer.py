# meeting_summarization/text_summarizer.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 에서 API 키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def summarize_meeting_text(text: str) -> str:
    prompt = f"""
You are the general project manager (PM) of a smart farming project. The following is a transcript of a team meeting.

Your job is to summarize the meeting in <strong>Korean</strong>, using the five sections below.  
<strong>You MUST include all five sections</strong>, even if a section was not discussed in the meeting.  
In that case, clearly state: "해당 내용에 대한 발언은 없었습니다."

Format the summary using <strong>HTML bold tags</strong> like <strong>...</strong> instead of Markdown (**).  
Use <br> tags to indicate paragraph breaks. Use bullet points (•) where needed.

Sections:
<strong>회의 주제:</strong> Summarize the main purpose or theme of the meeting.

<strong>주요 화자 발언:</strong> Highlight the core points made by the participants during the discussion.

<strong>논의된 기술 또는 이슈:</strong> List or explain the technical terms, models, or challenges that were brought up.

<strong>최종 결정 및 실행 항목:</strong> Clearly describe any decisions that were made and the specific tasks assigned.

<strong>신입 직원을 위한 생소한 용어:</strong> Provide a list of technical terms or acronyms that might be difficult for newcomers to understand.

Return the summary in fluent Korean, formatted as natural paragraphs.
\"\"\"{text}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    print("🧠 GPT 응답 전체:", response)

    return response.choices[0].message.content

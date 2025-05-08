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
You are an assistant helping to summarize Korean meeting transcripts.

Please:
1. Extract and summarize the key objectives, speakers, decisions, and action items from the following meeting text.
2. Return the result in fluent **Korean**, written in paragraph form.
3. Focus on clearly delivering what the meeting was about and what was decided.

Here is the meeting transcript:
\"\"\"{text}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

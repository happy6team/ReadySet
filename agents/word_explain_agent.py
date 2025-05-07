import os
import requests
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Tavily 웹 검색
def search_tavily(query: str, max_results: int = 3) -> list:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"query": query, "num_results": max_results}

    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    
    return [item["content"] for item in data.get("results", [])]

# 프롬프트 템플릿 로딩
def load_prompt_template() -> str:
    return """
You are a kind and professional mentor and senior engineer participating in the project: {project_name}.
This project is about: {project_explain}.

Below is a question from a new team member who is unfamiliar with the terminology used in this project.
Term: {term}
Web search results:
{web_result}

Please explain the term **in Korean**, in a simple and easy-to-understand way for a new employee.
Include relevant examples, and avoid overly technical language if possible.
"""

# 용어 설명 메인 함수
def explain_word(term: str, project_name: str, project_explain: str) -> str:
    search_results = search_tavily(term)
    web_context = "\n".join(search_results)

    prompt_template = load_prompt_template()
    full_prompt = prompt_template.format(
        project_name=project_name,
        project_explain=project_explain,
        term=term,
        web_result=web_context
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"당신은 친절하고 전문적인 {project_name} 멘토입니다."},
            {"role": "user", "content": full_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# LangGraph Supervisor용 invoke 함수
def invoke(state: dict, config: RunnableConfig) -> dict:
    term = state.get("input_query", "")
    project_name = state.get("project_name")
    project_explain = state.get("project_explain")

    assert term, "input_query (term) 값이 필요합니다."
    assert project_name and project_explain, "project_name과 project_explain은 필수입니다."

    thread_id = (
        getattr(config, "configurable", {}).get("thread_id")
        if hasattr(config, "configurable")
        else config.get("thread_id", "default")
    )

    explanation = explain_word(term, project_name, project_explain)

    # ✅ messages 누적
    new_messages = list(state.get("messages", []))  # 기존 메시지 유지
    new_messages.append(f"📘 용어 설명 결과:\n{explanation}")
    print(new_messages)
    return {
        **state,
        "messages": new_messages,
        "thread_id": thread_id
    }
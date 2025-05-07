import os
import requests
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from openai import OpenAI


load_dotenv()
client = OpenAI()

def search_tavily(query: str, max_results: int = 3) -> list:
    """
    Tavily API를 사용하여 웹 검색 수행
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY 환경 변수가 설정되지 않았습니다.")

    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"query": query, "num_results": max_results}

    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    
    return [item["content"] for item in data.get("results", [])]

def load_prompt_template() -> str:
    """
    GPT에게 전달할 영어 프롬프트 템플릿
    """
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

def explain_word(term: str) -> str:
    search_results = search_tavily(term)
    web_context = "\n".join(search_results)

    # 임시 하드코딩 (메인페이지 연동 전까지)
    project_name = "차세대 한국형 스마트팜 개발"
    project_explain = "차세대 한국형 스마트팜 기술개발 프로젝트는 4기관 19개 전담부서가 협업하여 핵심 요소 및 원천 기반기술의 확보를 위해 연구 역량을 집중하고 있고 국내 농업여건에 적합하게 기술수준별로 스마트팜 모델을 3가지 단계로 구분하여 개발을 추진하고 있다. 단계별 스마트팜은 1세대(편리성 증진), 2세대(생산성 향상-네덜란드추격형), 3세대(글로벌산업화-플랜트 수출형)로 구분되고 기술의 단계적 개발과 실용화 계획을 통해 노동력과 농자재의 사용을 줄이고, 생산성과 품질을 제고함으로 농가소득과 연계하며, 나아가 영농현장의 애로와 연관 산업의 문제를 동시에 해결해 간다는 계획이다."

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
            {"role": "system", "content": "당신은 친절하고 전문적인 한국어 멘토입니다."},
            {"role": "user", "content": full_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# LangGraph Supervisor용 함수
def invoke(state: dict, config: RunnableConfig) -> dict:
    term = state.get("term")
    thread_id = (
        getattr(config, "configurable", {}).get("thread_id")
        if hasattr(config, "configurable")
        else config.get("thread_id", "default")
    )

    explanation = explain_word(term)

    return {
        "term": term,
        "explanation": explanation,
        "thread_id": thread_id,
        "agent": "word_explain"
    }
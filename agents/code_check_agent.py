from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from vector_store.builder import ensure_code_rule_vector_db_exists

import os

# 최초 실행 시 벡터DB 생성 (없으면)
ensure_code_rule_vector_db_exists()

embedding_model = OpenAIEmbeddings()
global_vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="vector_store/db/code_rule_chroma"
)

# 코드 검수 LLM 체인 (최신 방식)
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate(
    input_variables=["code", "rules"],
    template="""
You are a code convention expert.
Below is a user's code and several related coding rules.

[User Code]
{code}

[Related Coding Rules]
{rules}

Please analyze whether this code violates any of the rules.
List clearly which rules are violated, and suggest how the code should be corrected. Explain in **Korean** so that a junior developer can easily understand.
"""
)
code_review_chain = prompt | llm 

# 코드 검수 실행 함수
def check_code(code: str) -> str:
    related_rules = global_vectorstore.similarity_search(code, k=3)
    rules_text = "\n\n".join([doc.page_content for doc in related_rules])
    return code_review_chain.invoke({"code": code, "rules": rules_text})


# LangGraph Supervisor용 invoke 함수
def invoke(state: dict, config) -> dict:
    code = state.get("input_query", "")
    thread_id = (
        getattr(config, "configurable", {}).get("thread_id")
        if hasattr(config, "configurable")
        else config.get("thread_id", "default")
    )

    feedback = check_code(code)
    # print(feedback)
    # ✅ messages 누적
    new_messages = list(state.get("messages", []))  # 기존 메시지 유지
    new_messages.append(f"🧠 코드 검수 결과:\n{feedback}")

    return {
        **state,
        "messages": new_messages,
        "thread_id": thread_id,
    }

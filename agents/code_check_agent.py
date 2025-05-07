from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os

#  VectorDB 초기화 함수 (최초 1회 실행용)
def initialize_code_rule_vectorstore():
    rules_file_path = "code_rules/coding_rules.txt"
    with open(rules_file_path, "r", encoding="utf-8") as f:
        raw_rules = f.read()

    rules_chunks = [chunk.strip() for chunk in raw_rules.split("\n\n") if chunk.strip()]
    documents = [Document(page_content=chunk) for chunk in rules_chunks]
    embedding_model = OpenAIEmbeddings()
    persist_path = "vector_store/code_rule_chroma"

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_path
    )
    vectorstore.persist()
    print(f" 코드 규칙 DB 저장 완료: {persist_path}")
    return vectorstore

# VectorStore 로딩 함수
def load_vectorstore():
    embedding = OpenAIEmbeddings()
    return Chroma(
        embedding_function=embedding,
        persist_directory="vector_store/code_rule_chroma"
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
    vs = load_vectorstore()
    related_rules = vs.similarity_search(code, k=3)
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
    print(feedback)
    # ✅ messages 누적
    new_messages = list(state.get("messages", []))  # 기존 메시지 유지
    new_messages.append(f"🧠 코드 검수 결과:\n{feedback}")

    return {
        **state,
        "messages": new_messages,
        "thread_id": thread_id,
    }

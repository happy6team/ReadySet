from langchain_core.runnables.config import RunnableConfig

def generate_fallback_response(message: str) -> str:
    return f"""
죄송합니다. 현재 질문은 아래의 기능 범주 중 어느 하나에도 정확히 해당하지 않아 답변을 생성할 수 없습니다.

현재 지원되는 기능은 다음과 같습니다:
1. 용어 설명: 프로젝트 관련 용어나 개념을 쉽게 설명
2. 코드 검수: 작성한 코드의 규칙 위반 및 개선 사항 분석
3. 이메일 작성 가이드: 상황에 맞는 이메일 예시 제공
4. 담당자 매칭: 특정 업무 관련 담당자 안내
5. 문서 검색: 계획서, 회의록 등 사내 문서 검색
6. 보고서 작성 도우미: 보고서의 필드를 어떻게 작성하면 좋을지 가이드 제공

보다 정확한 도움을 드릴 수 있도록, 질문을 다시 구체적으로 작성해주시겠어요?

[입력된 질문: "{message}"]
""".strip()

def invoke(state: dict, config: RunnableConfig) -> dict:
    input_query = state.get("input_query", "")
    thread_id = (
        getattr(config, "configurable", {}).get("thread_id")
        if hasattr(config, "configurable")
        else config.get("thread_id", "default")
    )

    fallback_answer = generate_fallback_response(input_query)

    # ✅ messages 누적
    new_messages = list(state.get("messages", []))
    new_messages.append(f"❗ 예외 처리 결과:\n{fallback_answer}")

    return {
        **state,
        "messages": new_messages,
        "agent": "exception_agent",
        "thread_id": thread_id
    }

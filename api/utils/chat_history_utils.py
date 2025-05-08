from typing import List, Optional, Any
import copy
from fastapi import FastAPI

# 히스토리 관련 유틸리티 함수
def get_thread_messages(app: FastAPI, thread_id: str) -> List[Any]:
    """특정 스레드의 메시지 히스토리 조회"""
    return copy.deepcopy(app.state.thread_message_history.get(thread_id, []))

def add_thread_messages(app: FastAPI, thread_id: str, messages: List[Any]) -> None:
    """특정 스레드의 메시지 히스토리 추가"""
    if thread_id not in app.state.thread_message_history:
        app.state.thread_message_history[thread_id] = []
    app.state.thread_message_history[thread_id].append(messages)

def get_thread_queries(app: FastAPI, thread_id: str) -> List[str]:
    """특정 스레드의 쿼리 히스토리 조회"""
    return copy.deepcopy(app.state.thread_query_history.get(thread_id, []))

def add_thread_query(app: FastAPI, thread_id: str, query: str) -> None:
    """특정 스레드에 새 쿼리 추가"""
    if thread_id not in app.state.thread_query_history:
        app.state.thread_query_history[thread_id] = []
    app.state.thread_query_history[thread_id].append(query)


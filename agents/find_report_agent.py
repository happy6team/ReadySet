import os
from typing import Dict, Any, List, Optional, Callable, TypeVar, cast
from copy import deepcopy
from dotenv import load_dotenv
import openai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables.config import RunnableConfig
from agent_state import AgentState

class FindReportAgent:
    """
    사용자 질의를 통해 관련 문서를 검색해오는 에이전트
    """
    
    def __init__(
        self,
        db_path: str = "./vector_store/db",
        embedding_model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        k: int = 5
    ):
        """
        검색 에이전트 초기화
        
        Args:
            db_path: 벡터 DB 경로
            embedding_model_name: 임베딩 모델 이름
            openai_model: OpenAI 모델 이름
            temperature: 생성 온도
            k: 검색할 문서 수
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.openai_model = openai_model
        self.temperature = temperature
        self.k = k
        
        # OpenAI API 키 로드
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # OpenAI 클라이언트 설정
        openai.api_key = self.openai_api_key
        
        # 임베딩 모델 및 벡터 DB 초기화
        self._init_vector_db()
    
    def _init_vector_db(self) -> None:
        print("✅ findReportAgent 초기화 완료")
        """벡터 DB 및 임베딩 모델 초기화"""
        try:
            # 임베딩 모델 초기화
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # 벡터 DB 경로 확인
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Vector DB not found at {self.db_path}")
            
            # 벡터 DB 초기화
            self.vectordb = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            print(f"Vector DB loaded from {self.db_path}")
            
        except Exception as e:
            self.vectordb = None
            self.embeddings = None
            print(f"Error initializing vector DB: {str(e)}")
            raise

    def generate_response(self, query: str, context: str) -> str:
        """
        OpenAI API를 사용하여 응답 생성
        
        Args:
            query: 사용자 질문
            context: 검색된 문서 컨텍스트
            
        Returns:
            str: 생성된 응답
        """
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": (
                "You are a project manager who understands the project deeply. "
                "Please answer based on the provided documents. "
                "Respond **only** in Korean."
            )},
                    {"role": "user", "content": f"질문: {query}\n\n관련 문서:\n{context}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"답변 생성 과정에서 오류가 발생했습니다: {str(e)}"

    def search_documents(self, query: str) -> Dict[str, Any]:

        if not self.vectordb:
            return {
                "answer": "벡터 DB가 초기화되지 않았습니다.",
                "sources": [],
                "success": False
            }

        try:

            # ✅ 검색 문서 수 제한 (정확도는 유지)
            top_k = min(self.k, 3)  # 기본값은 3
            retrieved_docs = self.vectordb.similarity_search(query=query, k=top_k)


            # ✅ 상위 문서만 context로 사용
            context_entries = []
            sources = []

            for i, doc in enumerate(retrieved_docs):
                source_path = doc.metadata.get('source', 'Unknown')
                section = doc.metadata.get('section',
                                        doc.metadata.get('dl_meta', {}).get('headings', ['미분류 섹션'])[0]
                                        if 'dl_meta' in doc.metadata else '미분류 섹션')
                filename = os.path.basename(source_path) if isinstance(source_path, str) else 'Unknown'

                context_entry = f"[문서 {i+1}] 출처: {filename}, 섹션: {section}\n{doc.page_content}"
                context_entries.append(context_entry)

                source_info = {
                    "content": doc.page_content,
                    "section": section,
                    "source": source_path,
                    "filename": filename,
                    "rank": i + 1
                }
                sources.append(source_info)

            context = "\n\n".join(context_entries)

            # ✅ 문맥은 그대로 유지하여 정확도 보존
            answer = self.generate_response(query, context)

            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "success": True
            }

        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return {
                "answer": f"문서 검색 과정에서 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "context": "",
                "success": False
            }


    def format_agent_response(self, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 응답 형식 구성
        
        Args:
            search_result: 검색 결과
            
        Returns:
            Dict: 형식화된 응답
        """
        if search_result["success"]:
            message = {
                "answer": search_result["answer"],
                "sources": search_result["sources"][:5]  # 상위 5개 소스만 포함
            }
        else:
            message = {
                "error": search_result["answer"]
            }
            
        return {
            "messages": [message]
        }

    def invoke(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        에이전트 호출 메서드
        
        Args:
            state: 현재 상태
            config: 실행 설정
            
        Returns:
            AgentState: 업데이트된 상태
        """
        # 입력 쿼리 추출
        query = state.get("input_query", "")
        
        # 스레드 ID 추출
        thread_id = (
            getattr(config, "configurable", {}).get("thread_id")
            if hasattr(config, "configurable")
            else config.get("thread_id", "default")
        )
        
        # 문서 검색 수행
        search_result = self.search_documents(query)
        
        # 응답 형식화
        response = self.format_agent_response(search_result)
        
        new_messages = list(state.get("messages", []))
        new_messages.extend(response["messages"])

        # 상태 업데이트
        return {
            **state,
            "messages": new_messages,
            "agent": "search_agent",
            "thread_id": thread_id,
            "raw_search_result": search_result  # 디버깅 및 고급 사용을 위해 원시 검색 결과 포함
        }

# 에이전트 인스턴스 생성 및 함수 형태로 노출
def create_search_agent(
    db_path: str = "./vector_store/db",
    embedding_model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    openai_model: str = "gpt-4o-mini"
) -> Callable[[AgentState, RunnableConfig], AgentState]:
    """
    검색 에이전트 생성 함수
    
    Args:
        db_path: 벡터 DB 경로
        embedding_model_name: 임베딩 모델 이름
        openai_model: OpenAI 모델 이름
        
    Returns:
        Callable: 에이전트 호출 함수
    """
    agent = FindReportAgent(
        db_path=db_path,
        embedding_model_name=embedding_model_name,
        openai_model=openai_model
    )
    
    return agent

GLOBAL_AGENT = create_search_agent(
    db_path="./vector_store/db/reports_chroma",
    embedding_model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    openai_model="gpt-4o-mini"
)

# 편의를 위한 함수형 인터페이스
def invoke(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    함수형 인터페이스로 검색 에이전트 호출
    
    Args:
        state: 현재 상태
        config: 실행 설정
        
    Returns:
        AgentState: 업데이트된 상태
    """
    return GLOBAL_AGENT.invoke(state, config)


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 상태 및 설정
    test_state = {"message": "스마트팜 프로젝트의 단계별 추진 체계와 책임자는 누구인가요?"}
    test_config = RunnableConfig(configurable={"thread_id": "test-thread-123"})
    
    # 에이전트 호출
    result = invoke(test_state, test_config)
    
    # 결과 출력
    print("\n=== 검색 에이전트 응답 ===")
    messages = result.get("messages", [])
    if messages:
        if "answer" in messages[0]:
            print("\n답변:")
            print(messages[0]["answer"])
            
            print("\n참고 문서:")
            for i, source in enumerate(messages[0].get("sources", []), 1):
                print(f"\n[소스 {i}] - 섹션: {source.get('section', 'Unknown')}")
                print(f"출처: {source.get('filename', 'Unknown')}")
                print(f"내용: {source.get('content', '')[:150]}...")
        else:
            print("\n오류:")
            print(messages[0].get("error", "알 수 없는 오류"))
    else:
        print("응답 없음")
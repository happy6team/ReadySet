from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from typing import List, Dict, Any, Optional

def test_vector_retrieval(
    query: str, 
    k: int = 2, 
    db_path: str = "./vector_store/db", 
    embedding_model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
) -> List[Dict[str, Any]]:
    """
    벡터 DB에서 검색 기능을 테스트하는 함수
    
    Args:
        query: 검색 쿼리
        k: 검색할 문서 수
        db_path: 벡터 DB 경로
        embedding_model_name: 임베딩 모델 이름
        
    Returns:
        검색 결과 리스트 (각 항목은 내용과 메타데이터 포함)
    """
    try:
        # 임베딩 모델 초기화
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 벡터 DB가 존재하는지 확인
        if not os.path.exists(db_path):
            print(f"오류: 벡터 DB가 '{db_path}'에 존재하지 않습니다.")
            return []
        
        # 벡터 DB 로드
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        # 검색 수행
        results = vectordb.similarity_search(query, k=k)
        
        # 결과 형식화
        formatted_results = []
        for i, doc in enumerate(results):
            # 파일명 추출
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if isinstance(source, str) else 'Unknown'
            
            formatted_result = {
                "rank": i + 1,
                "content": doc.page_content,
                "source": source,
                "filename": filename,
                "section": doc.metadata.get('section', 'Unknown'),
                "chunk_id": doc.metadata.get('chunk_id', 'Unknown')
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    except Exception as e:
        print(f"검색 과정에서 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []

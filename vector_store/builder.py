import os
import argparse
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings



class VectorDatabaseBuilder:
    """벡터 데이터베이스 구축을 위한 클래스"""
    
    def __init__(
        self,
        embedding_model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        db_path: str = "./vector_store/db"
    ):
        """
        VectorDatabaseBuilder 초기화
        
        Args:
            embedding_model_name: 임베딩 모델 이름
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            db_path: 벡터 DB 저장 경로
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_path = db_path
        
        # 임베딩 모델 초기화
        self.embeddings = self._init_embedding_model()
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
    
    def _init_embedding_model(self):
        """임베딩 모델 초기화"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            print(f"임베딩 모델 '{self.embedding_model_name}' 초기화 완료")
            return embeddings
        except Exception as e:
            print(f"임베딩 모델 초기화 중 오류 발생: {e}")
            traceback.print_exc()
            raise
    
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """
        여러 문서 파일 로드
        
        Args:
            file_paths: 문서 파일 경로 목록
            
        Returns:
            로드된 문서 객체 리스트
        """
        all_docs = []
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
                
                loader = PyMuPDFLoader(file_path=file_path)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"파일 '{os.path.basename(file_path)}' 로드 완료: {len(docs)}개 문서 조각")
            
            except Exception as e:
                print(f"파일 '{file_path}' 로드 과정에서 오류 발생: {e}")
                traceback.print_exc()
        
        print(f"총 로드된 문서 조각: {len(all_docs)}개")
        return all_docs
    
    def process_documents(self, docs: List[Any]) -> tuple:
        """
        문서 처리 및 청크 생성
        
        Args:
            docs: 문서 객체 리스트
            
        Returns:
            texts, metadatas: 텍스트와 메타데이터 튜플
        """
        all_chunks = []
        
        try:
            # 각 문서 처리
            for i, doc in enumerate(docs):
                # 문서 메타데이터에서 섹션 정보 추출
                source_path = doc.metadata.get("source", "")
                source_filename = os.path.basename(source_path) if source_path else "unknown"
                
                # 기본 메타데이터 설정
                metadata = {
                    "source": source_path,
                    "filename": source_filename,
                    "page": doc.metadata.get("page", 0),
                    "doc_index": i
                }
                
                # 텍스트 분할
                text = doc.page_content
                chunks = self.text_splitter.split_text(text)
                
                # 각 청크에 메타데이터 추가
                for j, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_index": j,
                            "chunk_id": f"{i}_{j}"
                        }
                    })
            
            # 벡터화를 위한 텍스트와 메타데이터 준비
            texts = [chunk["text"] for chunk in all_chunks]
            metadatas = [chunk["metadata"] for chunk in all_chunks]
            
            print(f"총 {len(all_chunks)}개의 청크를 생성했습니다.")
            return texts, metadatas
        
        except Exception as e:
            print(f"텍스트 분할 과정에서 오류 발생: {e}")
            traceback.print_exc()
            raise
    
    def create_vector_db(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> Optional[Chroma]:
        """
        벡터 데이터베이스 생성
        
        Args:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트
            
        Returns:
            생성된 Chroma 객체 또는 None (오류 발생 시)
        """
        try:
            # 저장 디렉토리 생성
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # 기존 DB 확인
            if os.path.exists(self.db_path):
                print(f"경고: 기존 데이터베이스가 '{self.db_path}'에 존재합니다. 덮어쓰기를 진행합니다.")
            
            # 벡터 DB 생성
            vectordb = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=self.db_path
            )
            
            # 저장
            vectordb.persist()
            print(f"벡터 데이터베이스가 '{self.db_path}'에 저장되었습니다.")
            
            return vectordb
        
        except Exception as e:
            print(f"벡터 데이터베이스 생성 과정에서 오류 발생: {e}")
            traceback.print_exc()
            return None
    
    def build(self, file_paths: List[str]) -> Optional[Chroma]:
        """
        전체 벡터 DB 구축 프로세스 실행
        
        Args:
            file_paths: 문서 파일 경로 목록
            
        Returns:
            생성된 Chroma 객체 또는 None (오류 발생 시)
        """
        try:
            # 1. 문서 로드
            docs = self.load_documents(file_paths)
            if not docs:
                print("처리할 문서가 없습니다.")
                return None
            
            # 2. 문서 처리 및 청크 생성
            texts, metadatas = self.process_documents(docs)
            if not texts:
                print("생성된 청크가 없습니다.")
                return None
            
            # 3. 벡터 DB 생성 및 저장
            vectordb = self.create_vector_db(texts, metadatas)
            return vectordb
        
        except Exception as e:
            print(f"벡터 DB 구축 과정에서 오류 발생: {e}")
            traceback.print_exc()
            return None


def build_code_rule_vector_db(
    rule_file_path: str = "vector_store/docs/code_rules/coding_rules.txt",
    db_path: str = "vector_store/db/code_rule_chroma"
) -> Chroma:
    """
    코드 컨벤션 규칙 텍스트 기반 벡터 DB 생성

    Args:
        rule_file_path: 코드 규칙 텍스트 파일 경로
        db_path: 저장 경로

    Returns:
        Chroma 객체
    """
    if not os.path.exists(rule_file_path):
        raise FileNotFoundError(f"코드 규칙 파일이 존재하지 않습니다: {rule_file_path}")

    with open(rule_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
    documents = [Document(page_content=chunk) for chunk in chunks]

    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=db_path
    )
    vectorstore.persist()
    print(f"✅ 코드 규칙 벡터 DB 저장 완료: {db_path}")
    return vectorstore


def ensure_code_rule_vector_db_exists():
    """
    코드 규칙용 벡터 DB가 없으면 생성
    """
    db_path = "vector_store/db/code_rule_chroma"
    rule_file_path = "vector_store/docs/code_rules/coding_rules.txt"

    if os.path.exists(db_path) and os.path.isdir(db_path) and len(os.listdir(db_path)) > 0:
        print(f"✅ 코드 규칙 벡터 DB 이미 존재: {db_path}")
        return

    print("📦 코드 규칙 벡터 DB 생성 시작")
    build_code_rule_vector_db(rule_file_path=rule_file_path, db_path=db_path)

def ensure_vector_db_exists(db_path: str = "./vector_store/db", file_path: str="./docs"):
    # DB가 이미 존재하는지 확인
    if os.path.exists(db_path) and os.path.isdir(db_path) and len(os.listdir(db_path)) > 0:
        print(f"벡터 데이터베이스가 '{db_path}'에 이미 존재합니다.")
        return True
    
    print(f"벡터 데이터베이스가 존재하지 않습니다. 새로 구축합니다...")

    # 벡터 DB 빌더 생성
    builder = VectorDatabaseBuilder(
        embedding_model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        chunk_size=400,
        chunk_overlap=50,
        db_path=db_path
    )
    
    # 파일 경로를 리스트로 변환
    if isinstance(file_path, str):
        if os.path.isdir(file_path):
            # 디렉토리인 경우 모든 파일 경로를 수집
            file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) 
                         if os.path.isfile(os.path.join(file_path, f))]
        else:
            # 단일 파일인 경우
            file_paths = [file_path]
    else:
        file_paths = file_path
    
    # 벡터 DB 구축
    return builder.build(file_paths)

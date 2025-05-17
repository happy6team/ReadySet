import os
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncAttrs

# .env 파일 로드
load_dotenv()

# 환경 변수에서 DB 비밀번호 가져오기
DB_PASSWORD = os.getenv("DB_PASSWORD", "")  # 기본값은 빈 문자열
DB_USER = os.getenv("DB_USER", "root")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

# ✅ MySQL 비동기 데이터베이스 URL
ASYNC_DB_URL = f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
print(f"ASYNC_DB_URL: {ASYNC_DB_URL}") 

# ✅ 비동기 데이터베이스 엔진 생성
async_engine = create_async_engine(ASYNC_DB_URL, echo=True)

# ✅ 비동기 세션 팩토리 설정
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ✅ 비동기 지원을 위한 SQLAlchemy 기본 클래스
Base = declarative_base(cls=AsyncAttrs)  # ✅ AsyncAttrs 추가

# ✅ 비동기 데이터베이스 세션을 반환하는 종속성 함수
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session  # ✅ 비동기 세션 반환 (async with 사용)
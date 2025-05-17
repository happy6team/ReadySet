import json
import asyncio
import os
from datetime import datetime
from pathlib import Path
from models.human_resource import HumanResource

from config.db_config import Base, async_engine, AsyncSessionLocal

'''
신입사원 만 건 데이터를 데이터베이스에 저장합니다.
'''

# JSON 파일 경로
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
# 신입사원 만 건 데이터 
JSON_FILE_PATH = PROJECT_ROOT / "data" / "hr_employees_data.json"

async def create_tables():
    """테이블이 없으면 생성합니다."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("데이터베이스 테이블이 생성되었습니다.")

async def load_json_data():
    """JSON 파일에서 데이터를 읽어옵니다."""
    try:
        # 디렉토리 생성 확인
        os.makedirs(os.path.dirname(JSON_FILE_PATH), exist_ok=True)
        
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get('employees', []) if isinstance(data, dict) else data
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {JSON_FILE_PATH}")
        return []
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 올바르지 않습니다: {JSON_FILE_PATH}")
        return []
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return []

async def insert_employee_data(employees):
    """직원 데이터를 데이터베이스에 비동기적으로 삽입합니다."""
    try:
        async with AsyncSessionLocal() as session:
            for emp in employees:
                # 이미 존재하는 직원인지 확인
                existing_emp = await session.get(HumanResource, emp.get('id'))
                
                # 객체 생성 또는 업데이트
                if existing_emp:
                    # 기존 직원 정보 업데이트
                    existing_emp.name = emp.get('name')
                    existing_emp.position = emp.get('position')
                    existing_emp.department = emp.get('department')
                    existing_emp.join_date = emp.get('join_date')
                    existing_emp.skills = emp.get('skills', [])
                    existing_emp.projects = emp.get('projects', [])
                    existing_emp.education_degree = emp.get('education', {}).get('degree')
                    existing_emp.education_school = emp.get('education', {}).get('school')
                    existing_emp.education_graduation_year = emp.get('education', {}).get('graduation_year')
                    existing_emp.certifications = emp.get('certifications', [])
                    existing_emp.languages = emp.get('languages', [])
                    existing_emp.profile_summary = emp.get('profile_summary')
                    existing_emp.updated_at = datetime.now().isoformat()
                else:
                    # 새 직원 생성
                    new_emp = HumanResource(
                        id=emp.get('id'),
                        name=emp.get('name'),
                        position=emp.get('position'),
                        department=emp.get('department'),
                        join_date=emp.get('join_date'),
                        skills=emp.get('skills', []),
                        projects=emp.get('projects', []),
                        education_degree=emp.get('education', {}).get('degree'),
                        education_school=emp.get('education', {}).get('school'),
                        education_graduation_year=emp.get('education', {}).get('graduation_year'),
                        certifications=emp.get('certifications', []),
                        languages=emp.get('languages', []),
                        profile_summary=emp.get('profile_summary')
                    )
                    session.add(new_emp)
            
            # 트랜잭션 커밋
            await session.commit()
            print(f"{len(employees)}개의 레코드가 성공적으로 삽입/업데이트되었습니다.")
            
    except Exception as e:
        print(f"데이터 삽입 오류: {e}")
        raise

async def main():
    """메인 비동기 함수"""
    # 테이블 생성
    await create_tables()
    
    # JSON 데이터 로드 및 삽입
    employees = await load_json_data()
    if employees:
        await insert_employee_data(employees)
    else:
        print("삽입할 데이터가 없습니다.")

if __name__ == "__main__":
    asyncio.run(main())
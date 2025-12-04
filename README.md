# The State of Fashion — AI Insight Engine

McKinsey & BoF *The State of Fashion* 2021–2025 리포트를 토대로 
인사이트를 빠르게 탐색할 수 있도록 만든 RAG 기반 AI 리서치 시스템입니다.
PDF를 벡터화하여 Streamlit UI에서 실사용 가능한 형태로 제공합니다.

##### 주요기능
- `AI Report Search`: 질문을 입력하면 SoF 전체를 RAG로 검색해 답변
- `Keyword Analytics`: 연도별 핵심 키워드 Top 10 및 트렌드 시각화
- `Chapter Insights`: 연도×챕터별 키워드/타임라인/맵핑 분석
- `Regional Insights`: 국가별(US, China, EU, Japan, India) 2024–2025 인사이트 요약
- `Strategy Chat & Report`: 챗봇과 대화한 뒤 대화 로그 기반 리포트 생성

##### Models & RAG
- RAG 구조 : LangChain Retriever + Groq LLM 조합
- LLM: Groq `llama-3.1-8b-instant`  
- Embedding: HuggingFace `all-MiniLM-L6-v2`  
- Vector DB: FAISS  
- 문서 처리: PyPDFLoader + RecursiveCharacterTextSplitter  
- UI/배포: Streamlit, GitHub 

##### 파이프라인
1. SoF 2021–2025 PDF 로드 및 연도·챕터 메타데이터 부착  
2. Chunking → 임베딩 생성 → FAISS 벡터스토어 구축  
3. 질문 입력 시 관련 chunk 검색 후 LLM이 근거 기반 답변 생성  
4. Streamlit에서 Report Search / Keyword Trends / Insights 기능 제공 

##### Demo
- 실행 페이지 : https://fashionsofv2.streamlit.app/
    - Streamlit Cloud에 배포된 데모 페이지로 바로 실행 가능합니다.
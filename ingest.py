from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = Path("data")
PDF_FILES = sorted(DATA_DIR.glob("*.pdf"))

# 더 세밀한 컨텍스트를 위한 chunk 설정 (기존 1000 -> 800)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 160


def extract_year_from_filename(path: Path):
    # sof21.pdf -> 2021, sof2025.pdf -> 2025 이런 식으로 처리
    digits = re.findall(r"\d+", path.stem)
    if not digits:
        return None
    num = digits[-1]
    if len(num) == 2:
        return 2000 + int(num)
    elif len(num) == 4:
        return int(num)
    return None


def detect_chapter(text: str, current_chapter: str | None):
    """
    페이지 텍스트 안에서 Global Economy / Consumer Shifts / Fashion System
    같은 챕터 타이틀이 등장하면 그걸 기준으로 현재 챕터를 업데이트.
    """
    lower = text.lower()
    if "global economy" in lower:
        return "Global Economy"
    if "consumer shifts" in lower:
        return "Consumer Shifts"
    if "fashion system" in lower:
        return "Fashion System"
    # 못 찾으면 직전 챕터 유지
    return current_chapter


def detect_region(text: str, current_region: str | None):
    """
    간단한 룰 기반 region 태그 감지.
    SoF에서 자주 등장하는 주요 지역/국가 중심으로 태깅.
    """
    lower = text.lower()

    # 국가/지역 키워드 매핑
    if "japan" in lower:
        return "Japan"
    if "india" in lower:
        return "India"
    if "united states" in lower or "u.s." in lower or " u.s " in lower or " us " in lower:
        return "United States"
    if "china" in lower:
        return "China"
    if "european union" in lower or "eu " in lower or " europe" in lower:
        return "European Union"

    # 명시적인 지역 키워드가 없는 경우 기존 값 유지, 없으면 Global
    return current_region or "Global"


def load_pdfs_with_metadata():
    docs = []
    for pdf_path in PDF_FILES:
        year = extract_year_from_filename(pdf_path)
        print(f"📄 Loading {pdf_path.name} (year={year})")

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        current_chapter = None
        current_region = "Global"
        for page_doc in pages:
            # 챕터/리전 감지 & 메타데이터 부여
            current_chapter = detect_chapter(page_doc.page_content, current_chapter)
            current_region = detect_region(page_doc.page_content, current_region)

            page_doc.metadata["year"] = year
            page_doc.metadata["chapter"] = current_chapter
            page_doc.metadata["region"] = current_region
            docs.append(page_doc)

    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    print("✂️ Splitting documents into chunks ...")
    return splitter.split_documents(docs)


def build_vectorstore(splits):
    print("🧠 Loading embedding model…")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("📦 Building FAISS vectorstore…")
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local("faiss_index")
    print("✅ Saved vectorstore to ./faiss_index")


def main():
    if not PDF_FILES:
        print("❌ data/ 폴더에 PDF가 없습니다.")
        return

    docs = load_pdfs_with_metadata()
    splits = split_documents(docs)
    build_vectorstore(splits)


if __name__ == "__main__":
    main()

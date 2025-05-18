from pathlib import Path

import chromadb
import nltk
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings
import os
from nltk.tokenize import sent_tokenize
from pathlib import Path

nltk.download("punkt_tab")


# PDF에서 텍스트 추출
def extract_text_from_pdf(pdf_path: str) -> str:
    return extract_text(pdf_path)

# 텍스트 -> 문장 리스트
def split_into_sentences(text: str):
    return sent_tokenize(text)

# 문장 -> 벡터
def embed_sentences(sentences: list[str], model_name: str = "princeton-nlp/sup-simcse-roberta-large"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings

# ChromaDB에 저장
def store_to_chroma(file_id: str, sentences, embeddings, collection_name="my-docs"):
    client = chromadb.PersistentClient(path="./chroma")
    collection = client.get_or_create_collection(collection_name)

    documents = [
        {
            "id": f"{file_id}-{i}",
            "document": s,
            "embedding": e.tolist()
        }
        for i, (s, e) in enumerate(zip(sentences, embeddings))
    ]

    for doc in documents:
        collection.add(
            documents=[doc["document"]],
            embeddings=[doc["embedding"]],
            ids=[doc["id"]],
            metadatas=[{"source": file_id}] # pdf 파일 이름
        )

    # client.persist()

def split_into_chunks(text: str, chunk_size=500, chunk_overlap=100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    return splitter.split_text(text)

if __name__ == "__main__":

    data_folder = Path("/Users/kjh/rag-assistant/data")
    pdf_files = list(data_folder.glob("*.pdf"))

    print(f"[INFO] {len(pdf_files)}개 PDF 파일을 처리합니다.")

    for pdf_path in pdf_files:
        file_id = pdf_path.stem
        print(f"[INFO] 처리 중: {file_id}")

        text = extract_text_from_pdf(pdf_path)
        # sentences = split_into_sentences(text)
        sentences = split_into_chunks(text)

        if not sentences:
            print(f"[WARN] {file_id}에서 문장을 추출하지 못했어요.")
            continue

        embeddings = embed_sentences(sentences)
        store_to_chroma(file_id, sentences, embeddings)

        print(f"[OK] {file_id} 처리 완료. ({len(sentences)} 문장)")

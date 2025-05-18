from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# from huggingface_hub import login
#
#
# login("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")

# 1. 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_name="princeton-nlp/sup-simcse-roberta-large",
    encode_kwargs={"normalize_embeddings": True}
)

# 2. ChromaDB 벡터 스토어 연결
db = Chroma(
    persist_directory="./chroma",
    collection_name="my-docs",
    embedding_function=embedding_model,
)

# 3. 로컬 LLM 설정 (Ollama 실행 중이어야 함)
llm = OllamaLLM(model="mistral")

# 4. 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template("""
당신은 사용자의 개인 문서만 참고하여 답변하는 비서입니다.
문서에 없는 내용은 절대 만들어내지 마세요.
반드시 아래 문서를 참고해서만 답변하고, 모르면 "문서에서 찾을 수 없습니다"라고 답하세요.

문서:
{context}

질문:
{input}

답변:
""")


# 5. 문서 결합 체인 구성 (stuff 방식)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# 6. RAG 체인 구성

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5
    }
)

rag_chain = create_retrieval_chain(
    retriever,
    combine_docs_chain
)

# 7. 사용자 입력 루프
if __name__ == "__main__":

    while True:
        query = input("무엇이 궁금한가요? (종료하려면 'exit'): ")
        if query.lower() in ("exit", "quit"):
            break

        result = rag_chain.invoke({"input": query})

        print(f"\n🧠 답변: {result['answer']}\n")

        # 검색된 문서 정보 확인
        for i, doc in enumerate(result["context"]):
            print(f"[{i+1}] {doc.page_content[:300]}")



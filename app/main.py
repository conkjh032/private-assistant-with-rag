from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi.responses import StreamingResponse
import time
import json
import os

# FastAPI 앱 생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# 질문 입력 모델
class Question(BaseModel):
    question: str


class Message(BaseModel):
    role: str
    content: str


# Open WebUI에서 보내는 형식
class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: Optional[bool] = False



# 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="princeton-nlp/sup-simcse-roberta-large",
    encode_kwargs={"normalize_embeddings": True}
)

# 벡터DB 연결
vectorstore = Chroma(
    persist_directory="./app/chroma",
    collection_name="my-docs",
    embedding_function=embedding_model
)

# LLM 설정
llm = OllamaLLM(model="mistral")

# 프롬프트
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

# 문서 결합 체인
combine_docs_chain = create_stuff_documents_chain(
    llm,
    prompt
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5
    }
)

# 최종 RAG 체인
rag_chain = create_retrieval_chain(
    retriever,
    combine_docs_chain
)


# POST /ask 엔드포인트
@app.post("/ask")
async def ask(question: Question):
    result = rag_chain.invoke({"input": question.question})
    return {"answer": result["answer"]}


@app.get("/v1/models")
async def get_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "custom-rag",
                "object": "model",
                "owned_by": "kjh"
            }
        ]
    })

# 비스트리밍 방식 : 응답을 한 번에 보여 주는 방법
# @app.post("/v1/chat/completions")
# async def chat_completion(req: ChatRequest):
#     user_message = next((m.content for m in req.messages if m.role == "user"), None)
#     if not user_message:
#         return JSONResponse(
#             status_code=400,
#             content={"error": "No user message found"}
#         )
#
#     result = rag_chain.invoke({"input": user_message})
#
#     return JSONResponse({
#         "id": "chatcmpl-rag",
#         "object": "chat.completion",
#         "created": 0,
#         "model": req.model,
#         "choices": [{
#             "index": 0,
#             "message": {"role": "assistant", "content": result["answer"]},
#             "finish_reason": "stop"
#         }],
#         "usage": {
#             "prompt_tokens": 0,
#             "completion_tokens": 0,
#             "total_tokens": 0
#         }
#     })


def generate_stream(content: str):
    for token in content:  # 실제 토크나이저 있으면 여기에 사용
        chunk = {
            "id": "chatcmpl-rag",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "custom-rag",
            "choices": [{
                "delta": {"content": token},
                "index": 0,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        time.sleep(0.02)  # 실제 스트림처럼 보이게 하려면 약간 딜레이

    # 마지막 마무리
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    user_msg = next((m.content for m in req.messages if m.role == "user"), None)
    result = rag_chain.invoke({"input": user_msg})
    print("resut : ", result)

    print("### RAG 기반 검색 결과 ###")
    for i, doc in enumerate(result.get("context", [])):
        print(f"[{i+1}] {doc.page_content[:200]}")

    return StreamingResponse(generate_stream(result["answer"]), media_type="text/event-stream")


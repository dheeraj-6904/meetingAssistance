
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from core.vector_store import build_vector_store, load_vector_store, get_retriever


def get_llm():
    return ChatMistralAI(
        model="voxtral-small-2507",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent meeting and video assistant.

Your task is to answer questions using the provided transcript context.

Instructions:
1. Use the transcript as your primary source of information.
2. You may interpret, summarize, and infer reasonable conclusions
   from the information present in the transcript.
3. For questions about the central idea, theme, or crux,
   provide an interpretation based on the overall context.
4. Do not require the answer to appear word-for-word in the transcript.
5. Do not invent facts that are unsupported by the transcript.
6. If the requested information is genuinely absent from the context,
   respond with:
   "I could not find this information in the meeting transcript."
7. Keep answers concise and precise.

Transcript context:
{context}"""
            ),
            ("human", "{question}")
        ]
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()

    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent meeting and video assistant.

Your task is to answer questions using the provided transcript context.

Instructions:
1. Use the transcript as your primary source of information.
2. You may interpret, summarize, and infer reasonable conclusions
   from the information present in the transcript.
3. For questions about the central idea, theme, or crux,
   provide an interpretation based on the overall context.
4. Do not require the answer to appear word-for-word in the transcript.
5. Do not invent facts that are unsupported by the transcript.
6. If the requested information is genuinely absent from the context,
   respond with:
   "I could not find this information in the meeting transcript."
7. Keep answers concise and precise.

Transcript context:
{context}"""
            ),
            ("human", "{question}")
        ]
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer
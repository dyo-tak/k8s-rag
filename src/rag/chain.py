"""The RAG chain: retrieve -> prompt -> LLM -> answer with citations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag.retrieval.retriever import format_context, retrieve

SYSTEM = """You are a Kubernetes support assistant. Answer the question using ONLY the context below.
Cite the doc sections you used like [title :: section].
If the context doesn't contain the answer, say exactly: "I don't know — check https://kubernetes.io/docs/"
Do not guess."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "context:\n{context}\n\nquestion: {question}"),
])

def build_chain(llm):
    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_context(retrieve(x["question"])))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

"""RAG chain - wires the Retriever to an LLM via a grounded prompt.

Flow:

    User Question
        |
        v
    Retriever
        |
        v
    Top 4 Chunks
        |
        v
    Prompt Template
        |
        v
    OpenAI Chat Model
        |
        v
    Answer + Sources

Design rules kept from earlier phases:
  * The chain depends ONLY on the abstract ``BaseEmbedder`` / retriever
    abstractions - it can talk to ChatOpenAI but is structured so the
    chat model is injectable (so tests use a fake LLM).
  * Generation never sees anything other than what the retriever returns;
    the system prompt enforces "answer only from context, cite sources,
    say 'I couldn't find...' otherwise".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.retrievers.retriever import RetrievalResult, Retriever

# --- Prompt -------------------------------------------------------------
# Spec-defined system prompt. Static across projects - only the context
# changes per query.

SYSTEM_PROMPT = """You are an AI onboarding assistant for software projects.

Answer only using the provided context.

If the answer cannot be found in the context, reply:
"I couldn't find that information in the project documentation."

Do not invent information.

At the end of every answer, list the source files you used."""

CONTEXT_BLOCK_TEMPLATE = """Context from the project documentation:

{context}

Question: {question}

Answer the question using ONLY the context above. If the context does not
contain the answer, say "I couldn't find that information in the project
documentation." At the end, list the source files you used."""


def format_context(hits: list[RetrievalResult]) -> str:
    """Render retrieved chunks into a single, source-labelled string.

    Each chunk is numbered and labelled with its source path so the
    LLM can reference it (and so we can map citations back later).
    """
    if not hits:
        return "(no relevant context was retrieved)"
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[{i}] source: {h.source}\n"
            f"    {h.page_content}"
        )
    return "\n\n".join(parts)


def sources_used(hits: list[RetrievalResult]) -> list[str]:
    """Unique, order-preserving list of source files feeding the answer
    (used for citations + the API response payload).
    """
    seen: dict[str, None] = {}
    for h in hits:
        if h.source not in seen:
            seen[h.source] = None
    return list(seen.keys())


@dataclass
class AnswerResult:
    """What ``answer_question`` returns - answer text plus provenance."""

    answer: str
    sources: list[str]
    retrieved: list[RetrievalResult]


class RagChain:
    """Compose Retriever + ChatModel + the grounded prompt.

    The chat model is injected, so a test can pass a fake; production
    passes a ``ChatOpenAI`` instance (built in ``app/main.py``).
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseChatModel,
    ):
        self._retriever = retriever
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", CONTEXT_BLOCK_TEMPLATE),
            ]
        )
        # The LangChain LCEL chain itself: build context -> prompt -> LLM
        # -> str. Keeping it as a composed runnable lets us reuse it for
        # streaming + async later without rewriting logic.
        self._chain = (
            RunnableLambda(self._retrieve_and_format)
            | self._prompt
            | self._llm
            | StrOutputParser()
        )

    # -- public -----------------------------------------------------------
    def answer(
        self,
        project_id: str,
        question: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AnswerResult:
        """End-to-end: retrieve top-k chunks, ask the LLM, return answer + sources."""
        hits = self._retriever.retrieve(
            project_id=project_id, query=question, k=k, filters=filters
        )
        # Use the composed chain for the actual generation step. We feed
        # both context and question via a dict; RunnablePassthrough keeps
        # the streams clean.
        answer_text = self._chain.invoke(
            {
                "project_id": project_id,
                "question": question,
                "k": k,
                "filters": filters,
            }
        )
        return AnswerResult(
            answer=answer_text,
            sources=sources_used(hits),
            retrieved=hits,
        )

    # -- streaming variant (used by the FastAPI SSE endpoint later) ------
    def stream(
        self,
        project_id: str,
        question: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ):
        """Yield answer text tokens, then ``sources`` are returned
        separately via the public ``last_sources()`` accessor. Caller
        gathers tokens; citations stay structured."""
        # We do retrieval synchronously once, before streaming generation,
        # so the citations are stable regardless of LLM token order.
        hits = self._retriever.retrieve(
            project_id=project_id, query=question, k=k, filters=filters
        )
        self._last_sources = sources_used(hits)
        # Build a one-shot chain that reuses the pre-fetched context.
        gen = (
            RunnablePassthrough()
            | self._prompt
            | self._llm
            | StrOutputParser()
        )
        context_block = CONTEXT_BLOCK_TEMPLATE.format(
            context=format_context(hits),
            question=question,
        )
        for chunk in gen.stream({"context": None, "context_block": context_block}):
            yield chunk

    def last_sources(self) -> list[str]:
        return list(getattr(self, "_last_sources", []))

    # -- internals --------------------------------------------------------
    def _retrieve_and_format(self, payload: dict[str, Any]) -> dict[str, Any]:
        """LCEL-compatible step: do retrieval, return ``{context, question}``.

        Retrieval happens *here* (not in the caller) so the whole chain
        is self-contained: invoke ``self._chain.invoke({...})`` and you
        get an answer back, with retrieval, prompt, generation all
        happening in order.
        """
        hits = self._retriever.retrieve(
            project_id=payload["project_id"],
            query=payload["question"],
            k=payload.get("k"),
            filters=payload.get("filters"),
        )
        # Stash so ``answer()`` can attach citations without double-retrieving.
        return {
            "context": format_context(hits),
            "question": payload["question"],
            "_hits": hits,
        }

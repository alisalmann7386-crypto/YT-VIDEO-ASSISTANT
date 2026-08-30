import os

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


# ============================================================
# 1. CREATE TIMESTAMPED DOCUMENTS
# ============================================================

def prepare_documents(segments, group_size=5):

    documents = []

    if not isinstance(segments, list):
        return documents

    for i in range(0, len(segments), group_size):

        group = segments[i:i + group_size]

        texts = []
        valid_segments = []

        for seg in group:

            text = seg.get("text", "").strip()

            if not text:
                continue

            texts.append(text)
            valid_segments.append(seg)

        if not texts:
            continue

        start = valid_segments[0].get(
            "start",
            "00:00"
        )

        end = valid_segments[-1].get(
            "end",
            start
        )

        combined_text = " ".join(texts)

        documents.append(
            Document(
                page_content=combined_text,
                metadata={
                    "start": str(start),
                    "end": str(end),
                    "start_raw": valid_segments[0].get(
                        "start_raw",
                        0
                    ),
                    "end_raw": valid_segments[-1].get(
                        "end_raw",
                        0
                    )
                }
            )
        )

    return documents


# ============================================================
# 2. BUILD RAG
# ============================================================

def build_rag_chain(segments):

    documents = prepare_documents(
        segments,
        group_size=5
    )

    if not documents:
        raise ValueError(
            "No timestamped transcript segments found."
        )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    # ========================================================
    # FORMAT RETRIEVED DOCUMENTS
    # ========================================================

    def format_docs_with_timestamps(docs):

        formatted = []

        for doc in docs:

            start = doc.metadata.get(
                "start",
                "Unknown"
            )

            end = doc.metadata.get(
                "end",
                "Unknown"
            )

            formatted.append(
                f"[{start} - {end}]\n"
                f"{doc.page_content}"
            )

        return "\n\n---\n\n".join(formatted)

    # ========================================================
    # LLM
    # ========================================================

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.1
    )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
You are an AI assistant that answers questions
about a video using timestamped transcript excerpts.

The context contains transcript sections with timestamps
in this format:

[MM:SS - MM:SS]

IMPORTANT RULES:

1. Answer ONLY using the provided transcript context.

2. If the user asks:
   - "When was X discussed?"
   - "At what timestamp was X mentioned?"
   - "Where did they talk about X?"
   - "When did they explain X?"

   return the timestamp of the most relevant section.

3. If multiple sections discuss the topic, return all
   important timestamp ranges.

4. Give the topic and timestamp clearly.

5. Do not invent timestamps.

6. If the topic is not present in the retrieved context,
   say that you could not find it.

7. When possible, briefly explain what was discussed
   at that timestamp.

Example:

User:
"When was PCA discussed?"

Good answer:

"PCA was discussed around 12:40–14:15.
The speaker explained PCA as a dimensionality
reduction technique."

CONTEXT:

{context}
"""
        ),

        (
            "human",
            "{question}"
        )
    ])

    # ========================================================
    # RAG CHAIN
    # ========================================================

    rag_chain = (
        {
            "context": retriever | format_docs_with_timestamps,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================
# 3. ASK QUESTION
# ============================================================

def ask_question(rag_chain, question):

    try:

        response = rag_chain.invoke(
            question
        )

        return response

    except Exception as e:

        return (
            "An error occurred while querying "
            f"the video model: {str(e)}"
        )
import os
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# ==========================================
# 1. PYDANTIC SCHEMA DEFINITION
# ==========================================
class KeyInsightsSchema(BaseModel):
    """Schema for extracting structured insights from a transcript."""
    action_items: List[str] = Field(
        default_factory=lambda: ["None discussed"],
        description=(
            "Actionable tasks assigned to individuals or teams with clear responsibilities. "
            "If none are explicitly mentioned, return ['None discussed']."
        )
    )
    key_decisions: List[str] = Field(
        default_factory=lambda: ["None discussed"],
        description=(
            "Key choices, agreements, policies, or conclusions agreed upon. "
            "If none are explicitly mentioned, return ['None discussed']."
        )
    )
    open_questions: List[str] = Field(
        default_factory=lambda: ["None discussed"],
        description=(
            "Unresolved questions, pending issues, or topics needing further clarification. "
            "If none are explicitly mentioned, return ['None discussed']."
        )
    )


# ==========================================
# 2. MAIN EXTRACTION PIPELINE
# ==========================================
def extract_insights(transcript: str) -> dict:
    if not transcript or not transcript.strip():
        return {
            "action_items": ["No transcript content provided."],
            "key_decisions": ["No transcript content provided."],
            "open_questions": ["No transcript content provided."]
        }

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return {
            "action_items": ["Mistral API key missing from environment."],
            "key_decisions": ["Mistral API key missing from environment."],
            "open_questions": ["Mistral API key missing from environment."]
        }

    try:
        llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            api_key=api_key
        )

        structured_llm = llm.with_structured_output(KeyInsightsSchema)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an executive meeting assistant. Process the meeting transcript and extract "
                "Action Items, Key Decisions, and Open Questions.\n\n"
                "STRICT RULES:\n"
                "1. Keep items concise, objective, and specific.\n"
                "2. Do NOT return empty lists or null values.\n"
                "3. If a section has no relevant content in the transcript, populate the list with ['None discussed']."
            ),
            ("human", "TRANSCRIPT CONTENT:\n{transcript}")
        ])

        chain = prompt | structured_llm
        result: KeyInsightsSchema = chain.invoke({"transcript": transcript})

        return {
            "action_items": result.action_items if result.action_items else ["None discussed"],
            "key_decisions": result.key_decisions if result.key_decisions else ["None discussed"],
            "open_questions": result.open_questions if result.open_questions else ["None discussed"]
        }

    except Exception as e:
        return {
            "action_items": [f"Extraction failed: {str(e)}"],
            "key_decisions": [f"Extraction failed: {str(e)}"],
            "open_questions": [f"Extraction failed: {str(e)}"]
        }


# ==========================================
# 3. BACKWARD-COMPATIBLE WRAPPERS
# ==========================================
def extract_action_items(transcript: str) -> str:
    """Wrapper function to get action items as a formatted string."""
    return extract_insights(transcript)["action_items"]


def extract_key_decisions(transcript: str) -> str:
    """Wrapper function to get key decisions as a formatted string."""
    return extract_insights(transcript)["key_decisions"]


def extract_questions(transcript: str) -> str:
    """Wrapper function to get open questions as a formatted string."""
    return extract_insights(transcript)["open_questions"]
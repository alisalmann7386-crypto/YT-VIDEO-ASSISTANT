#Actionableitems , decision , questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from concurrent.futures import ThreadPoolExecutor
import os 


def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)


def extract_all(transcript: str) -> dict:
    """
    Run action-item, decision, and question extraction concurrently.

    These are three independent LLM calls over the same transcript that
    were previously invoked one after another — each one waiting on a full
    network round-trip to Mistral before the next started. Since they don't
    depend on each other, running them in parallel threads (this is
    network-bound I/O, so Python's GIL isn't a bottleneck here) cuts this
    stage's wall-clock time to roughly that of a single call instead of
    three stacked back to back.
    """
    with ThreadPoolExecutor(max_workers=3) as executor:
        action_future = executor.submit(extract_action_items, transcript)
        decision_future = executor.submit(extract_key_decisions, transcript)
        question_future = executor.submit(extract_questions, transcript)

        return {
            "action_items": action_future.result(),
            "key_decisions": decision_future.result(),
            "open_questions": question_future.result(),
        }
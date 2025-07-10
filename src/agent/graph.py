from __future__ import annotations

from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from typing import TypedDict

from langgraph.graph import StateGraph


class Configuration(TypedDict):
    my_configurable_param: str


class Email(TypedDict):
    sender: str
    subject: str
    body: str


class EmailState(MessagesState):
    email: Email
    email_category: Optional[str]
    spam_reason: Optional[str]
    is_spam: Optional[bool]
    email_draft: Optional[str]
    messages: Annotated[list, add_messages]


model = ChatOpenAI(temperature=0)


def read_email(state: EmailState):
    email = state["email"]
    print(
        "Alfred is processing an email from "
        f"{email['sender']} with subject: {email['subject']}"
    )
    return {}


def classify_email(state: EmailState):
    email = state["email"]

    prompt = f"""As Alfred the butler, analyze this email and determine if it is spam or legitimate.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

First, determine if this email is spam. If it is spam, explain why.
If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
"""

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text

    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]

    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages,
    }


def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")
    return {}


def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"

    prompt = f"""As Alfred the butler, draft a polite preliminary response to this email.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

This email has been categorized as: {category}

Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
"""

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]

    return {"email_draft": response.content, "messages": new_messages}


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]

    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state["email_draft"])
    print("=" * 50 + "\n")

    return {}


def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"


email_graph = (
    StateGraph(EmailState, input_schema=MessagesState)
    .add_node("read_email", read_email)
    .add_node("classify_email", classify_email)
    .add_node("handle_spam", handle_spam)
    .add_node("draft_response", draft_response)
    .add_node("notify_mr_hugg", notify_mr_hugg)
    .add_edge(START, "read_email")
    .add_edge("read_email", "classify_email")
    .add_conditional_edges(
        "classify_email",
        route_email,
        {"spam": "handle_spam", "legitimate": "draft_response"},
    )
    .add_edge("handle_spam", END)
    .add_edge("draft_response", "notify_mr_hugg")
    .add_edge("notify_mr_hugg", END)
)

compiled_graph = email_graph.compile(name="Hugging face Email Assistant")

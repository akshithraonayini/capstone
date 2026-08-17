import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from state import GraphState


load_dotenv()


def response_agent(state: GraphState) -> GraphState:

    query = state["query"]
    contexts = state.get("retrieved_context", [])
    messages = state.get("messages", [])

    print("\n========== RESPONSE AGENT ==========")

    context = "\n\n".join(
        f"Context {i + 1}:\n{content}"
        for i, content in enumerate(contexts)
    )

    # Previous conversation
    history = ""

    for message in messages:
        if isinstance(message, HumanMessage):
            history += f"User: {message.content}\n"

        elif isinstance(message, AIMessage):
            history += f"Assistant: {message.content}\n"

    prompt = f"""
You are an Enterprise Knowledge Assistant.

Answer the user's question using ONLY the provided context.

You may use the previous conversation to understand references
such as "it", "that", "who approves it", etc.

If the answer cannot be found in the context, clearly say that
the information is not available in the provided knowledge base.

Do not invent or assume information.

Previous Conversation:
{history}

User Question:
{query}

Retrieved Context:
{context}

Provide a concise and accurate answer.
"""

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    response = llm.invoke(prompt)

    answer = response.content

    print("\nGenerated Answer:")
    print(answer)

    return {
        **state,
        "answer": answer,
        "messages": [
            *messages,
            HumanMessage(content=query),
            AIMessage(content=answer),
        ],
    }
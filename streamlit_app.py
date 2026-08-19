import os
import uuid

import certifi
import truststore

import streamlit as st


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

truststore.inject_into_ssl()

from src.graph import graph

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Enterprise Knowledge Assistant")
st.caption("Ask a question about company policies and documents.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show stored metadata (guardrail / RAGAS) if any
        meta = message.get("meta")

        if meta == "blocked":
            st.error(message.get("meta_text", "Blocked by guardrail."))

        elif meta == "ragas":
            st.caption(message.get("meta_text", ""))

query = st.chat_input("Type your question...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)


    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = graph.invoke(
                {"query": query},
                config=config,
            )


        if result.get("guardrail_blocked", False):

            answer = result.get(
                "answer",
                "I can't process that request.",
            )

            reason = result.get(
                "guardrail_reason",
                "Policy violation.",
            )

            st.markdown(answer)
            st.error(f"Guardrail BLOCKED — {reason}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "meta": "blocked",
                    "meta_text": f"Guardrail BLOCKED — {reason}",
                }
            )

        else:

            answer = result.get(
                "answer",
                "I couldn't generate an answer.",
            )

            st.markdown(answer)

            meta_text = ""
            evaluation = result.get("evaluation")

            if evaluation:

                faithfulness = evaluation.get("faithfulness")
                relevancy = evaluation.get("answer_relevancy")

                meta_text = (
                    f"RAGAS — Faithfulness: {faithfulness} | "
                    f"Answer Relevancy: {relevancy}"
                )

                st.caption(meta_text)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "meta": "ragas" if meta_text else None,
                    "meta_text": meta_text,
                }
            )

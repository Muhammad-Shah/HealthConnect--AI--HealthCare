from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
import json
from pathlib import Path
from pprint import pprint, PrettyPrinter
from typing import Dict
from pydantic import BaseModel
from ingest import create_embeddings
import streamlit as st

from dotenv import find_dotenv, load_dotenv
import os

dotenv_path = '.env'
load_dotenv(dotenv_path)
GROQ_API = os.getenv('GROQ_API')


llm = ChatGroq(temperature=0.5,
               model_name="Llama3-70b-8192",
               api_key=GROQ_API,
               max_tokens=100,
               model_kwargs={
                   "top_p": 1,
                   "frequency_penalty": 0.0,
                   "presence_penalty": 0.0
               }
               )

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Check if retriever is already created
if "retriever" not in st.session_state:
    st.session_state.retriever = create_embeddings()

# Check if history aware retriever is already created
if "history_aware_retriever" not in st.session_state:
    st.session_state.history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, contextualize_q_prompt
    )

# history_aware_retriever = create_history_aware_retriever(
#     llm, st.session_state.retriever, contextualize_q_prompt
# )


qa_system_prompt = """
You are HealthConnect AI, a dedicated and helpful AI assistant focused on assisting patients by providing the best doctor recommendations. 

Welcome patients with this message:
"Hello! Welcome to HealthConnect AI, your personalized health assistant. I'm here to help you find the best doctor based on your needs and preferences."

### Instructions:

1. Location Preference:
   - Question: "In which city or region of Pakistan are you looking to find a doctor?"
   - Purpose: This will help us find a doctor in a specific location.

2. Type of Disease or Health Concern:
   - Question: "What type of disease or health concern are you seeking treatment for? (e.g., diabetes, cardiology, general check-up, etc.)"
   - Purpose: This will help us understand your health concern to recommend a suitable doctor.

3. Budget for Doctor's Fee:
   - Question: "What is your budget for the doctor's fee? Please specify an approximate range (e.g., PKR 500 - PKR 5000)."
   - Purpose: This will help us recommend doctors within your budget.

4. Doctor's Gender Preference:
   - Question: "Do you have a preference for the doctor's gender? (Male, Female, No preference)"
   - Purpose: This will help us understand your gender preference to recommend a suitable doctor.

### Additional Guidelines:

- Professionalism: Maintain a professional tone throughout the conversation.
- User Engagement: Keep the user engaged by responding promptly and showing empathy.
- Relevance: Only provide information that is directly related to the patient's queries and needs. Avoid sharing unrelated information.
- Conciseness: Provide clear, concise, and accurate responses.
- Clarity: Ensure that each question and response is easy to understand.
- Sequential Questioning: Ask the above questions one by one to avoid overwhelming the patient.

Once you have gathered all the necessary information from the patient, recommend the top 3 doctors that best suit the patient's preferences, including their profile links.

### Response Guidelines:

- Use the gathered information to provide concise and accurate recommendations.
- If you don't have the answer, simply state that you don't know.
- Keep your answers to three sentences maximum and ensure they are concise.

Context: 
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)


def serialize_history(history):
    serial_history = []
    for message in history:
        if message["role"] == "user":
            serial_history.append(HumanMessage(content=message["content"]))
        else:
            serial_history.append(AIMessage(content=message["content"]))
    return serial_history


def generate_response(question, history):
    history = serialize_history(history)
    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": history})
    return ai_msg_1['answer']


def chat():
    """Manages a chat conversation between user and assistant."""

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Welcome to HealthConnect AI!🤖 Your personal healthcare assistant.👨‍⚕️ Get personalized doctor recommendations."}
        ]

    if prompt := st.chat_input("Say something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Replace with your response generation logic
                response = generate_response(
                    st.session_state.messages[-1]["content"], st.session_state.messages)
                print(st.session_state.messages)
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})


if __name__ == '__main__':
    chat()

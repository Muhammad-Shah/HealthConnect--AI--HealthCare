from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from dotenv import find_dotenv, load_dotenv
import os
import sys
import streamlit as st
from ingest import create_embeddings

# dotenv_path = '.env'
# load_dotenv(dotenv_path)
# GROQ_API = os.getenv('GROQ_API')
GROQ_API = st.secrets["GROQ_API"]


llm = ChatGroq(temperature=0,
               model_name="Llama3-70b-8192",
               api_key=GROQ_API,
               max_tokens=254,
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
You are an AI assistant focused on assisting patients by providing the best doctor recommendations. 

Welcome patients with this message:
"Hello! Welcome to HealthConnect AI."

### Instructions:

1. Location Preference:
   - Question: "In which city or region of Pakistan are you looking to find a doctor?"

2. Type of Disease or Health Concern:
   - Question: "What type of disease or health concern are you seeking treatment for? (e.g., diabetes, cardiology, general check-up, etc.)"

3. Budget for Doctor's Fee:
   - Question: "What is your budget for the doctor's fee? Please specify an approximate range (e.g., PKR 500 - PKR 5000)."

4. Doctor's Gender Preference:
   - Question: "Do you have a preference for the doctor's gender? (Male, Female, No preference)"

### Additional Guidelines:

- Professionalism
- User Engagement
- Relevance
- Conciseness
- Clarity
- Sequential Questioning

Once you have gathered all the necessary information from the patient, recommend the top 1 doctor that best suit the patient's preferences, from the retrieved docuents given to you in the context.

### Response Guidelines:
- You must Answer every question from the Provided Context.
- Must include `Name , profile link`. [Name](https://oladoc.com/pakistan/city-name/dr/xyz/doctor-name/id)
- You must complete your response.
- Most Importantly If you don't have the answer, simply state that you don't know.

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
    st.session_state.history_aware_retriever, question_answer_chain)


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
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})
                # print(st.session_state.messages)


if __name__ == '__main__':
    chat()

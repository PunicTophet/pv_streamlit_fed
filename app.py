import streamlit as st
pip install -r r3.txt
import pinecone
from sentence_transformers import SentenceTransformer
from azure.storage.blob import BlobServiceClient
import os
import re
import openai
import json
from pinecone import Pinecone
# Configuration


client = openai.OpenAI(api_key=OPENAI_API_KEY)

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pc_index = pc.Index("coa-query-2")

model = SentenceTransformer("all-MiniLM-L6-v2")
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

# Create the container client
container_client = blob_service_client.get_container_client("fed-coa-opinions")

def get_blob_data(id):
    # Remove .pdf_\d\d ending and replace with .json
    id = re.sub(r'\.pdf_\d\d$', '.json', id)
    
    blob_client = container_client.get_blob_client(id)
    
    try:
        data = blob_client.download_blob().readall()
        return json.loads(data)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def qa_app(x):
    b1 = []
    b2 = []
    sources = []
    q = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Your task is to restructure user questions to maximize the likelihood of relevant returns."},
            {"role": "user", "content": "Restructure the user's question for vectorization and search in a vector database."},
            {"role": "assistant", "content": x}
        ]
    ).choices[0].message.content
    
    for i in pc_index.query(vector=model.encode(q).tolist(), top_k=5, include_metadata=True)["matches"]:
        if i["score"] > .6:
            b1.append(i)
            sources.append(i["metadata"].get("Citation #", "No citation available"))
    
    if len(b1) > 0:
        for i in b1:
            blob_data = get_blob_data(i["id"])
            if blob_data and 'text' in blob_data:
                b2.append(client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "Your task is to summarize the facts, issues, and holdings of cases passed to you."},
                        {"role": "user", "content": blob_data['text']},
                        {"role": "assistant", "content": q}
                    ]
                ))
        if len(b2) > 0:
            summaries = [result.choices[0].message.content for result in b2]
            final_answer = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Your task is to answer the user's question based on the summaries of relevant cases provided."},
                    {"role": "user", "content": f"Original question: {q}\n\nSummaries of relevant cases:\n{summaries}"},
                    {"role": "assistant", "content": "Based on the summaries provided, here is my answer to your question:"}
                ]
            )
            return final_answer.choices[0].message.content, sources
        else:
            return "No relevant sources found. Please try rephrasing your query.", []
    else:
        return "No relevant sources found. Please try rephrasing your query.", []

# Streamlit app
st.title("Federal Court of Appeals Legal Q&A")

# User input
question = st.text_input("Enter your legal question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching for relevant cases and generating an answer..."):
            answer, sources = qa_app(question)
        
        st.subheader("Answer:")
        st.write(answer)
        
        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(source)
    else:
        st.warning("Please enter a question.")

# Optional: Add some information about the app
st.sidebar.title("About")
st.sidebar.info("This app uses AI to answer questions about Federal Court of Appeals cases. It searches through a database of case law to provide relevant information and citations.")

# Optional: Add a disclaimer
st.sidebar.warning("Disclaimer: This app provides information for educational purposes only and should not be considered legal advice. Always consult with a qualified legal professional for specific legal matters.")

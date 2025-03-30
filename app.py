import os
from flask import Flask, request, jsonify
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Flask App
app = Flask(__name__)

# Load FAQ Data
def load_faq_data(file_path="faq.txt"):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

# Process and store in FAISS
def setup_vector_store():
    documents = load_faq_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vector_store = setup_vector_store()
retriever = vector_store.as_retriever()

# LangChain QA Chain with Gemini
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    ),
    chain_type="stuff",
    retriever=retriever
)

# API Endpoint for Questions
class QuestionRequest(BaseModel):
    question: str

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question")
        if not question:
            return jsonify({"error": "Question is required"}), 400
        answer = qa_chain.run(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def build_rag_pipeline(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa

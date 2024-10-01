from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pymongo import MongoClient


client = MongoClient(mongodb_uri)
db = client[db_name]
collection = db[collection_name]

loader = PyPDFLoader("documents/mini-Brand-Guidelines.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vector_store = MongoDBAtlasVectorSearch.from_documents(
    splits, embeddings, collection=collection, index_name="vector_index"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def process_query(query):
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]

user_query = "What is this document about?"
answer, sources = process_query(user_query)

print(f"Answer: {answer}")
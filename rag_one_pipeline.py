from dotenv import load_dotenv
import os
import sys
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

try:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    mongodb_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv("COLLECTION_NAME")

    if not all([openai_api_key, mongodb_uri, db_name, collection_name]):
        raise ValueError("Missing required environment variables")

    try:
        client = MongoClient(mongodb_uri)
        client.admin.command('ping')  # Check if the connection is successful
        db = client[db_name]
        collection = db[collection_name]
    except (ConnectionFailure, ConfigurationError) as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

    try:
        loader = PyPDFLoader("documents/mini-Brand-Guidelines.pdf")
        documents = loader.load()
    except FileNotFoundError:
        raise FileNotFoundError("PDF file not found")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAIEmbeddings: {str(e)}")

    try:
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            splits, embeddings, collection=collection, index_name="vector_index"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatOpenAI: {str(e)}")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    def process_query(query):
        try:
            result = qa_chain({"query": query})
            return result["result"], result["source_documents"]
        except Exception as e:
            raise RuntimeError(f"Error processing query: {str(e)}")

    user_query = "What is this document about?"
    try:
        answer, sources = process_query(user_query)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {str(e)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    sys.exit(1)

finally:
    if 'client' in locals():
        client.close()
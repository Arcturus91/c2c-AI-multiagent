from dotenv import load_dotenv
import os
import sys
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

try:
    # Load environment variables
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    mongodb_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv("COLLECTION_NAME")

    if not all([openai_api_key, mongodb_uri, db_name, collection_name]):
        raise ValueError("Missing required environment variables")

    # Connect to MongoDB
    try:
        client = MongoClient(mongodb_uri)
        client.admin.command('ping')  # Check if the connection is successful
        db = client[db_name]
        collection = db[collection_name]
    except (ConnectionFailure, ConfigurationError) as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

    # Load and split documents
    try:
        loader = PyPDFDirectoryLoader("documents")
        documents = loader.load()
        print(f"Number of documents loaded: {len(documents)}")
    except FileNotFoundError:
        raise FileNotFoundError("PDF file not found")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    def check_vector_index(collection):
        indexes = collection.list_indexes()
        for index in indexes:
            if index.get('name') == 'vector_index_V0':
                return True
        return False

    # Initialize embeddings
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAIEmbeddings: {str(e)}")

    # Check for vector index and create embeddings if needed
    if not check_vector_index(collection):
        print("Vector index not found. Creating embeddings...")
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            splits, embeddings, collection=collection, index_name="vector_index_V0"
        )
        print("Embeddings created. Please create the vector index in MongoDB Atlas.")
        sys.exit(1)

    # If we get here, the index exists, so we can create the vector store
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index_V0"
    )

    # Set up retriever and LLM
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    try:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
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

    user_query = """You are an expert SEO-optimized article outliner for Class2Class. You will receive a topic  and some information for a blog article, which you must create an outline for, fitting for Class2Class' blog section on the website. The outline structure must always include: Topic/article title, description, aim of the article, main points of the content, CTA, and a list of the used SEO keywords, which you must always access through the attached "SEO Keywords" file, which you have access to in your knowledge, and this should be the only source for used SEO words, which should also be in bold. Always write your outlines considering a SEO optimized format, which is described in the rules section - also available in your knowledge. 

    __RULES for SEO optimized structure__
    MUST ALWAYS FOLLOW AND CONSIDER THESE INSTRUCTIONS FOR THE OUTLINE:

    Must directly or indirectly mention Class2Class
    Must access and use the knowledge file "SEO keywords" and mention at least 10 primary keywords, 5 secondary keywords and 3 long tail keywords in the article (marked bold in outline)
    Must sure the Focus Keywords are in the SEO Title.
    Must sure The Focus Keywords are in the SEO Meta Description.
    Make Sure The Focus Keywords appears in the first 10% of the content.
    Main content must be between 500-700 words
    Must include focus Keyword in the subheading(s).
    Must suggest 3 different titles.
    Titles must be short. 
    Must use a positive or a negative sentiment word in the Title.
    Must use a Power Keyword in the Title.
    Used SEO words must be written in a list
    You must mimic Class2Class' writing style, tone, voice and help them write SEO optimized articles in their communication style which is all accessible in your knowledge. The outline must also be adhering to their brand guidelines. 
    Your outlines focus on creating authentic, user-specific content for Class2Class website blogs and articles.

    Based on the documents you have access to, create an outline for a blog post about online education platforms."""

    # Process query
    try:
        answer = process_query(user_query)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error processing query: {str(e)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    sys.exit(1)

finally:
    if 'client' in locals():
        client.close()
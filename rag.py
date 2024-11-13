import os
import sys
import signal
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("API key not found. Please check your .env file.")
    sys.exit(1)

def signal_handler(sig, frame):
    print('\nThanks for using Gemini. :)')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Collection selection function
def select_collection():
    collection_names = [name for name in os.listdir('./chroma_db/') if os.path.isdir(f"./chroma_db/{name}")]
    if not collection_names:
        print("No collections found. Please add a collection first.")
        sys.exit(1)

    print("Select a collection to query:")
    for i, name in enumerate(collection_names, 1):
        print(f"{i}. {name}")
    
    selected = int(input("Enter the number of the collection: ").strip()) - 1
    collection_name = collection_names[selected] if 0 <= selected < len(collection_names) else collection_names[0]
    return collection_name

# Configure vector database
def get_vector_db(selected_collection):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=f"./chroma_db/{selected_collection}", embedding_function=embedding_function)

# Modify prompt to include metadata
def generate_rag_prompt(query, context, metadata):
    metadata_str = "\n".join([f"Source: {meta['source']} | Page: {meta.get('page_number', 'N/A')} | Section: {meta.get('section', 'N/A')}" for meta in metadata])
    context = context.replace("'", "").replace('"', "").replace("\n", " ")
    
    return (
        f"Rules and Regulations for the Construction and Classification of Steel Ships. answer the questions based on the context provided "
        # f"\n\n"
        f"QUESTION: '{query}'\n\nCONTEXT: '{context}'\n\n"
        f"Metadata: {metadata_str}\n\nANSWER:"
    )

# Retrieve context from vector database with metadata
def get_relevant_context_from_db(query, vector_db):
    try:
        search_results = vector_db.similarity_search(query, k=5)
        context = " ".join(result.page_content for result in search_results)
        metadata = [
            {
                "source": result.metadata.get("source", "N/A"),
                "page_number": result.metadata.get("page_number", "N/A"),
                "section": result.metadata.get("section", "N/A")
            }
            for result in search_results
        ]
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "", []
    
    return context.strip(), metadata

# Generate answer using Gemini API
def generate_answer(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        answer = model.generate_content(prompt)
        return answer.text
    except Exception as e:
        return f"Error generating answer: {e}"

# Select collection at startup
selected_collection = select_collection()
vector_db = get_vector_db(selected_collection)

# Main query loop with metadata handling
while True:
    print("\n-----------------------------------------------------------------------")
    query = input("What would you like to ask? ").strip()
    if not query:
        print("Query cannot be empty. Please enter a valid query.")
        continue

    # Retrieve context and metadata, and generate response
    context, metadata = get_relevant_context_from_db(query, vector_db)
    if not context:
        print("No relevant context found. Please ask another question.")
        continue

    # Generate RAG prompt and answer
    prompt = generate_rag_prompt(query=query, context=context, metadata=metadata)
    answer = generate_answer(prompt=prompt)
    
    # Print the answer and metadata (source, page, section)
    print(f"Answer: {answer}")
    print(f"Source Metadata: {metadata}")

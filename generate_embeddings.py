import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from unstructured.partition.docx import partition_docx
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize a text splitter for chunking PDF and DOCX content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Load PDF files and split them into chunks with metadata (page number)
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    documents = []
    
    for page_number, page in enumerate(pages, start=1):
        full_text = page.page_content
        chunks = text_splitter.split_text(full_text)
        
        # Add metadata with page number
        for chunk in chunks:
            metadata = {
                "source": file_path,
                "page_number": page_number
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents

# Load DOCX files and split them into chunks with metadata (section info)
def load_docx(file_path):
    elements = partition_docx(filename=file_path)
    full_text = "\n".join(element.text for element in elements if element.text)
    documents = []
    
    chunks = text_splitter.split_text(full_text)
    
    for chunk in chunks:
        # Here you can add additional metadata based on your needs (e.g., sections, headings)
        metadata = {
            "source": file_path,
            "section": "General Section"  # Or more specific section info based on your DOCX structure
        }
        documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents

# Load all PDF and DOCX files from a folder with chunking and metadata
def load_documents_from_folder(folder_path):
    docs = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.endswith('.pdf'):
                docs.extend(load_pdf(file_path))
            elif file_path.endswith('.docx'):
                docs.extend(load_docx(file_path))
    return docs

# Function to add documents in batches to Chroma
def add_documents_in_batches(vectorstore, documents, batch_size=5000):
    total_docs = len(documents)
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        try:
            # Attempt to add the batch to Chroma
            vectorstore.add_documents(batch)
            print(f"Added batch {i // batch_size + 1} of {total_docs // batch_size + 1}")
        except ValueError as e:
            print(f"Error adding batch {i // batch_size + 1}: {e}")
            continue  # Skip the problematic batch and move to the next

# Create or load a collection in Chroma
def get_or_create_collection():
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    base_path = "./chroma_db/"
    os.makedirs(base_path, exist_ok=True)
    existing_collections = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    if existing_collections:
        print("Existing collections:")
        for i, name in enumerate(existing_collections, 1):
            print(f"{i}. {name}")
        user_choice = input("Enter the collection number to use an existing collection or type 'new' to create a new one: ").strip()

        if user_choice.lower() == 'new':
            collection_name = input("Enter a new collection name: ").strip()
            collection_path = os.path.join(base_path, collection_name)
            return Chroma(embedding_function=embedding_function, persist_directory=collection_path)
        
        elif user_choice.isdigit() and 1 <= int(user_choice) <= len(existing_collections):
            collection_name = existing_collections[int(user_choice) - 1]
            print(f"Using existing collection: {collection_name}")
            collection_path = os.path.join(base_path, collection_name)
            return Chroma(embedding_function=embedding_function, persist_directory=collection_path)

        else:
            print("Invalid choice. Defaulting to the first existing collection.")
            collection_name = existing_collections[0]
            collection_path = os.path.join(base_path, collection_name)
            return Chroma(embedding_function=embedding_function, persist_directory=collection_path)
    else:
        collection_name = input("No existing collections found. Enter a new collection name: ").strip()
        collection_path = os.path.join(base_path, collection_name)
        return Chroma(embedding_function=embedding_function, persist_directory=collection_path)

# Prompt user for the folder containing files
documents_folder = input("Enter the folder path where files are stored: ").strip()

# Load documents from the specified folder with chunking and metadata
docs = load_documents_from_folder(documents_folder)

# Get or create a collection
vectorstore = get_or_create_collection()

# Only add documents that are not already in the vectorstore
existing_docs = vectorstore.get()["documents"]
new_docs = [doc for doc in docs if doc.page_content not in existing_docs]

# Add each chunked document to the vectorstore
if new_docs:
    add_documents_in_batches(vectorstore, new_docs)

# Print the count of documents in the collection to confirm
print("Total documents in collection:", vectorstore._collection.count())

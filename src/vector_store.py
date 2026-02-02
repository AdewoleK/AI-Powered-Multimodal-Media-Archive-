from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma 
from langchain.schema import Document 
from langchain_community.vectorstores.utils import filter_complex_metadata  
from config import *
from src.metadata_enhancer import extract_years  

embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

def clean_metadata_value(value):
    """Helper function to clean metadata values for ChromaDB compatibility"""
    if isinstance(value, list):
        if value:
            return str(value)  
        else:
            return ""  
    elif value is None:
        return ""  
    else:
        return value

def build_unified_vector_store(audio_segments, pdf_pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    documents = []

    # Audio chunks
    for seg in audio_segments:
        chunks = splitter.split_text(seg["text"])
        for chunk in chunks:
            years = extract_years(chunk)  
            
            # Create clean metadata
            metadata = {
                "source": AUDIO_FILE.split("/")[-1],
                "type": "audio",
                "timestamp": seg["timestamp"],
                "start_sec": seg["start"],
                "years_mentioned": clean_metadata_value(years)  
            }
            
            # Clean all metadata values
            clean_metadata = {k: clean_metadata_value(v) for k, v in metadata.items()}
            
            documents.append(Document(
                page_content=chunk,
                metadata=clean_metadata
            ))

    # PDF chunks
    for pg in pdf_pages:
        chunks = splitter.split_text(pg["text"])
        for chunk in chunks:
            years = extract_years(chunk)  
            
            # Create clean metadata
            metadata = {
                "source": PDF_FILE.split("/")[-1],
                "type": "text",
                "page": pg["page"],
                "years_mentioned": clean_metadata_value(years)  
            }
            
            # Clean all metadata values
            clean_metadata = {k: clean_metadata_value(v) for k, v in metadata.items()}
            
            documents.append(Document(
                page_content=chunk,
                metadata=clean_metadata
            ))

    # Filter complex metadata as additional safety
    filtered_documents = filter_complex_metadata(documents)

    vectorstore = Chroma.from_documents(
        documents=filtered_documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    return vectorstore, documents
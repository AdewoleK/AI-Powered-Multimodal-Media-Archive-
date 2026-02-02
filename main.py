from src.ingest_audio import transcribe_with_timestamps
from src.ingest_pdf import extract_pdf_with_pages
from src.vector_store import build_unified_vector_store
from src.rag_query import create_query_engine
from config import AUDIO_FILE, PDF_FILE, WHISPER_MODEL

if __name__ == "__main__":
    print("Transcribing audio...")
    audio_segments, _ = transcribe_with_timestamps(AUDIO_FILE, WHISPER_MODEL)

    print("Extracting PDF...")
    pdf_pages = extract_pdf_with_pages(PDF_FILE)

    print("Building vector store...")
    vectorstore, all_documents = build_unified_vector_store(audio_segments, pdf_pages)  

    print("Creating query engine with hybrid search...")
    query_archive = create_query_engine(vectorstore, all_documents)  

    # === Test Questions ===
    test_questions = [
        "What did the speaker say about urban planning?",
        "What was discussed in 2018?",
        "Compare mentions of technology in 2020 vs 2023",
        "Any reference to 1989?",
        "What does the document say about climate policy?"
    ]

    for question in test_questions:
        print("\n" + "="*60)
        print(f"Question: {question}")
        print("\nAnswer:")
        print(query_archive(question))
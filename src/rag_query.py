from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA 
from config import OLLAMA_LLM_MODEL
from src.hybrid_retriever import HybridRetriever  

def create_query_engine(vectorstore, all_documents):
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0.0)

    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    hybrid = HybridRetriever(all_documents, vector_retriever)
    retriever = hybrid.as_retriever()

    prompt_template = """You are an expert archival analyst. Answer the question using ONLY the provided context.

When comparing time periods (e.g., 2018 vs 2023), clearly separate evidence from each year.

Cite sources precisely:
- Audio: (Source: interview.mp3 @ 04:20-04:35)
- PDF: (Source: document.pdf, Page 12)

Include citations inline after relevant statements.

Context:
{context}

Question: {question}
Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    def query_archive(question: str):
        result = qa_chain.invoke({"query": question})
        answer = result["result"]

        sources = "\n\nSources:\n"
        seen = set()
        for doc in result["source_documents"]:
            meta = doc.metadata
            if meta["type"] == "audio":
                cite = f"• Audio: {meta['source']} @ {meta['timestamp']}"
            else:
                cite = f"• PDF: {meta['source']}, Page {meta['page']}"
            
            years = meta.get("years_mentioned", [])
            if years:
                cite += f" (mentions years: {', '.join(map(str, years))})"
            
            if cite not in seen:
                sources += cite + "\n"
                seen.add(cite)
        
        return answer + sources

    return query_archive
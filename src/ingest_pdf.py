import fitz

def extract_pdf_with_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():  
            pages.append({
                "text": text.strip(),
                "page": page_num + 1
            })
    return pages
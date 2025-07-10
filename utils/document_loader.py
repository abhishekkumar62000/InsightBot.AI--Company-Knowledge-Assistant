import os
from PyPDF2 import PdfReader
import docx2txt
try:
    from tika import parser as tika_parser
    TIKA_AVAILABLE = True
except ImportError:
    TIKA_AVAILABLE = False

def extract_text_from_pdf(file_path):
    text = ''
    # Try PyPDF2 first
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    except Exception:
        pass
    # Fallback to Tika if PyPDF2 fails or text is too short
    if (not text or len(text.strip()) < 50) and TIKA_AVAILABLE:
        try:
            parsed = tika_parser.from_file(file_path)
            tika_text = parsed.get('content', '')
            if tika_text and len(tika_text.strip()) > len(text.strip()):
                text = tika_text
        except Exception:
            pass
    return text.strip()

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError('Unsupported file type')

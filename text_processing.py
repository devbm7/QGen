import re
import pymupdf
from nltk.tokenize import sent_tokenize

def get_pdf_text(pdf_file):
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"[\n]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = text.replace('\xad', '')
    text = re.sub(r'[‒–—―]', '-', text)
    return text

# Function to create text chunks
def segment_text(text, max_segment_length=700, batch_size=7):
    sentences = sent_tokenize(text)
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        if len(current_segment) + len(sentence) <= max_segment_length:
            current_segment += sentence + " "
        else:
            segments.append(current_segment.strip())
            current_segment = sentence + " "
    
    if current_segment:
        segments.append(current_segment.strip())
    
    # Create batches
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    return batches 
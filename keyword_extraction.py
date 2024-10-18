from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from transformers import pipeline
from gliner import GLiNER
from load_models import load_nlp_models

nlp, s2v = load_nlp_models()

def filter_keywords(extracted_keywords):
    unwanted_keywords =[
    # Common punctuation marks
    '.', ',', '!', '?', ':', ';', '-', '_', '(', ')', '[', ']', '{', '}',
    '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>',
    '`', '~', '"', "'",
    
    # Common contractions (if not already removed as stopwords)
    "n't", "'s", "'m", "'re", "'ll", "'ve", "'d",
    
    # Common abbreviations
    'etc', 'eg', 'ie', 'ex', 'vs', 'viz',

    'tbd', 'tba',  # To be determined/announced
    'na', 'n/a',  # Not applicable

    # Single characters
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',

    # HTML-related tags (if the text contains any HTML content)
    '<html>', '</html>', '<body>', '</body>', '<head>', '</head>', '<div>', '</div>', '<p>', '</p>', '<br>', '<hr>', '<h1>', '</h1>', '<h2>', '</h2>', '<h3>', '</h3>',
    
    # Random technical or common abbreviations that aren't meaningful keywords
    'etc', 'e.g', 'i.e', 'vs', 'ex', 'vol', 'sec', 'pg', 'id', 'ref', 'eq', 

    # Miscellaneous tokens
    'www', 'com', 'http', 'https', 'ftp', 'pdf', 'doc', 'img', 'gif', 'jpeg', 'jpg', 'png', 'mp4', 'mp3', 'org', 'net', 'edu',
    'untitled', 'noname', 'unknown', 'undefined',

    # Single letters commonly used in bullet points or references
    'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii',
    
    # Common file extensions (if filenames are included in the text)
    '.jpg', '.png', '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.csv', '.txt', '.zip', '.tar', '.gz', '.exe', '.bat', '.sh', '.py', '.cpp', '.java',

    # Other tokens related to formatting or structure
    'chapter', 'section', 'figure', 'table', 'appendix', 

    # Miscellaneous general noise terms
    'note', 'item', 'items', 'number', 'numbers', 'figure', 'case', 'cases', 'example', 'examples', 'type', 'types', 'section', 'part', 'parts'
    ]
    # Convert both lists to sets for efficient lookup
    extracted_set = set(extracted_keywords)
    unwanted_set = set(unwanted_keywords)
    
    # Remove unwanted keywords
    filtered_keywords = extracted_set - unwanted_set
    
    # Convert back to a list and sort (optional)
    return sorted(list(filtered_keywords))


def remove_stopwords(keywords):
    stop_words = set(stopwords.words('english'))
    modified_keywords = [''.join(keyword.split()) for keyword in keywords]
    filtered_keywords = [keyword for keyword in modified_keywords if keyword.lower() not in stop_words]
    original_keywords = []
    for keyword in filtered_keywords:
        for original_keyword in keywords:
            if ''.join(original_keyword.split()).lower() == keyword.lower():
                original_keywords.append(original_keyword)
                break
    return original_keywords

def enhanced_ner(text):
    nlp = spacy.load("en_core_web_trf")
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    doc = nlp(text)
    spacy_entities = set((ent.text, ent.label_) for ent in doc.ents)
    hf_entities = set((ent['word'], ent['entity']) for ent in ner_pipeline(text))
    combined_entities = spacy_entities.union(hf_entities)
    keywords = [entity[0] for entity in combined_entities]
    return list(keywords)

def extract_keywords(text, extract_all):
    try:
        text = text.lower()
        enhanced_ner_entities = enhanced_ner(text)
        print("Enhanced ner entities: ",enhanced_ner_entities)
        enhanced_ner_entities = remove_stopwords(enhanced_ner_entities)
        enhanced_ner_entities = filter_keywords(enhanced_ner_entities)
        print("Enhanced ner entities after applying filter and stopwords removal: ",enhanced_ner_entities)

        gliner_model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
        labels = ["person", "organization", "phone number", "address", "email", "date of birth", 
                  "mobile phone number", "medication", "ip address", "email address", 
                  "landline phone number", "blood type", "digital signature", "postal code", 
                  "date"]
        entities = gliner_model.predict_entities(text, labels, threshold=0.5)
    
        gliner_keywords = set(remove_stopwords([ent["text"] for ent in entities]))
        print(f"Gliner keywords:{gliner_keywords}")

        # if extract_all is False:
        #     return list(gliner_keywords)
            
        doc = nlp(text)
        spacy_keywords = set(remove_stopwords([ent.text for ent in doc.ents]))
        print(f"\n\nSpacy Entities: {spacy_keywords} \n\n")  

        if extract_all is False:
            combined_keywords_without_all = list(spacy_keywords.union(gliner_keywords).union(enhanced_ner_entities))
            filtered_results = filter_keywords(combined_keywords_without_all)
            print("Keywords returned: ",filtered_results)
            return list(filtered_results)
        
        rake = Rake()
        rake.extract_keywords_from_text(text)
        rake_keywords = set(remove_stopwords(rake.get_ranked_phrases()))
        print(f"\n\nRake Keywords: {rake_keywords} \n\n")
        
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([text])
        tfidf_keywords = set(remove_stopwords(vectorizer.get_feature_names_out()))
        print(f"\n\nTFIDF Entities: {tfidf_keywords} \n\n")

        combined_keywords = list(rake_keywords.union(spacy_keywords).union(tfidf_keywords).union(gliner_keywords))
        filtered_results = filter_keywords(combined_keywords)
        print("Keywords returned: ",filtered_results)
        return list(filtered_results)
    
    except Exception as e:
        raise Exception(f"Error in keyword extraction: {str(e)}")

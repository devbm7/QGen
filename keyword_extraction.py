from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
# import spacy
from gliner import GLiNER
from load_models import load_nlp_models

nlp, s2v = load_nlp_models()

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

def extract_keywords(text, extract_all):
    try:
        text = text.lower()
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
            combined_keywords_without_all = list(spacy_keywords.union(gliner_keywords))
            print("Keywords returned: ",combined_keywords_without_all)
            return list(combined_keywords_without_all)
        
        rake = Rake()
        rake.extract_keywords_from_text(text)
        rake_keywords = set(remove_stopwords(rake.get_ranked_phrases()))
        print(f"\n\nRake Keywords: {rake_keywords} \n\n")
        
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([text])
        tfidf_keywords = set(remove_stopwords(vectorizer.get_feature_names_out()))
        print(f"\n\nTFIDF Entities: {tfidf_keywords} \n\n")

        combined_keywords = rake_keywords.union(spacy_keywords).union(tfidf_keywords).union(gliner_keywords)
        
        return list(combined_keywords)
    except Exception as e:
        raise Exception(f"Error in keyword extraction: {str(e)}")
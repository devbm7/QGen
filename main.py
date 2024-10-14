import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
from rake_nltk import Rake
import pandas as pd
from fpdf import FPDF
import wikipediaapi
from functools import lru_cache
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('brown')
from nltk.tokenize import sent_tokenize
nltk.download('wordnet')
from nltk.corpus import wordnet
import random
import sense2vec
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import os
from sentence_transformers import SentenceTransformer, util
import textstat
from spellchecker import SpellChecker
from transformers import pipeline
import re
import pymupdf
import uuid
import time
import asyncio
import aiohttp
from datetime import datetime
import base64
from io import BytesIO
# '-----------------'
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email import encoders
# '------------------'
from gliner import GLiNER
# -------------------
from langchain_community.llms import Ollama
import time
# '------------------'
from nltk.corpus import stopwords
print("***************************************************************")
llm = Ollama(model='llama3:latest')

st.set_page_config(
    page_icon='cyclone',
    page_title="Question Generator",
    initial_sidebar_state="auto",
    menu_items={
        "About" : "Hi this our project."
    }
)

st.set_option('deprecation.showPyplotGlobalUse',False)

class QuestionGenerationError(Exception):
    """Custom exception for question generation errors."""
    pass


# Initialize Wikipedia API with a user agent
user_agent = 'QGen/1.2'
wiki_wiki = wikipediaapi.Wikipedia(user_agent= user_agent,language='en')

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def initialize_state(session_id):
    if 'session_states' not in st.session_state:
        st.session_state.session_states = {}

    if session_id not in st.session_state.session_states:
        st.session_state.session_states[session_id] = {
            'generated_questions': [],
            # add other state variables as needed
        }
    return st.session_state.session_states[session_id]

def get_state(session_id):
    return st.session_state.session_states[session_id]

def set_state(session_id, key, value):
    st.session_state.session_states[session_id][key] = value


@st.cache_resource
def load_model(modelname):
    model_name = modelname
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load Spacy Model
@st.cache_resource
def load_nlp_models():
    nlp = spacy.load("en_core_web_md")
    s2v = sense2vec.Sense2Vec().from_disk('s2v_old')
    return nlp, s2v

# Load Quality Assurance Models
@st.cache_resource
def load_qa_models():
    # Initialize BERT model for sentence similarity
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    spell = SpellChecker()
    return similarity_model, spell

with st.sidebar:
    select_model = st.selectbox("Select Model", ("T5-large","T5-small"))
if select_model == "T5-large":
    modelname = "DevBM/t5-large-squad"
elif select_model == "T5-small":
    modelname = "AneriThakkar/flan-t5-small-finetuned"
nlp, s2v = load_nlp_models()
similarity_model, spell = load_qa_models()
context_model = similarity_model
model, tokenizer = load_model(modelname)


# Info Section
def display_info():
    st.sidebar.title("Information")
    st.sidebar.markdown("""
        ### Question Generator System
        This system is designed to generate questions based on the provided context. It uses various NLP techniques and models to:
        - Extract keywords from the text
        - Map keywords to sentences
        - Generate questions
        - Provide multiple choice options
        - Assess the quality of generated questions

        #### Key Features:
        - **Keyword Extraction:** Combines RAKE, TF-IDF, and spaCy for comprehensive keyword extraction.
        - **Question Generation:** Utilizes a pre-trained T5 model for generating questions.
        - **Options Generation:** Creates contextually relevant multiple-choice options.
        - **Question Assessment:** Scores questions based on relevance, complexity, and spelling correctness.
        - **Feedback Collection:** Allows users to rate the generated questions and provides statistics on feedback.

        #### Customization Options:
        - Number of beams for question generation
        - Context window size for mapping keywords to sentences
        - Number of questions to generate
        - Additional display elements (context, answer, options, entity link, QA scores)

        #### Outputs:
        - Generated questions with multiple-choice options
        - Download options for CSV and PDF formats
        - Visualization of overall scores

    """)

def get_pdf_text(pdf_file):
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# def save_feedback_og(question, answer, rating, options, context):
def save_feedback_og(feedback):

    feedback_file = 'feedback_data.json'
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []
    # tpl = {
    #     'question' : question,
    #     'answer' : answer,
    #     'context' : context,
    #     'options' : options,
    #     'rating' : rating,
    # }
    # feedback_data[question] = rating
    feedback_data.append(feedback)
    print(feedback_data)
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f)
    
    return feedback_file

# -----------------------------------------------------------------------------------------
def send_email_with_attachment(email_subject, email_body, recipient_emails, sender_email, sender_password, attachment):
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server
    smtp_port = 587  # Replace with your SMTP port

    # Create the email message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = ", ".join(recipient_emails) 
    message['Subject'] = email_subject
    message.attach(MIMEText(email_body, 'plain'))

    # Attach the feedback data if available
    if attachment:
        attachment_part = MIMEApplication(attachment.getvalue(), Name="feedback_data.json")
        attachment_part['Content-Disposition'] = f'attachment; filename="feedback_data.json"'
        message.attach(attachment_part)

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            print(sender_email)
            print(sender_password)
            server.login(sender_email, sender_password)
            text = message.as_string()
            server.sendmail(sender_email, recipient_emails, text)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False
# ----------------------------------------------------------------------------------

def collect_feedback(i,question, answer, context, options):
    st.write("Please provide feedback for this question:")
    edited_question = st.text_input("Enter improved question",value=question,key=f'fdx1{i}')
    clarity = st.slider("Clarity", 1, 5, 3, help="1 = Very unclear, 5 = Very clear",key=f'fdx2{i}')
    difficulty = st.slider("Difficulty", 1, 5, 3, help="1 = Very easy, 5 = Very difficult",key=f'fdx3{i}')
    relevance = st.slider("Relevance", 1, 5, 3, help="1 = Not relevant, 5 = Highly relevant",key=f'fdx4{i}')
    option_quality = st.slider("Quality of Options", 1, 5, 3, help="1 = Poor options, 5 = Excellent options",key=f'fdx5{i}')
    overall_rating = st.slider("Overall Rating", 1, 5, 3, help="1 = Poor, 5 = Excellent",key=f'fdx6{i}')
    comments = st.text_input("Additional Comments", "",key=f'fdx7{i}')

    if st.button("Submit Feedback",key=f'fdx8{i}'):
        feedback = {
            "context": context,
            "question": question,
            'edited_question':edited_question,
            "answer": answer,
            "options": options,
            "clarity": clarity,
            "difficulty": difficulty,
            "relevance": relevance,
            "option_quality": option_quality,
            "overall_rating": overall_rating,
            "comments": comments
        }
        # save_feedback(feedback)
        save_feedback_og(feedback)
        st.success("Thank you for your feedback!")

# def save_feedback(feedback):
#     st.session_state.feedback.append(feedback)
    # feedback_file = 'question_feedback.json'
    # with open(feedback_file, 'w') as f:
    #     json.dump(feedback, f)
    
    # return feedback_file


def analyze_feedback():
    if not st.session_state.feedback_data:
        st.warning("No feedback data available yet.")
        return

    df = pd.DataFrame(st.session_state.feedback_data)
    
    st.write("Feedback Analysis")
    st.write(f"Total feedback collected: {len(df)}")
    
    metrics = ['clarity', 'difficulty', 'relevance', 'option_quality', 'overall_rating']
    
    for metric in metrics:
        fig, ax = plt.subplots()
        df[metric].value_counts().sort_index().plot(kind='bar', ax=ax)
        plt.title(f"Distribution of {metric.capitalize()} Ratings")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        st.pyplot(fig)

    st.write("Average Ratings:")
    st.write(df[metrics].mean())

    # Word cloud of comments
    comments = " ".join(df['comments'])
    if len(comments) > 1:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments)
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)


def export_feedback_data():
    if not st.session_state.feedback_data:
        st.warning("No feedback data available.")
        return None

    # Convert feedback data to JSON
    json_data = json.dumps(st.session_state.feedback_data, indent=2)
    
    # Create a BytesIO object
    buffer = BytesIO()
    buffer.write(json_data.encode())
    buffer.seek(0)
    
    return buffer
# ---------------------------------------------------------------------
# Function to clean text 
# updated clean_text function
def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]", " ", text)  # Replace non-ASCII characters
    text = re.sub(r"[\n]", " ", text)  # Replace newline characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'[“”]', '"', text)  # Replace fancy double quotes with straight double quotes
    # Normalize curly single quotes to straight single quotes
    text = re.sub(r"[‘’]", "'", text)  # Replace fancy single quotes with straight single quotes
    text = text.replace('\xad', '')  # Remove soft hyphen
    text = re.sub(r'[‒–—―]', '-', text)  # Replace various dashes with a hyphen
    return text
# -----------------------------------------------------------------------
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

# Function to extract keywords using combined techniques
def extract_keywords(text, extract_all):
    try:
        text = text.lower()
        gliner_model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
        # labels = ["person", "organization", "email", "Award", "Date", "Competitions", "Teams", "location", "percentage", "money"]
        labels = ["person", "organization", "phone number", "address","email", "date of birth", 
                  "mobile phone number", "medication", "ip address", "email address", 
                  "landline phone number", "blood type", "digital signature", "postal code", 
                  "date"]
        entities = gliner_model.predict_entities(text, labels, threshold=0.5)
    
        gliner_keywords = set(remove_stopwords([ent["text"] for ent in entities]))  # Convert to set
        print(f"Gliner keywords:{gliner_keywords}")
        
        # Use Only Gliner Entities
        # if extract_all is False:
        #     return list(gliner_keywords)
            
        doc = nlp(text)
        spacy_keywords = set(remove_stopwords([ent.text for ent in doc.ents]))  # Convert to set
        print(f"\n\nSpacy Entities: {spacy_keywords} \n\n")  
        
        # Use spaCy for NER and POS tagging
        spacy_keywords.update([token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]])
        print(f"\n\nSpacy Keywords: {spacy_keywords} \n\n")
        
        if extract_all is False:
            spacy_keywords.union(gliner_keywords)
            return list(spacy_keywords)
        
        # Use RAKE
        rake = Rake()
        rake.extract_keywords_from_text(text)
        rake_keywords = set(remove_stopwords(rake.get_ranked_phrases()))  # Convert to set
        print(f"\n\nRake Keywords: {rake_keywords} \n\n")
        
        # Use TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([text])
        tfidf_keywords = set(remove_stopwords(vectorizer.get_feature_names_out()))  # Convert to set
        print(f"\n\nTFIDF Entities: {tfidf_keywords} \n\n")

        # Combine all keywords
        combined_keywords = rake_keywords.union(spacy_keywords).union(tfidf_keywords).union(gliner_keywords)
        
        return list(combined_keywords)
    except Exception as e:
        raise QuestionGenerationError(f"Error in keyword extraction: {str(e)}")

# # -----------------------------------------------------------
# def remove_stopwords(keywords):
#     stop_words = set(stopwords.words('english'))
#     modified_keywords = [''.join(keyword.split()) for keyword in keywords]
#     filtered_keywords = [keyword for keyword in modified_keywords if keyword.lower() not in stop_words]
#     original_keywords = []
#     for keyword in filtered_keywords:
#         for original_keyword in keywords:
#             if ''.join(original_keyword.split()).lower() == keyword.lower():
#                 original_keywords.append(original_keyword)
#                 break
    
#     return original_keywords



# # -----------------------------------------------------------
# # Function to extract keywords using combined techniques
# def extract_keywords(text, extract_all):
#     try:
#         gliner_model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
#         labels = ["person", "organization", "email", "Award", "Date", "Competitions", "Teams", "location", "percentage", "money"]
#         entities = gliner_model.predict_entities(text, labels, threshold=0.7)
    
#         gliner_keywords = list(set([ent["text"] for ent in entities]))
#         gliner_keywords = remove_stopwords(gliner_keywords)
#         print(f"Gliner keywords:{gliner_keywords}")
#         # Use Only Gliner Entities
#         if extract_all is False:
#             return list(gliner_keywords) 
            
#         doc = nlp(text)
#         spacy_keywords = set([ent.text for ent in doc.ents])
#         spacy_entities = remove_stopwords(spacy_keywords)
#         print(f"\n\nSpacy Entities: {spacy_entities} \n\n")  

#         # 
#         # if extract_all is False:
#         #     return list(spacy_entities) 
        
#         # Use RAKE
#         rake = Rake()
#         rake.extract_keywords_from_text(text)
#         rake_keywords = set(rake.get_ranked_phrases())
#         rake_keywords = remove_stopwords(rake_keywords)
#         print(f"\n\nRake Keywords: {rake_keywords} \n\n")
#         # Use spaCy for NER and POS tagging
#         spacy_keywords.update([token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]])
#         print(f"\n\nSpacy Keywords: {spacy_keywords} \n\n")
#         # Use TF-IDF
#         vectorizer = TfidfVectorizer(stop_words='english')
#         X = vectorizer.fit_transform([text])
#         tfidf_keywords = set(vectorizer.get_feature_names_out())
#         tfidf_keywords = remove_stopwords(tfidf_keywords)
#         print(f"\n\nTFIDF Entities: {tfidf_keywords} \n\n")

#         # Combine all keywords
#         combined_keywords = rake_keywords.union(spacy_keywords).union(tfidf_keywords).union(gliner_keywords)
        
#         return list(combined_keywords)
#     except Exception as e:
#         raise QuestionGenerationError(f"Error in keyword extraction: {str(e)}")

def get_similar_words_sense2vec(word, n=3):
    # Try to find the word with its most likely part-of-speech
    word_with_pos = word + "|NOUN"
    if word_with_pos in s2v:
        similar_words = s2v.most_similar(word_with_pos, n=n)
        return [word.split("|")[0] for word, _ in similar_words]
    
    # If not found, try without POS
    if word in s2v:
        similar_words = s2v.most_similar(word, n=n)
        return [word.split("|")[0] for word, _ in similar_words]
    
    return []

def get_synonyms(word, n=3):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word and lemma.name() not in synonyms:
                synonyms.append(lemma.name())
                if len(synonyms) == n:
                    return synonyms
    return synonyms

def gen_options(answer,context,question):
    prompt=f'''Given the following context, question, and correct answer, 
    generate {4} incorrect but plausible answer options. The options should be:
    1. Contextually related to the given context
    2. Grammatically consistent with the question
    3. Different from the correct answer
    4. Not explicitly mentioned in the given context

    Context: {context}
    Question: {question}
    Correct Answer: {answer}

    Provide the options in a semi colon-separated list. Output must contain only the options and nothing else.
    '''
    options= [answer]
    response = llm.invoke(prompt, stop=['<|eot_id|>'])
    incorrect_options = [option.strip() for option in response.split(';')]
    options.extend(incorrect_options)
    random.shuffle(options)
    print(options)
    return options
    # print(response)

def generate_options(answer, context, n=3):
    options = [answer]
    
    # Add contextually relevant words using a pre-trained model
    context_embedding = context_model.encode(context)
    answer_embedding = context_model.encode(answer)
    context_words = [token.text for token in nlp(context) if token.is_alpha and token.text.lower() != answer.lower()]

    # Compute similarity scores and sort context words
    similarity_scores = [util.pytorch_cos_sim(context_model.encode(word), answer_embedding).item() for word in context_words]
    sorted_context_words = [word for _, word in sorted(zip(similarity_scores, context_words), reverse=True)]
    options.extend(sorted_context_words[:n])

    # Try to get similar words based on sense2vec
    similar_words = get_similar_words_sense2vec(answer, n)
    options.extend(similar_words)
    
    # If we don't have enough options, try synonyms
    if len(options) < n + 1:
        synonyms = get_synonyms(answer, n - len(options) + 1)
        options.extend(synonyms)
    
    # If we still don't have enough options, extract other entities from the context
    if len(options) < n + 1:
        doc = nlp(context)
        entities = [ent.text for ent in doc.ents if ent.text.lower() != answer.lower()]
        options.extend(entities[:n - len(options) + 1])
    
    # If we still need more options, add some random words from the context
    if len(options) < n + 1:
        context_words = [token.text for token in nlp(context) if token.is_alpha and token.text.lower() != answer.lower()]
        options.extend(random.sample(context_words, min(n - len(options) + 1, len(context_words))))
    print(f"\n\nAll Possible Options: {options}\n\n")    
    # Ensure we have the correct number of unique options
    options = list(dict.fromkeys(options))[:n+1]
    
    # Shuffle the options
    random.shuffle(options)
    
    return options

# Function to map keywords to sentences with customizable context window size
def map_keywords_to_sentences(text, keywords, context_window_size):
    sentences = sent_tokenize(text)
    keyword_sentence_mapping = {}
    print(f"\n\nSentences: {sentences}\n\n")
    for keyword in keywords:
        for i, sentence in enumerate(sentences):
            if keyword in sentence:
                # Combine current sentence with surrounding sentences for context
                # start = max(0, i - context_window_size)
                # end = min(len(sentences), i + context_window_size + 1)
                start = max(0,i - context_window_size)
                context_sentenses = sentences[start:i+1]
                context = ' '.join(context_sentenses)
                # context = ' '.join(sentences[start:end])
                if keyword not in keyword_sentence_mapping:
                    keyword_sentence_mapping[keyword] = context
                else:
                    keyword_sentence_mapping[keyword] += ' ' + context
    return keyword_sentence_mapping


# Function to perform entity linking using Wikipedia API
@lru_cache(maxsize=128)
def entity_linking(keyword):
    page = wiki_wiki.page(keyword)
    if page.exists():
        return page.fullurl
    return None

async def generate_question_async(context, answer, num_beams):
    try:
        input_text = f"<context> {context} <answer> {answer}"
        print(f"\n{input_text}\n")
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        outputs = await asyncio.to_thread(model.generate, input_ids, num_beams=num_beams, early_stopping=True, max_length=250)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n{question}\n")
        # print(type(question))
        return question
    except Exception as e:
        raise QuestionGenerationError(f"Error in question generation: {str(e)}")

async def generate_options_async(answer, context, n=3):
    try:
        options = [answer]
        
        # Add contextually relevant words using a pre-trained model
        context_embedding = await asyncio.to_thread(context_model.encode, context)
        answer_embedding = await asyncio.to_thread(context_model.encode, answer)
        context_words = [token.text for token in nlp(context) if token.is_alpha and token.text.lower() != answer.lower()]

        # Compute similarity scores and sort context words
        similarity_scores = [util.pytorch_cos_sim(await asyncio.to_thread(context_model.encode, word), answer_embedding).item() for word in context_words]
        sorted_context_words = [word for _, word in sorted(zip(similarity_scores, context_words), reverse=True)]
        options.extend(sorted_context_words[:n])

        # Try to get similar words based on sense2vec
        similar_words = await asyncio.to_thread(get_similar_words_sense2vec, answer, n)
        options.extend(similar_words)
        
        # If we don't have enough options, try synonyms
        if len(options) < n + 1:
            synonyms = await asyncio.to_thread(get_synonyms, answer, n - len(options) + 1)
            options.extend(synonyms)
        
        # Ensure we have the correct number of unique options
        options = list(dict.fromkeys(options))[:n+1]
        
        # Shuffle the options
        random.shuffle(options)
        
        return options
    except Exception as e:
        raise QuestionGenerationError(f"Error in generating options: {str(e)}")


# Function to generate questions using beam search
async def generate_questions_async(text, num_questions, context_window_size, num_beams, extract_all_keywords):
    try:
        batches = segment_text(text)
        keywords = extract_keywords(text, extract_all_keywords)
        all_questions = []
        
        progress_bar = st.progress(0) 
        status_text = st.empty()
        print("Final keywords:",keywords)
        print("Number of questions that needs to be generated: ",num_questions)
        for i, batch in enumerate(batches):
            status_text.text(f"Processing batch {i+1} of {len(batches)}...")
            batch_questions = await process_batch(batch, keywords, context_window_size, num_beams,num_questions)
            all_questions.extend(batch_questions) 
            progress_bar.progress((i + 1) / len(batches))
            print("Length of the all questions list: ",len(all_questions))
            
            if len(all_questions) >= num_questions:
                break
        
        progress_bar.empty()
        status_text.empty()
        
        return all_questions[:num_questions]
    except QuestionGenerationError as e:
        st.error(f"An error occurred during question generation: {str(e)}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return []

async def generate_fill_in_the_blank_questions(context,answer):
    answerSize = len(answer)
    replacedBlanks = ""
    for i in range(answerSize):
        replacedBlanks += "_"
    blank_q = context.replace(answer,replacedBlanks)
    return blank_q

async def process_batch(batch, keywords, context_window_size, num_beams, num_questions):
    questions = []
    flag = False
    for text in batch:
        if flag:
            break
        keyword_sentence_mapping = map_keywords_to_sentences(text, keywords, context_window_size)
        for keyword, context in keyword_sentence_mapping.items():
            print("Length of questions list from process batch function: ",len(questions))
            if len(questions)>=num_questions:
                flag = True
                break
            question = await generate_question_async(context, keyword, num_beams)
            # options = await generate_options_async(keyword, context)
            options = gen_options(keyword, context, question)
            blank_question = await generate_fill_in_the_blank_questions(context,keyword)
            overall_score, relevance_score, complexity_score, spelling_correctness = assess_question_quality(context, question, keyword)
            if overall_score >= 0.5:
                questions.append({
                    "question": question,
                    "context": context,
                    "answer": keyword,
                    "options": options,
                    "overall_score": overall_score,
                    "relevance_score": relevance_score,
                    "complexity_score": complexity_score,
                    "spelling_correctness": spelling_correctness,
                    "blank_question": blank_question,
                })
    return questions

# Function to export questions to CSV
def export_to_csv(data):
    # df = pd.DataFrame(data, columns=["Context", "Answer", "Question", "Options"])
    df = pd.DataFrame(data)
    # csv = df.to_csv(index=False,encoding='utf-8') 
    csv = df.to_csv(index=False)
    return csv

# Function to export questions to PDF
def export_to_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for item in data:
        pdf.multi_cell(0, 10, f"Context: {item['context']}")
        pdf.multi_cell(0, 10, f"Question: {item['question']}")
        pdf.multi_cell(0, 10, f"Answer: {item['answer']}")
        pdf.multi_cell(0, 10, f"Options: {', '.join(item['options'])}")
        pdf.multi_cell(0, 10, f"Overall Score: {item['overall_score']:.2f}")
        pdf.ln(10)
    
    return pdf.output(dest='S').encode('latin-1')

def display_word_cloud(generated_questions):
    word_frequency = {}
    for question in generated_questions:
        words = question.split()
        for word in words:
            word_frequency[word] = word_frequency.get(word, 0) + 1

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequency)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()


def assess_question_quality(context, question, answer):
    # Assess relevance using cosine similarity
    context_doc = nlp(context)
    question_doc = nlp(question)
    relevance_score = context_doc.similarity(question_doc)

    # Assess complexity using token length (as a simple metric)
    complexity_score = min(len(question_doc) / 20, 1)  # Normalize to 0-1

    # Assess Spelling correctness
    misspelled = spell.unknown(question.split())
    spelling_correctness = 1 - (len(misspelled) / len(question.split()))  # Normalize to 0-1

    # Calculate overall score (you can adjust weights as needed)
    overall_score = (
        0.4 * relevance_score +
        0.4 * complexity_score +
        0.2 * spelling_correctness
    )

    return overall_score, relevance_score, complexity_score, spelling_correctness

def main():
    # Streamlit interface
    st.title(":blue[Question Generator System]")
    session_id = get_session_id()
    state = initialize_state(session_id)
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []

    with st.sidebar:
        show_info = st.toggle('Show Info',False)
        if show_info:
            display_info()
        st.subheader("Customization Options")
        # Customization options
        input_type = st.radio("Select Input Preference", ("Text Input","Upload PDF"))
        with st.expander("Choose the Additional Elements to show"):
            show_context = st.checkbox("Context",True)
            show_answer = st.checkbox("Answer",True)
            show_options = st.checkbox("Options",True)
            show_entity_link = st.checkbox("Entity Link For Wikipedia",True)
            show_qa_scores = st.checkbox("QA Score",True)
            show_blank_question = st.checkbox("Fill in the Blank Questions",True)
        num_beams = st.slider("Select number of beams for question generation", min_value=2, max_value=10, value=2)
        context_window_size = st.slider("Select context window size (number of sentences before and after)", min_value=1, max_value=5, value=1)
        num_questions = st.slider("Select number of questions to generate", min_value=1, max_value=1000, value=5)
        col1, col2 = st.columns(2)
        with col1:
            extract_all_keywords = st.toggle("Extract Max Keywords",value=False)
        with col2:
            enable_feedback_mode = st.toggle("Enable Feedback Mode",False)

    text = None
    if input_type == "Text Input":
        text = st.text_area("Enter text here:", value="Joe Biden, the current US president is on a weak wicket going in for his reelection later this November against former President Donald Trump.", help="Enter or paste your text here")
    elif input_type == "Upload PDF":
        file = st.file_uploader("Upload PDF Files")
        if file is not None:
            try:
                text = get_pdf_text(file)
            except Exception as e:
                st.error(f"Error reading PDF file: {str(e)}")
                text = None
    if text:
        text = clean_text(text)
    with st.expander("Show text"):
        st.write(text)
        # st.text(text)
    generate_questions_button = st.button("Generate Questions",help="This is the generate questions button")
    # st.markdown('<span aria-label="Generate questions button">Above is the generate questions button</span>', unsafe_allow_html=True)

    # if generate_questions_button:
    if generate_questions_button and text:
        start_time = time.time()
        with st.spinner("Generating questions..."):
            try:
                state['generated_questions'] = asyncio.run(generate_questions_async(text, num_questions, context_window_size, num_beams, extract_all_keywords))
                if not state['generated_questions']:
                    st.warning("No questions were generated. The text might be too short or lack suitable content.")
                else:
                    st.success(f"Successfully generated {len(state['generated_questions'])} questions!")
            except QuestionGenerationError as e:
                st.error(f"An error occurred during question generation: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
        data = get_state(session_id)
        print(data)
        end_time = time.time()
        print(f"Time Taken to generate: {end_time-start_time}")
        set_state(session_id, 'generated_questions', state['generated_questions'])
    
    # sort question based on their quality score
    state['generated_questions'] = sorted(state['generated_questions'],key = lambda x: x['overall_score'], reverse=True)
    # Display generated questions
    if state['generated_questions']:
        st.header("Generated Questions:",divider='blue')
        for i, q in enumerate(state['generated_questions']):
            st.subheader(body=f":orange[Q{i+1}:] {q['question']}")

            if show_blank_question is True:
                st.write(f"**Fill in the Blank Question:** {q['blank_question']}")
            if show_context is True:
                st.write(f"**Context:** {q['context']}")
            if show_answer is True:
                st.write(f"**Answer:** {q['answer']}")
            if show_options is True:
                st.write(f"**Options:**")
                for j, option in enumerate(q['options']):
                    st.write(f"{chr(65+j)}. {option}")
            if show_entity_link is True:
                linked_entity = entity_linking(q['answer'])
                if linked_entity:
                    st.write(f"**Entity Link:** {linked_entity}")
            if show_qa_scores is True:
                m1,m2,m3,m4 = st.columns([1.7,1,1,1])
                m1.metric("Overall Quality Score", value=f"{q['overall_score']:,.2f}")
                m2.metric("Relevance Score", value=f"{q['relevance_score']:,.2f}")
                m3.metric("Complexity Score", value=f"{q['complexity_score']:,.2f}")
                m4.metric("Spelling Correctness", value=f"{q['spelling_correctness']:,.2f}")

                # q['context'] = st.text_area(f"Edit Context {i+1}:", value=q['context'], key=f"context_{i}")
            if enable_feedback_mode:
                collect_feedback(
                    i,
                    question = q['question'],
                    answer = q['answer'],
                    context = q['context'],
                    options = q['options'],
                )
            st.write("---")
        
            
        # Export buttons
        # if st.session_state.generated_questions:
        if state['generated_questions']:
            with st.sidebar:   
                # Adding error handling while exporting the files 
                # --------------------------------------------------------------------- 
                try:
                    csv_data = export_to_csv(state['generated_questions'])
                    st.download_button(label="Download CSV", data=csv_data, file_name='questions.csv', mime='text/csv')
                    pdf_data = export_to_pdf(state['generated_questions'])
                    st.download_button(label="Download PDF", data=pdf_data, file_name='questions.pdf', mime='application/pdf')
                except Exception as e:
                    st.error(f"Error exporting CSV: {e}")

                # ---------------------------------------------------------------------
                # csv_data = export_to_csv(state['generated_questions'])
                # st.download_button(label="Download CSV", data=csv_data, file_name='questions.csv', mime='text/csv')

                # pdf_data = export_to_pdf(state['generated_questions'])
                # st.download_button(label="Download PDF", data=pdf_data, file_name='questions.pdf', mime='application/pdf')

            with st.expander("View Visualizations"):
                questions = [tpl['question'] for tpl in state['generated_questions']]
                overall_scores = [tpl['overall_score'] for tpl in state['generated_questions']]
                st.subheader('WordCloud of Questions',divider='rainbow')
                display_word_cloud(questions)
                st.subheader('Overall Scores',divider='violet')
                overall_scores = pd.DataFrame(overall_scores,columns=['Overall Scores'])
                st.line_chart(overall_scores)


    # View Feedback Statistics
    with st.expander("View Feedback Statistics"):
        analyze_feedback()
        if st.button("Export Feedback"):
            feedback_data = export_feedback_data()
            pswd = st.secrets['EMAIL_PASSWORD']
            send_email_with_attachment(
                email_subject='feedback from QGen',
                email_body='Please find the attached feedback JSON file.',
                recipient_emails=['apjc01unique@gmail.com', 'channingfisher7@gmail.com'],
                sender_email='apjc01unique@gmail.com',
                sender_password=pswd,
                attachment=feedback_data
            ) 

    print("********************************************************************************")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please try refreshing the page. If the problem persists, contact support.")

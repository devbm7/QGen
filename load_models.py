import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
import sense2vec
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
import wikipediaapi
from langchain_community.llms import Ollama
# import time

def load_llama():
    llm = Ollama(model='llama3:latest')
    return llm

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

def initialize_wikiapi():
    # Initialize Wikipedia API with a user agent
    user_agent = 'QGen/1.2'
    wiki_wiki = wikipediaapi.Wikipedia(user_agent= user_agent,language='en')
    return user_agent, wiki_wiki




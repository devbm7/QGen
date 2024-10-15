import streamlit as st
import uuid
from load_models import initialize_wikiapi
from functools import lru_cache

class QuestionGenerationError(Exception):
    """Custom exception for question generation errors."""
    pass

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


# Function to perform entity linking using Wikipedia API
@lru_cache(maxsize=128)
def entity_linking(keyword):
    user_agent, wiki_wiki = initialize_wikiapi()
    page = wiki_wiki.page(keyword)
    if page.exists():
        return page.fullurl
    return None


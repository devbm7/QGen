import streamlit as st
import json
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

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
    st.session_state.feedback_data.append(feedback)
    return feedback_file

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
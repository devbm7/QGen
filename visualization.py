from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

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
from nltk.tokenize import sent_tokenize

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
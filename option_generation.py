import nltk
import random
import asyncio
nltk.download('wordnet')
from nltk.corpus import wordnet
from sentence_transformers import util
from load_models import load_nlp_models, load_llama, load_qa_models
from utils import QuestionGenerationError

nlp, s2v = load_nlp_models()
llm = load_llama()
similarity_model, spell = load_qa_models()
context_model = similarity_model

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


import asyncio
import streamlit as st
from text_processing import segment_text
from keyword_extraction import extract_keywords
from utils import QuestionGenerationError
from mapping_keywords import map_keywords_to_sentences
from option_generation import gen_options
from fill_in_the_blanks_generation import generate_fill_in_the_blank_questions
from load_models import load_nlp_models, load_qa_models, load_model

nlp, s2v = load_nlp_models()
similarity_model, spell = load_qa_models()


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


async def process_batch(batch, keywords, context_window_size, num_beams, num_questions, modelname):
    questions = []
    print("inside process batch function")
    flag = False
    for text in batch:
        if flag:
            break
        keyword_sentence_mapping = map_keywords_to_sentences(text, keywords, context_window_size)
        print(keyword_sentence_mapping)
        for keyword, context in keyword_sentence_mapping.items():
            print("Length of questions list from process batch function: ",len(questions))
            if len(questions)>=num_questions:
                flag = True
                break
            question = await generate_question_async(context, keyword, num_beams,modelname)
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


async def generate_question_async(context, answer, num_beams,modelname):
    model, tokenizer = load_model(modelname)
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
    
# Function to generate questions using beam search
async def generate_questions_async(text, num_questions, context_window_size, num_beams, extract_all_keywords,modelname):
    try:
        batches = segment_text(text.lower())
        keywords = extract_keywords(text, extract_all_keywords)
        all_questions = []
        
        progress_bar = st.progress(0) 
        status_text = st.empty()
        print("Final keywords:",keywords)
        print("Number of questions that needs to be generated: ",num_questions)
        print("totoal no of batches:", batches)
        for i, batch in enumerate(batches):
            print("batch no: ", len(batches))
            status_text.text(f"Processing batch {i+1} of {len(batches)}...")
            batch_questions = await process_batch(batch, keywords, context_window_size, num_beams,num_questions,modelname)
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




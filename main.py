import streamlit as st

st.set_page_config(
    page_icon='cyclone',
    page_title="Question Generator",
    initial_sidebar_state="auto",
    menu_items={
        "About" : "Hi this our project."
    }
)


from text_processing import clean_text, get_pdf_text
from question_generation import generate_questions_async
from visualization import display_word_cloud
from data_export import export_to_csv, export_to_pdf
from feedback import collect_feedback, analyze_feedback, export_feedback_data
from utils import get_session_id, initialize_state, get_state, set_state, display_info, QuestionGenerationError, entity_linking
import asyncio
import time
import pandas as pd
from data_export import send_email_with_attachment

st.set_option('deprecation.showPyplotGlobalUse',False)


with st.sidebar:
    select_model = st.selectbox("Select Model", ("T5-large","T5-small"))
if select_model == "T5-large":
    modelname = "DevBM/t5-large-squad"
elif select_model == "T5-small":
    modelname = "AneriThakkar/flan-t5-small-finetuned"

def main():
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
            show_context = st.checkbox("Context",False)
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

    if generate_questions_button and text:
        start_time = time.time()
        with st.spinner("Generating questions..."):
            try:
                state['generated_questions'] = asyncio.run(generate_questions_async(text, num_questions, context_window_size, num_beams, extract_all_keywords,modelname))
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

import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
import base64


st.set_page_config(page_title="Chat with any PDF documents", page_icon=':shark:')

@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits


@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list

    st.info("`Generating sample questions ...`")
    n = len(text)

    # Check if the chunk size is smaller than the total size of the document
    if chunk >= n:
        st.error("Chunk size should be smaller than the total size of the document.")
        st.stop()

    # Generate random starting indices within valid range
    starting_indices = [random.randint(0, n - chunk) for _ in range(N)]
    sub_sequences = [text[i:i + chunk] for i in starting_indices]

    # Create question-answer pairs using the sub-sequences
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:", i + 1)
        except:
            st.warning('Error generating question %s.' % str(i + 1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


def main():

    st.title("Chat with any PDF documents ")

    # Add custom CSS
    st.markdown(
        """
        <style>

        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
                color:black;
            }

            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }

            .css-zt5igj {left:0;
            }

            span.css-10trblm {margin-left:0;
            }

            div.css-1kyxreq {margin-top: -40px;
            }



        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("img/logo1.png")

    st.sidebar.title("Menu")

    embedding_option = "OpenAI Embeddings"
    retriever_type = "SIMILARITY SEARCH"

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    if 'openai_api_key' not in st.session_state:
        openai_api_key = st.text_input(
            'Please enter your OpenAI API key or [get one here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)
    
    st.sidebar.header("PDFs Preview")
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            pdf_contents = uploaded_file.read()
            encoded_pdf = base64.b64encode(pdf_contents).decode("utf-8")
            st.sidebar.markdown(f'<embed src="data:application/pdf;base64,{encoded_pdf}" width="100%" height="600"></embed>', unsafe_allow_html=True)

    if uploaded_files:
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)

        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

        embeddings = OpenAIEmbeddings()

        retriever = FAISS.from_texts(splits, embeddings).as_retriever(k=5)

        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = BaseCallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        if 'eval_set' not in st.session_state:
            num_eval_questions = 5
            st.session_state.eval_set = generate_eval(
                loaded_text, num_eval_questions, 3000)

        for i, qa_pair in enumerate(st.session_state.eval_set):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.write("Ready to answer questions.")

        user_question = st.text_input("Enter your question:")
        if user_question:
            answer = qa.run(user_question)
            st.write("Answer:", answer)


if __name__ == "__main__":
    main()



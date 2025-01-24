import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
import fitz  # PyMuPDF
from PyPDF2 import PdfReader

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Change the page title and icon
st.set_page_config(
    page_title="PDF ChatBot",  # Page title
    page_icon="📄",  # Page icon
    layout="wide"  # Page layout
)

# Function to apply CSS based on language direction
def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{
                direction: {direction};
                text-align: {direction};
            }}
            .stChatInput {{
                direction: {direction};
            }}
            .stChatMessage {{
                direction: {direction};
                text-align: {direction};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Sidebar configuration
with st.sidebar:
    # Language selection dropdown
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])

    # Apply CSS direction based on selected language
    if interface_language == "العربية":
        apply_css_direction("rtl")  # Right-to-left for Arabic
        st.title("الإعدادات")  # Sidebar title in Arabic
    else:
        apply_css_direction("ltr")  # Left-to-right for English
        st.title("Settings")  # Sidebar title in English

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Define the chat prompt template with memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful assistant. Your task is to answer questions based on the provided context from the PDF. Follow these rules strictly:

            1. **Language Handling:**
               - If the question is in English, answer in English.
               - If the question is in Arabic, answer in Arabic.
               - If the user explicitly asks for a response in a specific language, respond in that language.

            2. **Contextual Answers:**
               - Provide accurate and concise answers based on the context provided.
               - Do not explicitly mention the source of information unless asked.

            3. **Handling Unclear or Unanswerable Questions:**
               - If the question is unclear or lacks sufficient context, respond with:
                 - In English: "I'm sorry, I couldn't understand your question. Could you please provide more details?"
                 - In Arabic: "عذرًا، لم أتمكن من فهم سؤالك. هل يمكنك تقديم المزيد من التفاصيل؟"
               - If the question cannot be answered based on the provided context, respond with:
                 - In English: "I'm sorry, I don't have enough information to answer that question."
                 - In Arabic: "عذرًا، لا أملك معلومات كافية للإجابة على هذا السؤال."

            4. **Professional Tone:**
               - Maintain a professional and respectful tone in all responses.
               - Avoid making assumptions or providing speculative answers.
            """),
            MessagesPlaceholder(variable_name="history"),  # Add chat history to the prompt
            ("human", "{input}"),
            ("system", "Context: {context}"),
        ])

# Main area for chat interface
st.title("PDF ChatBot")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Read PDF content
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Display PDF in the right column
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Chat")
        # Initialize session state for chat messages if not already done
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize memory if not already done
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Text input field
        if interface_language == "العربية":
            human_input = st.chat_input("اكتب سؤالك هنا...")
        else:
            human_input = st.chat_input("Type your question here...")

        # If text input is detected, process it
        if human_input:
            st.session_state.messages.append({"role": "user", "content": human_input})
            with st.chat_message("user"):
                st.markdown(human_input)

            # Create and configure the document chain and retriever
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = FAISS.from_texts([text], GoogleGenerativeAIEmbeddings()).as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Get response from the assistant
            response = retrieval_chain.invoke({
                "input": human_input,
                "context": retriever.get_relevant_documents(human_input),
                "history": st.session_state.memory.chat_memory.messages  # Include chat history
            })
            assistant_response = response["answer"]

            # Append and display assistant's response
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # Add user and assistant messages to memory
            st.session_state.memory.chat_memory.add_user_message(human_input)
            st.session_state.memory.chat_memory.add_ai_message(assistant_response)

            # Highlight text in PDF
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_instances = page.search_for(assistant_response)
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
            doc.save("highlighted.pdf")

    with col2:
        st.header("PDF Viewer")
        # Display highlighted PDF
        with fitz.open("highlighted.pdf") as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                st.image(pix.tobytes(), use_column_width=True)

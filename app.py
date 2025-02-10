import streamlit as st
import os
import base64
import time
from PIL import Image
import io
import re
import tempfile
from typing import List, Tuple
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
import uuid


GOOGLE_API_KEY = "AIzaSyCJKO3-pEFRXKnydO_vwB-laKcDpug7dYQ"  


os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

chat = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def looks_like_base64(sb):
    """Check if string looks like base64."""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    """Check if base64 data is an image."""
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    """Resize base64 encoded image."""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """Split documents into base64-encoded images and texts."""
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def get_chat_history(messages: List[dict], k: int = 3) -> str:
    """Extract the last k conversation turns from the chat history."""
    recent_messages = messages[-2*k:] if len(messages) > 2*k else messages
    formatted_history = []

    for msg in recent_messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        formatted_history.append(f"{role}: {content}")

    return "\n".join(formatted_history)

def create_augmented_query(question: str, chat_history: str) -> str:
    """Create an augmented query combining current question with chat history."""
    messages = [
        HumanMessage(content=f"""Chat History:
        {chat_history}

        Current Question: {question}

        Generate a search query that captures the context from both the history and the current question.""")
    ]

    response = chat.invoke(messages)
    return response.content

def img_prompt_func(data_dict, chat_history: str = ""):
    """Create prompt with images, text context, and chat history."""
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    text_message = {
        "type": "text",
        "text": (
            "You are financial analyst tasking with providing insights based on data provided.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide answers to the question asked by user.\n"
            f"Recent conversation history:\n{chat_history}\n\n"
            f"Current question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

@st.cache_resource
def extract_pdf_elements(uploaded_file) -> List:
    """Extract elements from uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    temp_dir = tempfile.mkdtemp()

    elements = partition_pdf(
        filename=tmp_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=temp_dir
    )

    os.unlink(tmp_path)
    return elements, temp_dir

def categorize_elements(raw_pdf_elements):
    """Categorize PDF elements into tables and texts."""
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

def summarize_elements(texts, tables, chat, progress_bar):
    """Summarize text and table elements with progress tracking."""
    text_prompt = """You are an assistant tasked with summarizing text for retrieval.
    These summaries will be embedded and used to retrieve the raw text.
    Provide a concise, retrieval-optimized summary of the following text."""

    table_prompt = """You are an assistant tasked with summarizing tables for retrieval.
    These summaries will be embedded and used to retrieve the raw table.
    Provide a concise summary of the table's contents and purpose."""

    text_summaries = []
    table_summaries = []

    total_elements = len(texts) + len(tables)
    processed = 0

    for text in texts:
        try:
            msg = chat.invoke([HumanMessage(content=f"{text_prompt}\n\nElement: {text}")])
            text_summaries.append(msg.content)
            processed += 1
            progress_bar.progress(processed / total_elements)
        except Exception as e:
            st.error(f"Error summarizing text: {e}")

    for table in tables:
        try:
            msg = chat.invoke([HumanMessage(content=f"{table_prompt}\n\nElement: {table}")])
            table_summaries.append(msg.content)
            processed += 1
            progress_bar.progress(processed / total_elements)
        except Exception as e:
            st.error(f"Error summarizing table: {e}")

    return text_summaries, table_summaries

def process_images(image_dir, chat, progress_bar):
    """Process and summarize images with progress tracking."""
    img_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    img_base64_list = []
    image_summaries = []

    prompt = """You are an assistant tasked with summarizing financial statement images for retrieval.
    These summaries will be embedded and used to retrieve the raw image.
    These summaries should capture the main theme of the financial statement, emphasizing each and every topics
    and categories without including specific numbers."""

    for i, img_file in enumerate(img_files):
        try:
            img_path = os.path.join(image_dir, img_file)
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            img_base64_list.append(base64_image)

            summary = chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ]
                    )
                ]
            ).content
            image_summaries.append(summary)
            progress_bar.progress((i + 1) / len(img_files))
        except Exception as e:
            st.error(f"Error processing image {img_file}: {e}")

    return img_base64_list, image_summaries

@st.cache_resource
def create_multi_vector_retriever(_vectorstore, _embeddings, text_summaries, texts,
                                table_summaries, tables, image_summaries, _images):
    """Create multi-vector retriever for different content types."""
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=_vectorstore,
        docstore=store,
        id_key=id_key,
        embedding=_embeddings,
        search_kwargs={"k": 10},
    )

    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, _images)

    return retriever

def main():
    st.title("PDF Chat Assistant")

    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    with st.sidebar:
        st.title("Instructions")
        st.markdown("1. Upload a PDF file")
        st.markdown("2. Wait for processing")
        st.markdown("3. Start chatting with your document")

    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])

    if uploaded_file and "processed_file" not in st.session_state:
        try:
            with st.spinner("Processing PDF..."):
                progress = st.progress(0)

                st.info("Extracting PDF elements...")
                raw_pdf_elements, image_dir = extract_pdf_elements(uploaded_file)
                progress.progress(20)

                st.info("Categorizing elements...")
                texts, tables = categorize_elements(raw_pdf_elements)
                progress.progress(40)

                st.info("Summarizing content...")
                text_summaries, table_summaries = summarize_elements(texts, tables, chat, progress)
                progress.progress(60)

                st.info("Processing images...")
                img_base64_list, image_summaries = process_images(image_dir, chat, progress)
                progress.progress(80)

                st.info("Creating retriever...")
                vectorstore = Chroma(
                    collection_name="pdf_chat",
                    embedding_function=embeddings
                )

                st.session_state.retriever = create_multi_vector_retriever(
                    _vectorstore=vectorstore,
                    _embeddings=embeddings,
                    text_summaries=text_summaries,
                    texts=texts,
                    table_summaries=table_summaries,
                    tables=tables,
                    image_summaries=image_summaries,
                    _images=img_base64_list
                )
                st.session_state.processed_file = True

                try:
                    if os.path.exists(image_dir):
                        import shutil
                        shutil.rmtree(image_dir)
                except Exception as e:
                    st.warning(f"Could not clean up temporary files: {e}")

                progress.progress(100)
                st.success("PDF processed successfully!")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.session_state.pop("processed_file", None)
            return

    if st.session_state.retriever:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your PDF"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        
                        chat_history = get_chat_history(st.session_state.messages, k=3)

                       
                        augmented_query = create_augmented_query(prompt, chat_history)

                        
                        docs = st.session_state.retriever.get_relevant_documents(augmented_query)
                        context = split_image_text_types(docs)

                        
                        messages = img_prompt_func(
                            {"context": context, "question": prompt},
                            chat_history=chat_history
                        )
                        response = chat.invoke(messages).content
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error generating response: {e}")

    with st.sidebar:
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.session_state.pop("processed_file", None)
            if "retriever" in st.session_state:
              del st.session_state.retriever
            st.rerun()


if __name__ == "__main__":
    main()
Multimodal RAG for Financial Document Analysis
Project Overview
This project is a Retrieval-Augmented Generation (RAG) system designed to analyze financial documents using multimodal capabilities. It combines the power of unstructured data extraction, advanced summarization using Gemini, and an interactive Streamlit app for querying financial data. The system efficiently extracts and processes text, tables, and images from financial statements, stores them in a ChromaDB vector database for retrieval, and provides a chatbot interface powered by Gemini for natural language queries.


Key Features
1. Multimodal RAG
Supports retrieval and summarization of text, tables, and images.
Incorporates Gemini for generating concise and informative summaries for both textual and visual data.

2. Data Processing with Unstructured
Utilizes the Unstructured library to extract text, tables, and images from financial documents (e.g., PDFs).
Handles complex document layouts efficiently.

3. ChromaDB for Vector Retrieval
Stores embeddings for text and image summaries in a ChromaDB vector database.
Enables fast and accurate retrieval for RAG-based systems.

4. Gemini-Powered Chatbot
A chatbot interface built with Gemini to answer user queries based on retrieved financial data.
Provides detailed and contextually relevant answers.

5. Streamlit App Deployment
A user-friendly Streamlit app that allows users to:
Upload financial statements (PDFs or images).
View extracted content and summaries.
Interact with a chatbot to ask questions about the uploaded financial documents.

Workflow

1.Data Extraction
Extracts text, tables, and images from financial documents using the Unstructured library.

2.Summarization
Generates summaries for extracted text and images using Gemini.

3.Vectorization & Storage
Embeds text and image summaries using Gemini Embeddings and stores them in ChromaDB for retrieval.

4.Question-Answering with Gemini
Retrieves relevant content using ChromaDB.
Generates answers for user queries through a Gemini-powered chatbot.

5.Streamlit App
Provides a simple interface for uploading documents and querying them interactively.

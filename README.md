# 🎓 Academic RAG Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Ollama](https://img.shields.io/badge/AI-Ollama_Llama3-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Academic RAG Assistant** is a state-of-the-art **Retrieval-Augmented Generation (RAG)** application designed for researchers, students, and academics. It allows users to chat with their private documents (PDFs, Lecture Notes, Research Papers) using a local LLM, ensuring data privacy while providing accurate, citation-backed answers.

---

## ✨ Key Features

* **📄 Multi-Format Ingestion:** Seamlessly upload and process **PDF**, **DOCX**, and **TXT** files.
* **🧠 Local Intelligence:** Powered by **Ollama (Llama 3)** running locally for maximum privacy and zero latency.
* **🔍 Smart Retrieval:** Uses **ChromaDB** and HuggingFace embeddings (`all-MiniLM-L6-v2`) for semantic search.
* **📚 Citation System:** Every answer includes a **Sources Badge** and expandable context citations to verify facts.
* **📊 RAG Analytics:** Integrated dashboard using **RAGAS** metrics (Faithfulness, Relevancy) to monitor answer quality.
* **🎨 Professional UI:** A clean, responsive interface built with Streamlit, featuring real-time processing status.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **LLM Server:** [Ollama](https://ollama.com/)
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **Evaluation:** RAGAS (RAG Assessment)
* **Visualization:** Plotly

---

## ⚙️ Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.9+**
2.  **Ollama**: You must have Ollama installed and the Llama 3 model pulled.
    * Download: [ollama.com](https://ollama.com)
    * **Run this command in your terminal:**
        ```bash
        ollama pull llama3
        ```

---

## 🚀 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/academic-rag-assistant.git](https://github.com/your-username/academic-rag-assistant.git)
    cd academic-rag-assistant
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Create a file named `requirements.txt` with the dependencies below, or install them directly:
    ```bash
    pip install streamlit langchain langchain-community langchain-huggingface chromadb pypdf python-docx plotly pandas ragas datasets ollama
    ```

---

## 💻 Usage Guide

1.  **Start the Application**
    Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```

2.  **Workflow**
    * **Step 1: Upload:** Go to the **📂 Knowledge Base** tab and upload your research papers.
    * **Step 2: Process:** Click **⚡ Start Processing**. The system will chunk and embed your documents.
    * **Step 3: Chat:** Switch to the **💬 Chat** tab. Ask complex questions about your documents.
    * **Step 4: Verify:** Click the **"📚 Sources Cited"** badge to see exactly where the AI got the information.

---

# How it working:
### 1. install requirements.txt ---> pip install -r requirements.txt 
### 2. Download and install Ollama from https://ollama.ai
  ### Then run:
   -ollama pull llama3
   -ollama serve
### 3. Run the application ------>   streamlit run app.py


## 📂 Project Structure

```plaintext
academic-rag-assistant/
├── app.py              # Main Streamlit application entry point
├── config.py           # Central configuration (Paths, Model names, Constants)
├── utils.py            # Core logic (Document Loading, Splitting, Vector Store)
├── evaluation.py       # RAGAS evaluation metrics and visualization logic
├── vector_store/       # (Auto-generated) Persistent ChromaDB storage
└── requirements.txt    # Python dependencies

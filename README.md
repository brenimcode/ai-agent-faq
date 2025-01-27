# AI Agent FAQ with Retrieval-Augmented Generation (RAG)

This personal project involves the development of an **AI Agent** designed to intelligently and personally respond to Frequently Asked Questions (FAQ) using a **Large Language Model (LLM)**. The chosen model, **DeepSeek** , is integrated with the **Retrieval-Augmented Generation (RAG)** technique to enhance response accuracy and relevance. This solution is applicable across a variety of industries that encounter frequent customer inquiries, enabling efficient, context-aware automation.

---

## **Features**
- **Intelligent FAQ Handling**: Understands the context and retrieves the most relevant answers from the FAQ database.
- **Built on RAG**: Combines retrieval capabilities with advanced generative AI models to enhance accuracy and contextual understanding.
- **Open Source Tools**: Uses `LangChain`, `SentenceTransformers`, and `Chroma` for flexibility and integration.
- **Customizable and Extendable**: Add your FAQ content and easily modify the system for your specific use case.

---

## **Requirements**

### **Libraries and Tools**
- Python 3.9+
- LangChain
- SentenceTransformers
- ChromaDB
- LangChain-GROQ (optional for LLM integration)
- DeepSeek

### **Installation**
Before running the notebook, ensure you install the necessary dependencies:

```bash
pip install langchain sentence-transformers chromadb langchain-groq
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/ai-agent-faq.git
cd ai-agent-faq
```

### **2. Add Your FAQ Content**
Modify the `texto` variable inside the notebook to include your FAQ content. Ensure the structure matches your use case, as shown in the sample code.

### **3. Install Dependencies**
Run the following command inside your notebook to install the required packages:
```python
!pip install -qU langchain sentence-transformers chromadb langchain-groq
```

### **4. Prepare the Knowledge Base**
The knowledge base is created using the **LangChain** framework and **ChromaDB**. It involves:
1. Converting the FAQ text into `Document` objects.
2. Splitting the content into manageable chunks using `RecursiveCharacterTextSplitter`.
3. Generating embeddings with **SentenceTransformers**.

```python
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Add your FAQ content
texto = "..."

# Prepare documents
documents = [Document(page_content=texto, metadata={"source": "https://example.com"})]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
chunks = text_splitter.split_documents(documents)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="db")
```

### **5. Configure API Keys**
If you use an external LLM (e.g., GROQ API), ensure the API key is set in your environment:
```python
import os
os.environ['GROQ_API_KEY'] = 'your-groq-api-key'
```

### **6. Run the RAG Function**
You can query the AI Agent by calling the `rag()` function:
```python
response = rag("What products does your store sell?")
print(response)
```

---

## **File Breakdown**
- **`notebook.ipynb`**: Main code file that builds and runs the AI Agent.
- **`db/`**: Directory where the ChromaDB database is stored.
- **`requirements.txt`**: List of required Python dependencies.

---

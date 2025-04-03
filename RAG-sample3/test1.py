
!pip install langchain langchain_community PyPDF sentence_transformers faiss-gpu

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import os
os.environ['HUGGINGFACEHUB_API_TOKEN']="<HF API KEY>"


## Read the pdfs from the folder
loader=PyPDFDirectoryLoader("/content/sample_data/")
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents=text_splitter.split_documents(documents)

final_documents[0]
len(final_documents) #length --> 316

## Embedding Using Huggingface
huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)
## VectorStore Creation
vectorstore=FAISS.from_documents(final_documents[:120],huggingface_embeddings)

## Query using Similarity Search
query="WHAT IS HEALTH INSURANCE COVERAGE?"
relevant_docments=vectorstore.similarity_search(query)
print(relevant_docments[0].page_content)

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
print(retriever)

from huggingface_hub import login
login(token="<HF API KEY>")

from langchain_community.llms import HuggingFaceHub
hf=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature":0.1,"max_length":500}

)
query="What is the health insurance coverage?"
hf.invoke(query)



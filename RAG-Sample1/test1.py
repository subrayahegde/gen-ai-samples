# Install dependencies
!pip install huggingface_hub
!pip install chromadb
!pip install langchain
!pip install pypdf
!pip install sentence-transformers
!pip install langchain_community

# import required libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# Load the pdf file and split it into smaller chunks
loader = PyPDFLoader('/content/sample_data/MF.pdf')
documents = loader.load()

# Split the documents into smaller chunks 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

!pip install transformers -U
from sentence_transformers import SentenceTransformer

 # We will use HuggingFace embeddings 
embeddings = HuggingFaceEmbeddings()

#Using Chroma vector database to store and retrieve embeddings of our text
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={'k': 2})

# We are using Mistral-7B for this question answering 
repo_id = "mistralai/Mistral-7B-v0.1"
llm = HuggingFaceHub(huggingfacehub_api_token='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', 
                     repo_id=repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})

# Create the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever,return_source_documents=True)

#We will run an infinite loop to ask questions to LLM and retrieve answers untill the user wants to quit
import sys
chat_history = []
while True:
    query = input('Prompt: ')
    #To exit: use 'exit', 'quit', 'q', or Ctrl-D.",
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))

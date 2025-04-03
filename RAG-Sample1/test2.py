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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load the pdf file and split it into smaller chunks
loader = PyPDFLoader('/content/sample_data/MF.pdf')
docs = loader.load()

print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())

retriever = vectorstore.as_retriever()

repo_id = "mistralai/Mistral-7B-v0.1"
llm = HuggingFaceHub(huggingfacehub_api_token='hf_<HF API KEY>', 
                     repo_id=repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "What is a Mutual Fund?"})

results

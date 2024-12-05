import tempfile
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma 
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class Rag:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self) -> None:
        # Initialize the embeddings using GPT4AllEmbeddings
        self.embeddings = GPT4AllEmbeddings()
        self.model = ChatOllama(model="mistral", temperature=0)
        
        # Define the prompt template for question-answering
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
            maximum and keep the answer concise. [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

    def set_retriever(self):
        # Set up the retriever from the vector store
        # Here we're not using it for retrieval, as we'll call `similarity_search` directly
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,  # Retrieve top 3 most similar chunks
                "score_threshold": 0.2,  # Only consider chunks with similarity above 0.2
            },
        )

    def augment(self):
        # Combine retriever, prompt, and model into a chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, query: str):
        # Ensure that the chain is set up before making a query
        if not self.vector_store:
            return "Please upload a PDF file for context."

        # Use similarity_search on the vector store directly instead of the retriever
        response = self.vector_store.similarity_search(query, k=5)  # Use similarity_search directly
        if not response:
            return "No relevant context found."
        
        # Use the context (page_content) to answer the question
        context = " ".join([doc.page_content for doc in response])  # Access 'page_content' for text
         # Format the prompt using the PromptTemplate
        prompt = self.prompt.format(question=query, context=context)

        # Call the model with the formatted prompt
        result = self.model.invoke(prompt)
        
        return result

    def feed(self, file_path: str):
        # Load and process the PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        # Split the text into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_documents(pages)

        # Store the chunks in a vector database using Chroma and GPT4AllEmbeddings
        self.vector_store = Chroma.from_documents(chunks, self.embeddings, persist_directory="./chroma_db")

        # Set up retriever and augment the context for the prompt
        self.set_retriever()
        self.augment()

    def clear(self):
        # Clear the vector store, chain, and retriever
        self.vector_store = None
        self.chain = None
        self.retriever = None

# Streamlit UI setup for file upload and query
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# Initialize the RAG system
rag_system = Rag()

# File upload for PDF
uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')

# Input for query text
query_text = st.text_input('Enter your question:', placeholder='Ask a question related to the document.')

# Handle form submission
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not (uploaded_file and query_text))
    
    if uploaded_file is not None:
        # Store the uploaded PDF and prepare the vector store
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            file_path = temp_file.name
        
        # Feed the uploaded document into the RAG system
        rag_system.feed(file_path)

    if submitted and uploaded_file and query_text:
        # Ask the question using the RAG system and show the result
        with st.spinner('Calculating...'):
            response = rag_system.ask(query_text)
            result.append(response)

# Display the response
if len(result):
    st.info(result[0])  # Displaying the result of the query

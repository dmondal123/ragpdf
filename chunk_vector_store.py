import streamlit as st
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import chroma
from langchain_community.embeddings import fastembed
import chromadb
from chromadb.config import Settings

class ChunkVectorStore:

    def __init__(self) -> None:
        configuration = {
            "client": "HttpClient",
            "host": "localhost",
            "port": 8000,
        }

        self.conn = st.connection(name="http_connection", type=ChromadbConnection, **configuration)

        # Create or fetch the Chroma collection
        self.collection_name = "documents_collection"
        self.collection = self.conn.get_collection(self.collection_name)

    def split_into_chunks(self, file_path: str):
        """
        Splits the given PDF into chunks based on the RecursiveCharacterTextSplitter.
        """
        # Load document from PDF file
        doc = PyPDFLoader(file_path).load()

        # Split the document into smaller chunks with some overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
        chunks = text_splitter.split_documents(doc)

        # Filter metadata to ensure only relevant chunks are returned
        chunks = filter_complex_metadata(chunks)
        return chunks

    def store_to_vector_database(self, chunks):
        """
        Converts document chunks to vectors and stores them in Chroma.
        """
        embedding_model = fastembed.FastEmbedEmbeddings()

        # Use Chroma's from_documents method to store chunks in the collection
        vector_store = chroma.Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            client=self.conn.client  # Using the connection's client
        )

        # Optionally, add the documents to the collection
        self.collection.add_documents(vector_store)  # Add documents to Chroma collection
        
        return vector_store  # Return the vector store object for further use (optional)

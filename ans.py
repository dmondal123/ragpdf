from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter

class Rag:
    def __init__(self, file_path: str):
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
        # Directly initialize the loader with the file_path
        self.loader = PyPDFLoader(file_path)

        # Create text splitter with larger chunk size
        self.splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

        # Setup the Chroma vector store
        self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=GPT4AllEmbeddings())

        # Load and split the document into chunks directly in initialization
        self.load_and_split_document()

    def load_and_split_document(self):
        # Load the PDF document and split into text chunks
        documents = self.loader.load()
        text_chunks = self.splitter.split_documents(documents)

        # Add the document chunks to the vector store
        self.vectorstore.add_documents(text_chunks)
        print(f"Documents in the vector store: {len(self.vectorstore)}")

        # Set up the retriever (we're using it directly in the ask method)
        self.set_retriever()

    def set_retriever(self):
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,  # Retrieve top 5 most similar chunks
                "score_threshold": 0.2,  # Lower threshold to retrieve more documents
            },
        )

    def ask(self, query: str):
        if not self.vectorstore:
            return "Please upload a PDF file for context."
        
        # Use similarity_search on the vectorstore directly
        response = self.vectorstore.similarity_search(query, k=5)  # Call similarity_search directly on vectorstore
        print(f"Found {len(response)} results.")
        return response



# Usage example
file_path = "2101.00387v2.pdf"  # Provide the correct file path
rag = Rag(file_path)
result = rag.ask("What are the key audio features?")
print(result)

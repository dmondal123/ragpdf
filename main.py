import streamlit as st
import tempfile
import os

# Import Rag and ChunkVectorStore classes
from rag import Rag
from chunk_vector_store import ChunkVectorStore  # Import the class from the file where it's defined

# Display all messages stored in session_state
def display_messages():
  for message in st.session_state.messages:
    with st.chat_message(message['role']):
      st.markdown(message['content'])

def process_file():
  st.session_state["assistant"].clear()
  st.session_state.messages = []

  for file in st.session_state["file_uploader"]:
    # Store the file at a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      tf.write(file.getbuffer())
      file_path = tf.name

    # Feed the file to the ChunkVectorStore for processing and storing it in Chroma
    with st.session_state["feeder_spinner"], st.spinner("Uploading and indexing the document..."):
      # Initialize ChunkVectorStore and process the file
      vector_store = ChunkVectorStore()

      # Split the file into chunks and store them in the vector database
      chunks = vector_store.split_into_chunks(file_path)
      vector_store.store_to_vector_database(chunks)
      
    # Clean up by removing the temporary file
    os.remove(file_path)

def process_input():
  # See if user has typed in any message and assign to prompt.
  if prompt := st.chat_input("What can I do?"):
    with st.chat_message("user"):
      st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response and write back to the chat container.
    response = st.session_state["assistant"].ask(prompt)
    with st.chat_message("assistant"):
      st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

def main():
  st.title("DocueMentor")

  # Initialize the session_state
  if len(st.session_state) == 0:
    st.session_state["assistant"] = Rag()
    st.session_state.messages = []

  # Code for file upload functionality
  st.file_uploader(
      "Upload the document",
      type=["pdf"],
      key="file_uploader",
      on_change=process_file,
      label_visibility="collapsed",
      accept_multiple_files=True,
  )

  st.session_state["feeder_spinner"] = st.empty()

  display_messages()
  process_input()

if __name__ == "__main__":
  main()

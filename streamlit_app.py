# import streamlit as st
# # !pip install faiss-cpu
# import faiss
# import numpy as np

# st.title('🤖Machine Learning App🤖')

# # st.write('')
# st.info('This is steel-plate-fault-prediction app')

# with st.expander('Data'):
#   st.write('**Raw data**')

# uploaded_file = st.file_uploader("Upload a text file", type="pdf")
# model_choice = st.selectbox("Select a Language model", ["GPT-J", "GPT-NeoX", "LLaMA", "BLOOM"])

# if uploaded_file is not None:
#   file_content = uploaded_file.read().decode("utf-8")
#   st.write(f"Uploaded file: {uploaded_file.name}")

#   text_chunks = [chunk for chunk in file_content.split("\n") if chunk.strip()]
#   st.write("File successfully processed.")
#   st.write(text_chunks)  # Display the text chunks (for debugging purposes)
  

# # text_chunks = [chunk for chunk in uploaded_file.read().split("\n") if chunk.strip()]
# vector_dim = 768

# vectors = np.random.random((len(text_chunks), vector_dim)).astype('float32')

# index = faiss.IndexFlatL2(vector_dim)
# index.add(vectors)

import streamlit as st
from PyPDF2 import PdfReader

# Title of the Streamlit app
st.title("PDF Text Extraction")

# File uploader to allow users to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Check if a file is uploaded
if uploaded_file is not None:
    try:
        # Create a PDF reader object
        reader = PdfReader(uploaded_file)

        # Initialize an empty string to hold extracted text
        pdf_text = ""

        # Loop through each page in the PDF
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            # Extract text from the current page
            pdf_text += page.extract_text()

        # Display the extracted text in a text area in the Streamlit app
        st.write("Extracted Text:")
        st.text_area("PDF Content", pdf_text, height=300)

    except Exception as e:
        # If any error occurs, display it to the user
        st.error(f"Error extracting text from the PDF: {e}")


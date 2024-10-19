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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import faiss
import numpy as np

# Title of the Streamlit app
st.title("Custom RAG Application with Open-Source LLMs")

# File uploader to allow users to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Step 1: Extract text from PDF
        reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()

        st.write("Extracted Text:")
        st.text_area("PDF Content", pdf_text, height=300)

        # Step 2: Process the text into chunks
        text_chunks = [chunk.strip() for chunk in pdf_text.split("\n") if chunk.strip()]

        # Step 3: Set up FAISS for text chunk indexing
        vector_dim = 768  # Assumes we're using a model with 768-dimensional embeddings
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with st.spinner("Generating embeddings..."):
                outputs = embedding_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()

        # Generate embeddings for all text chunks
        chunk_embeddings = np.array([get_embeddings(chunk)[0] for chunk in text_chunks])

        # Build FAISS index
        index = faiss.IndexFlatL2(vector_dim)  # L2 distance
        index.add(chunk_embeddings)

        # Step 4: User Input for Query
        user_query = st.text_input("Enter your query:")

        if user_query:
            # Step 5: Query embedding
            query_embedding = get_embeddings(user_query)

            # Step 6: Search for relevant chunks using FAISS
            D, I = index.search(query_embedding, k=5)  # Retrieve top 5 relevant chunks
            relevant_chunks = [text_chunks[i] for i in I[0]]

            st.write("Top Relevant Chunks:")
            for chunk in relevant_chunks:
                st.write(chunk)

            # Step 7: Augment user query with relevant chunks
            augmented_query = " ".join(relevant_chunks) + "\n" + user_query

            # Step 8: Integrate Open-Source LLM for text generation
            # You can use GPT-J, GPT-Neo, BLOOM, etc.
            model_choice = st.selectbox("Select a Language Model", ["EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"])
            language_model = AutoModelForCausalLM.from_pretrained(model_choice)
            tokenizer_lm = AutoTokenizer.from_pretrained(model_choice)

            def generate_response(prompt):
                inputs = tokenizer_lm(prompt, return_tensors="pt")
                outputs = language_model.generate(inputs['input_ids'], max_length=200)
                return tokenizer_lm.decode(outputs[0], skip_special_tokens=True)

            # Generate response using the augmented query
            if st.button("Generate Response"):
                response = generate_response(augmented_query)
                st.write("Generated Response:")
                st.write(response)

    except Exception as e:
        st.error(f"Error processing the PDF or generating the response: {e}")


# import streamlit as st
# from PyPDF2 import PdfReader

# # Title of the Streamlit app
# st.title("PDF Text Extraction")

# # File uploader to allow users to upload a PDF file
# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# # Check if a file is uploaded
# if uploaded_file is not None:
#     try:
#         # Create a PDF reader object
#         reader = PdfReader(uploaded_file)

#         # Initialize an empty string to hold extracted text
#         pdf_text = ""

#         # Loop through each page in the PDF
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             # Extract text from the current page
#             pdf_text += page.extract_text()

#         # Display the extracted text in a text area in the Streamlit app
#         st.write("Extracted Text:")
#         st.text_area("PDF Content", pdf_text, height=300)

#     except Exception as e:
#         # If any error occurs, display it to the user
#         st.error(f"Error extracting text from the PDF: {e}")


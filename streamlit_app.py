import streamlit as st
!pip install faiss-cpu
import faiss
import numpy as np

st.title('🤖Machine Learning App🤖')

# st.write('')
st.info('This is steel-plate-fault-prediction app')

with st.expander('Data'):
  st.write('**Raw data**')

uploaded_file = st.file_uploader("Upload a text file", type="pdf")
model_choice = st.selectbox("Select a Language model", ["GPT-J", "GPT-NeoX", "LLaMA", "BLOOM"])

if uploaded_file is not None:
  st.write(f"Uploaded file: {uploaded_file.name}")

text_chunks = [chunk for chunk in uploaded_file.read().split("\n") if chunk.strip()]
vector_dim = 768

vectors = np.random.random((len(text_chunks), vector_dim)).astype('float32')

index = faiss.IndexFlatL2(vector_dim)
index.add(vectors)

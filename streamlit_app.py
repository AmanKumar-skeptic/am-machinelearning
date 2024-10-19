import streamlit as st

st.title('🤖Machine Learning App🤖')

# st.write('')
st.info('This is steel-plate-fault-prediction app')

with st.expander('Data'):
  st.write('**Raw data**')

upload_file = st.file_uploader("Upload a text file", type="pdf")
model_choice = st.selectbox("Select a Language model", ["GPT-J", "GPT-NeoX", "LLaMA", "BLOOM"])

if upload_file is not None:
  st.write(f"Uploaded file: {uploaded_file.name}")

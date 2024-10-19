import streamlit as st

st.title('🤖Machine Learning App🤖')

# st.write('')
st.info('This is steel-plate-fault-prediction app')

with st.expander('Data'):
  st.write('**Raw data**')

upload_file = st.file_upload("Upload a text file", type="txt")
model_choice = st.selectbox("Select a Language model", ["GPT-J", "GPT-NeoX", "LLaMA", "BLOOM"])

if upload_file is not NOne:
  st.write(f"Uploaded file: {uploaded_file.name}")

import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

# only download once; Streamlit reruns the script on every interaction
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# helper to load pickles with error messages
def load_pickle(path):
    if not os.path.exists(path):
        st.error(f"Required file '{path}' not found. Did you train the model?")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load '{path}': {e}")
        return None

model = load_pickle("model.pkl")
vectorizer = load_pickle("vectorizer.pkl")

if model is None or vectorizer is None:
    st.stop()  # nothing to do until files are in place

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

st.title("📰 Fake News Detection App")

input_text = st.text_area("Enter News Article")

if st.button("Predict"):
    cleaned = clean_text(input_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        st.success("✅ This News is Real")
    else:
        st.error("❌ This News is Fake")
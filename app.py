import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        max-width: 700px;
        margin: auto;
        color: white;
    }
    .title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section-header {
        color: #2980b9;
        font-size: 1.5rem;
        margin-bottom: 10px;
    }
    .stTextArea>div>div>textarea {
        background-color: #2b2b2b;
        color: white;
    }
    .stTextArea>div>div>textarea:hover {
        background-color: white;
        color: black;
    }
    .label {
        color: #2980b9;
    }
    .footer {
        color: white;
        text-align: center;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1rem;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">📧 Email/SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="label section-header">Enter the message below to check if it is Spam or Not Spam</div>', unsafe_allow_html=True)

# User input
st.markdown('<div class="label">📝 Enter your message here:</div>', unsafe_allow_html=True)
input_sms = st.text_area("", height=200, key="input_sms")  # Adjusted height and added key for caching

if st.button('🚀 Predict'):
    with st.spinner('Analyzing...'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")

# Adding footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
---
Made with ❤️ by [Mukit](https://www.linkedin.com/in/abdulmukitds/)

Thank you for visiting, have a good day.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

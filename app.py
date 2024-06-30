import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from PIL import Image

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

# Adding a custom logo
logo = Image.open('logo.png')
st.image(logo, width=150)

st.title("📧 Email/SMS Spam Classifier")
st.write("""
### Enter the message below to check if it is Spam or Not Spam
""")

# Using markdown for better text formatting
input_sms = st.text_area("📝 Enter your message here:")

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
st.markdown("""
---
Made with ❤️ by [Abdul Mukit](https://www.linkedin.com/in/abdulmukitds/)
""")

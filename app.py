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
    /* Set the entire page background color */
    .css-18e3th9, .css-1d391kg, .css-12oz5g7, .css-1lzenr4, .css-1v3fvcr {
        background-color: #9FD3BF;  
        color: black; 
    }

    /* Target the main content area */
    .stApp {
        background-color: #9FD3BF;
    }

    .title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section-header {
        color: #00008B;
        font-size: 1.5rem;
        margin-bottom: 10px;
    }
    .stTextArea>div>div>textarea {
        background-color: #FFFFF7;
        color: black;
    }
    .stTextArea>div>div>textarea:hover {
        background-color: #FFFFFF;
        color: black;
    }
    .label {
        color: #00008B;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 0.8rem;
    }
    .footer a {
        color: black;
    }
    .stButton>button {
        background-color: #2F4F4F;
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
    .result-header {
        font-size: 2rem;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Example Texts")
example_1 = "Congratulations! You have won a $1000 Walmart gift card. Click here to claim."
example_2 = "Hi, just checking in to see how you are doing. Let me know if you need anything."
example_3 = "Your subscription to our service has been renewed successfully. Thank you for your continued support."
example_4 = "Urgent! Your account has been compromised. Please update your information immediately to avoid account suspension."
example_5 = "Reminder: Your appointment with Dr. Smith is scheduled for tomorrow at 10:00 AM. Please reply to confirm."

if st.sidebar.button('Load Example 1'):
    st.session_state.input_sms = example_1

if st.sidebar.button('Load Example 2'):
    st.session_state.input_sms = example_2

if st.sidebar.button('Load Example 3'):
    st.session_state.input_sms = example_3

if st.sidebar.button('Load Example 4'):
    st.session_state.input_sms = example_4

if st.sidebar.button('Load Example 5'):
    st.session_state.input_sms = example_5

st.markdown('<div class="title">📧 Email/SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Enter the message below to check if it is Spam or Not Spam</div>', unsafe_allow_html=True)

# User input
st.markdown('<div class="label">📝 Enter your message here:</div>', unsafe_allow_html=True)
input_sms = st.text_area("", value=st.session_state.get('input_sms', ""), height=200, key="input_sms")  # Adjusted height and added key for caching

if st.button('🚀 Predict'):
    with st.spinner('Analyzing...'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display result
        if result == 1:
            st.markdown('<div class="result-header">🚨 Spam</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-header">✅ Not Spam</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<footer>
    &copy;
    <a href="https://www.linkedin.com/in/abdul-mukit-1bbb72218" target="_blank" class='highlight'>Abdul Mukit</a>.
</footer>
""", unsafe_allow_html=True)

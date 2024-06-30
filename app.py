import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('titanic_model.pkl')

# Add custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        max-width: 700px;
        margin: auto;
    }
    .title {
        color: #2c3e50;
        font-size: 2.5rem;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section-header {
        color: #2980b9;
        font-size: 1.5rem;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the Streamlit app
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">Titanic Survival Prediction</div>', unsafe_allow_html=True)

# Display an image
st.image('https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg', use_column_width=True)

# User inputs for prediction
st.markdown('<div class="section-header">Enter Passenger Details:</div>', unsafe_allow_html=True)
Pclass = st.selectbox('Passenger Class', [1, 2, 3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.slider('Age', 0, 100, 30)
SibSp = st.slider('Siblings/Spouses Aboard', 0, 10, 0)
Parch = st.slider('Parents/Children Aboard', 0, 10, 0)
Fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=50.0)

# Convert inputs to numeric
sex = 1 if Sex == 'female' else 0

# Predict function
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare):
    input_data = np.array([[Pclass, sex, Age, SibSp, Parch, Fare]])
    prediction = model.predict(input_data)
    return prediction

# Predict button
if st.button('Predict'):
    prediction = predict_survival(Pclass, sex, Age, SibSp, Parch, Fare)
    if prediction[0] == 0:
        st.error('Unfortunately, the passenger is predicted not to survive.')
    else:
        st.success('The passenger is predicted to survive!')

st.markdown('</div>', unsafe_allow_html=True)

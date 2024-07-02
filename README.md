# Email Spam Classifier

## Introduction

This project is an Email Spam Classifier application built using Streamlit. The model leverages machine learning techniques to classify emails as spam or not spam based on their content. This tool is designed to help users filter out unwanted emails and maintain a clean inbox.

## Features

- **Real-time Classification**: Input email text and get immediate spam or not spam classification.
- **User-friendly Interface**: Simple and intuitive interface built with Streamlit.
- **Machine Learning Model**: Trained using a dataset of labeled emails to ensure accurate predictions.

## Live web app: https://emailspamclassifier-sjxca9f6jqesekujcobynn.streamlit.app/

## Dataset
The model is trained on a publicly available kaggle spam email dataset. The dataset contains labeled examples of spam and non-spam emails, which helps in training an accurate classifier.

## Model
The machine learning model used for this project is based on a Multinomial Naive Bayes classifier, which is well-suited for text classification tasks. The model has been trained and evaluated to ensure it performs well on unseen data.

## File Structure


** app.py: Main file to run the Streamlit app.
** model.py: Contains the code for training and saving the machine learning model.
** requirements.txt: Lists the dependencies required to run the project.
** spam_classifier.pkl: The pre-trained machine learning model.
** logo.png: For importing an image as logo
** model.pkl: The pre-trained machine learning model.
** vectorizer.pkl: The pre-trained vectorizer used for transforming text data.
** README.md: Project documentation

## Contributing
Contributions are welcome! If you have any ideas or suggestions, feel free to open an issue or submit a pull request.

## Contact
For any questions or suggestions, please reach out to [Kazirafi98@gmail.com]

Thank you for checking out the Email Spam Classifier project! We hope you find it useful and easy to use.


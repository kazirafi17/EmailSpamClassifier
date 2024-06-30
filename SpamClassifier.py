#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[8]:


import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score



# In[9]:


# Load the data
df = pd.read_csv('email.csv')


# In[10]:


# Print samples
df.sample(2)


# #### 1. Data cleaning

# In[11]:


# Check the shape of the dataset
df.shape


# In[12]:


# Check for the null values
df.isnull().sum()


# In[13]:


# Check for the suplicated values
df.duplicated().sum()


# In[14]:


# Drop the duplicate values
df = df.drop_duplicates(keep='first')


# In[15]:


# Label encoding for the category coulumn
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])


# In[16]:


# Remove the column which have Category value 2
df= df[df['Category'] != 2]


# In[17]:


# New shape of the df
df.shape


# #### 2.EDA

# In[18]:


# Pie plot for showing the precentages of spam and ham
import matplotlib.pyplot as plt
plt.pie(df['Category'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[19]:


nltk.download('punkt')


# In[20]:


# Create a new column for number of chacters
df['num_characters'] = df['Message'].apply(len)


# In[21]:


df.sample(1)


# In[22]:


# Create a new column for number of words
df['num_words'] = df['Message'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[23]:


df.sample(1)


# In[24]:


# Create a new column for number of sentences
df['num_sentences'] = df['Message'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[25]:


df.sample(1)


# In[26]:


df.describe()


# In[27]:


# see the correlations between the columns
temp_df = df.drop(['Message'],axis=1)
sns.heatmap(temp_df.corr(),annot=True)


# There are multicollinearity between num_characters,num_words and num_sentences. Thus I keep only num_characters 
# bewteen them as it has strongest correlation with categrory.

# In[28]:


# Drop two columns
df = df.drop(columns=['num_words', 'num_sentences'])


# In[29]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['Category']==1]['num_characters'],color='red')
sns.histplot(df[df['Category']==0]['num_characters'],color='yellow')


# #### 3.Data preprocessing
# 
#  - Lower case
#  - Tokenization
#  - Remove special characters
#  - Remove stop words and punctuations
#  - Stemming

# In[30]:


import nltk
nltk.download('stopwords')


# In[31]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

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
    
    # Create an instance of PorterStemmer
    ps = PorterStemmer()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[32]:


df['transformed_Message'] = df['Message'].apply(transform_text)


# In[33]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[34]:


spam_wc = wc.generate(df[df['Category']==1]['transformed_Message'].str.cat(sep=''))


# In[35]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[36]:


ham_wc = wc.generate(df[df['Category']==0]['transformed_Message'].str.cat(sep=''))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[37]:


spam_corpus = []
for msg in df[df['Category']==1]['transformed_Message'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[38]:


from collections import Counter

#Calculate the most common 30 words
common_words = Counter(spam_corpus).most_common(30)

# Convert the result to a DataFrame
common_words_df = pd.DataFrame(common_words, columns=['word', 'count'])

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='word', y='count', data=common_words_df)
plt.xticks(rotation=90)
plt.title('Top 30 Most Common Words in Spam Corpus')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.show()


# In[39]:


ham_corpus = []
for msg in df[df['Category']==0]['transformed_Message'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[40]:


#Calculate the most common 30 words
common_words = Counter(ham_corpus).most_common(30)

# Convert the result to a DataFrame
common_words_df = pd.DataFrame(common_words, columns=['word', 'count'])

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='word', y='count', data=common_words_df)
plt.xticks(rotation=90)
plt.title('Top 30 Most Common Words in ham Corpus')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.show()


# #### 4. Model Building

# In[41]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=1000)


# In[42]:


X = tfidf.fit_transform(df['transformed_Message']).toarray()


# In[43]:


y = df['Category'].values


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)








mnb = MultinomialNB()


# Train the classifier
mnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mnb.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test,y_pred)


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")


import pickle
pickle.dump(mnb,open('model.pkl','wb'))



pickle.dump(tfidf,open('vectorizer.pkl','wb'))




import sys

print("Python version:", sys.version)


# In[ ]:


import nbformat
from nbconvert import PythonExporter

def convert_ipynb_to_py(ipynb_file, py_file):
    # Read the notebook content
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook_content = f.read()

    # Parse the notebook content
    notebook = nbformat.reads(notebook_content, as_version=4)

    # Convert the notebook to Python script
    python_exporter = PythonExporter()
    python_script, _ = python_exporter.from_notebook_node(notebook)

    # Write the Python script to a file
    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(python_script)

# Example usage
convert_ipynb_to_py('example_notebook.ipynb', 'converted_script.py')


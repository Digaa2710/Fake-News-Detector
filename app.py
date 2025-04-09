import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import streamlit as st

nltk.download('punkt_tab')
nltk.download('stopwords')
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

news_df=pd.read_csv('train.csv', encoding='ISO-8859-1')
news_df = news_df[news_df['label'].isin([0, 1])]
news_df = news_df[['id', 'title', 'author', 'text', 'label']]
news_df=news_df.fillna(' ')     
news_df['content'] = news_df['author'] + ' ' + news_df['title'] 
print(news_df['label'].value_counts())


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
      if i.isalnum():
        y.append(i)
    text=y[:]
    y.clear()

    for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation:
        y.append(i)
    text=y[:]
    y.clear()

    for i in text:
      y.append(ps.stem(i))

    return " ".join(y)

news_df['transformed_text']=news_df['content'].apply(transform_text)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)
X=tfidf.fit_transform(news_df['transformed_text']).toarray()
y=news_df['label'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

st.title('Fake News Detection')
input_text = st.text_area("Enter the news content:")

def prediction(input_text):
    transformed_input = transform_text(input_text)
    input_data = tfidf.transform([transformed_input]).toarray()
    prediction = model.predict(input_data)
    return prediction[0]

# st.title('Fake News Detection')
# input_text = st.text_area("📰 Enter the news content:")

if st.button("Check News"):
    if input_text.strip() != "":
        result = prediction(input_text)
        if result == 1:
            st.success("🚨 The news is FAKE.")
        else:
            st.error(" ✅ The news is REAL.")
    else:
        st.warning("Please enter some news text.")

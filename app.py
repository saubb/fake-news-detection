import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# load model and vectorizer
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content

st.title("Fake News Detection System")

st.write("Enter a news headline or article to check if it is real or fake.")

news_input = st.text_area("News Text")

if st.button("Predict"):

    processed_text = stemming(news_input)

    vector_input = vectorizer.transform([processed_text])

    prediction = model.predict(vector_input)

    if prediction[0] == 0:
        st.success("This News is Real")
    else:
        st.error("This News is Fake")
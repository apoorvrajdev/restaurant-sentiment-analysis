import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Restaurant Review Sentiment Analyzer")

review = st.text_input("Enter a restaurant review")

if st.button("Predict Sentiment"):

    review_vector = vectorizer.transform([review]).toarray()

    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😠")
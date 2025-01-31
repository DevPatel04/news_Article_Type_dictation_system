import streamlit as st
import pickle

# # Define or import the ToArrayTransformer class
# class ToArrayTransformer:
#     # Define the class as it was when the model was saved
#     def transform(self, X):
#         return X.toarray()


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ToArrayTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):

        return X.toarray()

    def fit(self, X, y=None):
        return self
# Load the pre-trained model
filename = 'nlp.pkl'
try:
    with open(filename, 'rb') as f:
        pickle_load = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()
except AttributeError as e:
    st.error(f"Attribute error: {e}")
    st.stop()

# Load a pre-trained model for sentiment analysis
model = pickle_load

# Streamlit app
st.title("New Article Type Analysis App")

# Input text from user
user_input = st.text_area("Enter for Article analysis:")

# Perform sentiment analysis when the user clicks the button
if st.button("Analyze"):
    if user_input:
        result = model.predict([user_input])  # Use predict method  
        if result[0] == 0:
            ans = 'business'
        elif result[0] == 1:
            ans = 'entertainment'
        elif result[0] == 2:
            ans = 'politics'
        elif result[0] == 3:
            ans = "sport"
        elif result[0] == 4:
            ans = 'tech'
        else:
            ans = 'unknown'
        st.write("Sentiment:", ans)
    else:
        st.write("Please enter some text to analyze.")

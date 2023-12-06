import streamlit as st
from PIL import Image
st.title("Results")
st.image(Image.open('/Users/sarahbradford/Downloads/tea_please.jpg'), use_column_width=True)
st.write("By using all features in the Starbucks dataset, it can be concluded that the XGBoost model is the best model for predicting whether a drink is tea or not. The XGBoost model had an accuracy of 0.99, which is the highest accuracy out of all the models. The XGBoost model also had the highest precision, recall, and F1 score.")
st.write("Go ahead and order that tea! Now that you have have some class!")

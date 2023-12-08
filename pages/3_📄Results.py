import streamlit as st
from PIL import Image
import pandas as pd
st.sidebar.success("Page Navigation")
st.title("Results")
st.subheader("By using all nutritional values in the Starbucks dataset, it can be concluded that the XGBoost model is the best model for predicting whether a drink is tea or not. The XGBoost model had an accuracy of 0.99, which is the highest accuracy out of all the models. The XGBoost model also had the highest precision, recall, and F1 score.")
model_names = ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'KNN', 'Naive Bayes', 'XGBoost']
accuracy_scores = ['0.82', '0.78', '0.92', '0.85', '0.86', '0.60', '0.99']
model_accuracy = pd.DataFrame(list(zip(model_names, accuracy_scores)), columns=['Model', 'Accuracy Score'])
model_accuracy.style.highlight_max(color = 'lightgreen', axis = 0)
col1, col2 = st.columns(2)
with col1:
    st.image(Image.open('tea_please.jpg'), use_column_width=True)
with col2:
    st.dataframe(model_accuracy.style.highlight_max(color = 'lightgreen', axis = 0))
st.subheader("Go ahead and order that tea! Now that you have have some class!")


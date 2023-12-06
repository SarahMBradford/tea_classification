# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import altair as alt
import plotly.express as px
import streamlit as st
from PIL import Image

# Setting Page Configurations

st.set_page_config(page_title='Tea or Coffee for you today?', 
                                    page_icon=':house:', 
                                    layout='wide')
st.title('Tea or Coffee for you today?')

# upload image
tea_image = Image.open('tea_coffee.jpg')
st.image(tea_image, use_column_width=True)
st.header("Starbucks: Tea Classification")
st.sidebar.success("Page Navigation")
st.write("Welcome to the Starbucks Tea Classification App! This app will enable you to easily view the various tea drinks that Starbucks has to offer, if one day you are trying to decide whether you should order tea or coffee! The data used in this app is from the Starbucks Menu dataset on Kaggle. The data was collected in 2017 and contains nutritional information on Starbucks' food and drink menu. The dataset has 242 rows and 18 columns. Additionally, the data contains a total of 9 beverage categories. Out of the 9, roughly 78%, while the remaining 22% of the beverages are tea-based. ")
st.subheader("Why Tea?")
st.write("Tea is an excellent alternative to coffee. Tea has several health benefits, including reducing the risk of strokes and heart attacks while boosting the immune system and simultaneously reducing the risk of cancer. Tea contains less caffeine than coffee, so it is an ideal alternative for those who get overstimulated. Lastly, tea is also a great option for people who desire to reduce their sugar and fat intake, as many tea drinks are naturally sweet and do not require added sugar or creamers.")

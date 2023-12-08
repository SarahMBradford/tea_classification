import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import prettytable
from PIL import Image

starbucks_drinks = pd.read_csv("starbucks_drinks.csv")
starbucks_drinks.head()

starbucks_drinks['Caffeine (mg)'] = pd.to_numeric(starbucks_drinks['Caffeine (mg)'], errors='coerce')
starbucks_drinks['Caffeine (mg)'].fillna(starbucks_drinks['Caffeine (mg)'].mean(), inplace=True)
starbucks_drinks.info()

print(starbucks_drinks['Beverage_category'].unique())
starbucks_drinks['Beverage_category'] = starbucks_drinks['Beverage_category'].replace('Tazo® Tea Drinks', 'Tea')

starbucks_drinks = starbucks_drinks.drop(['Beverage_category'], axis=1)
starbucks_drinks = starbucks_drinks.drop(['Beverage'], axis=1)

print(starbucks_drinks['Total Fat (g)'].unique())
starbucks_drinks[ 'Total Fat (g)'] = starbucks_drinks[ 'Total Fat (g)'].str.replace('3 2', '3.2')

def float_converter(value):
    try:
        return float(str(value).replace('%', ''))
    except (ValueError, TypeError):
        return np.nan

starbucks_drinks = starbucks_drinks.applymap(float_converter)

# perform one hot encoding on the beverage category and beverage prep columns
def onehot_encode(starbucks_drinks, columns, prefixes):
    starbucks_drinks = starbucks_drinks.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(starbucks_drinks[column], prefix=prefix)
        starbucks_drinks = pd.concat([starbucks_drinks, dummies], axis=1)

    # Drop the original columns after processing all specified columns
    starbucks_drinks = starbucks_drinks.drop(columns, axis=1)
    return starbucks_drinks

# Example usage
starbucks_drinks = onehot_encode(
    starbucks_drinks,
    columns=['Beverage_prep'],
    prefixes=['Prep']
)
starbucks_drinks.head()
st.title("Methods")
st.subheader("Data Cleaning & Preprocessing")
st.write("The data was cleaned and preprocessed using the following methods:")
st.write("1. The data was imported into a pandas dataframe.")
starbucks_drinks = pd.read_csv("starbucks_drinks.csv")
starbucks_dataframe = st.checkbox("Show Starbucks Dataframe")
if starbucks_dataframe==True:
    st.write(starbucks_drinks)
    st.markdown("Table 1: Starbucks Drinks Dataframe")
st.write("2. The data was checked for missing values. Only 1 missing value was found in the 'Caffeine (mg) column, so it was replaced with the mean value of the column.")
starbucks_drinks['Caffeine (mg)'] = pd.to_numeric(starbucks_drinks['Caffeine (mg)'], errors='coerce')
starbucks_drinks['Caffeine (mg)'].fillna(starbucks_drinks['Caffeine (mg)'].mean(), inplace=True)
starbucks_drinks.info()
st.write("3. A class distribution was created to determine the percentage of tea and coffee drinks in the dataset. ")
if st.checkbox("Show Class Distribution of All Beverages") == True:
    class_dist = starbucks_drinks['Beverage_category'].value_counts(normalize=True)
    class_dist_table = prettytable.PrettyTable()
    class_dist_table.add_column("Beverage Category", class_dist.index)
    class_dist_table.add_column("Percentage", np.round(class_dist.values, 4))
    st.write(class_dist_table)
    st.write("Based on the nutrituional information, the majority of the beverages are coffee-based, while the remaining 21% are tea-based.")
st.write("4. The Tazo Tea Drinks were renamed to 'Tea'.")
st.write("5. All coffee-based and smoothie drinks were converted to 0 and all tea-based drinks were converted to 1.")
st.write("6. A second class distribution was created to determine the percentage of tea and coffee drinks in the dataset.")
if st.checkbox("Show Second Class Distribution: Tea or Not?") == True:
    starbucks_drinks['Beverage_category'] = starbucks_drinks['Beverage_category'].replace('Tazo® Tea Drinks', 'Tea')
    starbucks_drinks['Tea'] = starbucks_drinks['Beverage_category'].apply(lambda x: 1 if x == 'Tea' else 0)
    class_dist = starbucks_drinks['Tea'].value_counts(normalize=True)
    class_dist_table = prettytable.PrettyTable()
    class_dist_table.add_column("Tea?", class_dist.index)
    class_dist_table.add_column("Percentage", np.round(class_dist.values, 2))
    st.write(class_dist_table)
st.write("7. One hot encoding was done to split the 'Beverage_prep column into their own columns.")
st.write("8. The 'Beverage_category' column was dropped.")
st.write("9. The 'Total Fat (g)' had a value that was inconsistent as '3 2', so it was replaced with '3.2'.")
st.write("10. Train-test split was done to split the data into training and testing sets. A scaler was created and assigned to X. The scaler was then fit to X_train and transformed to X_test.")
st.subheader("Now that the data has been cleaned and preprocessed, the data is ready to be used for modeling. Click on the 'Classification Models' tab to view the models!")
meme_image = Image.open('meme.jpg')
st.image(meme_image, use_column_width=True)
st.write("Not sure if there would be tea, but it's worth a shot to find out!")

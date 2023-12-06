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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


starbucks_drinks = pd.read_csv("/Users/sarahbradford/Downloads/starbucks_drinks.csv")
starbucks_drinks.head()

starbucks_drinks['Caffeine (mg)'] = pd.to_numeric(starbucks_drinks['Caffeine (mg)'], errors='coerce')
starbucks_drinks['Caffeine (mg)'].fillna(starbucks_drinks['Caffeine (mg)'].mean(), inplace=True)
starbucks_drinks.info()

class_dist = starbucks_drinks['Beverage_category'].value_counts(normalize=True)
class_dist_table = prettytable.PrettyTable()
class_dist_table.add_column("Beverage Category", class_dist.index)
class_dist_table.add_column("Percentage", np.round(class_dist.values, 4))
print(class_dist_table)

print(starbucks_drinks['Beverage_category'].unique())
starbucks_drinks['Beverage_category'] = starbucks_drinks['Beverage_category'].replace('Tazo® Tea Drinks', 'Tea')
# class distribution for tea
starbucks_drinks['Tea'] = starbucks_drinks['Beverage_category'].apply(lambda x: 1 if x == 'Tea' else 0)
class_dist = starbucks_drinks['Tea'].value_counts(normalize=True)

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

# scale the data
X = starbucks_drinks.drop(['Tea'], axis=1)
y = starbucks_drinks['Tea']
starbucks_scaler = StandardScaler()
X = starbucks_scaler.fit_transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Classification Models")
choices = ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'KNN', 'Naive Bayes', 'XGBoost']
selected_choice = st.selectbox("Select a Classification Model", choices)
st.write(selected_choice)
if selected_choice == 'Logistic Regression':
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
# confusion matrix
    lr_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(lr_confusion_matrix), annot=True, cmap="YlGn", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Tea vs. Coffee: Logistic Regression Confusion Matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot()
# classification report
    print(metrics.classification_report(y_test, y_pred))
    st.write("Figure 1: Logistic Regression Classification Report")
    st.write("The Logistic Regression model has an accuracy of 0.82. 56 teas were predicted correctly, 13 drinks were incorrectly predicted as teas, 4 drinks were incorrectly as coffees, and 0 drinks were incorrectly rejected")
    # put the classification report in a pretty table
    lr_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    lr_classification_report_df = pd.DataFrame(lr_classification_report).transpose()
    lr_classification_report_df = lr_classification_report_df.round(2)
    lr_classification_report_df = lr_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
    st.write(lr_classification_report_df)


#SVM
if selected_choice == 'SVM':
    # SVM 
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))
# confusion matrix
    svm_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    svm_confusion_matrix
# plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(svm_confusion_matrix), annot=True, cmap="YlGn", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Tea vs. Coffee: SVM Confusion Matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot()
    # classification report
    print(metrics.classification_report(y_test, y_pred))
    st.write("Figure 2: SVM Classification Report")
    st.write("The SVM model has an accuracy of 0.78. 56 teas were predicted correctly, 16 drinks were incorrectly predicted as teas, 1 drink was incorrectly as coffees, and 0 drinks were incorrectly rejected")
    # put the classification report in a pretty table
    svm_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    svm_classification_report_df = pd.DataFrame(svm_classification_report).transpose()
    svm_classification_report_df = svm_classification_report_df.round(2)
    svm_classification_report_df = svm_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
    st.write(svm_classification_report_df)
# Decision Tree
if selected_choice == 'Decision Tree':
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtree.score(X_test, y_test)))
    # confusion matrix
    dtree_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    dtree_confusion_matrix
    # plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(dtree_confusion_matrix), annot=True, cmap="YlGn", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Tea vs. Coffee: Decision Tree Confusion Matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot()
    # classification report
    print(metrics.classification_report(y_test, y_pred))
    st.write("Figure 3: Decision Tree Classification Report")
    st.write("The Decision Tree model has an accuracy of 0.95. 55 teas were predicted correctly, 3 drinks were incorrectly predicted as teas, 14 drinks were incorrectly as coffees, and 1 drink was incorrectly rejected")
    # put the classification report in a pretty table
    dtree_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    dtree_classification_report_df = pd.DataFrame(dtree_classification_report).transpose()
    dtree_classification_report_df = dtree_classification_report_df.round(2)
    dtree_classification_report_df = dtree_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
    st.write(dtree_classification_report_df)
# Random Forest
if selected_choice == 'Random Forest':
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))
    # confusion matrix
    rf_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    rf_confusion_matrix
    # plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(rf_confusion_matrix), annot=True, cmap="YlGn", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Tea vs. Coffee: Random Forest Confusion Matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot()
    # classification report
    print(metrics.classification_report(y_test, y_pred))
    st.write("Figure 4: Random Forest Classification Report")
    st.write("The Random Forest model has an accuracy of 0.89. 55 teas were predicted correctly, 4 drinks were incorrectly predicted as teas, 13 drinks were incorrectly as coffees, and 1 drink was incorrectly rejected")
    # put the classification report in a pretty table
    rf_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    rf_classification_report_df = pd.DataFrame(rf_classification_report).transpose()
    rf_classification_report_df = rf_classification_report_df.round(2)
    rf_classification_report_df = rf_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
    st.write(rf_classification_report_df)
# KNN
if selected_choice == "KNN":
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
    # confusion matrix
    knn_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    knn_confusion_matrix
    # plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(knn_confusion_matrix), annot=True, cmap="YlGn", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Tea vs. Coffee: KNN Confusion Matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot()
    # classification report
    print(metrics.classification_report(y_test, y_pred))
    st.write("Figure 5: KNN Classification Report")
    st.write("The KNN model has an accuracy of 0.86. 50 teas were predicted correctly, 4 drinks were incorrectly predicted as teas, 13 drinks were incorrectly as coffees, and 6 drinks were incorrectly rejected")
    # put the classification report in a pretty table
    knn_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    knn_classification_report_df = pd.DataFrame(knn_classification_report).transpose()
    knn_classification_report_df = knn_classification_report_df.round(2)
    knn_classification_report_df = knn_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
    st.write(knn_classification_report_df)
# Naive Bayes
if selected_choice == "Naive Bayes":
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(nb.score(X_test, y_test)))
    # confusion matrix
    nb_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    nb_confusion_matrix
    # plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(nb_confusion_matrix), annot=True, cmap="YlGn", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Tea vs. Coffee: Naive Bayes Confusion Matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot()
    # classification report
    print(metrics.classification_report(y_test, y_pred))
    st.write("Figure 6: Naive Bayes Classification Report")
    st.write("The Naive Bayes model has an accuracy of 0.60. Only 30 teas were predicted correctly, 4 drinks were incorrectly predicted as teas, 13 drinks were incorrectly as coffees, and 25 drinks were incorrectly rejected. This model is not recommended for this dataset.")
    # put the classification report in a pretty table
    nb_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    nb_classification_report_df = pd.DataFrame(nb_classification_report).transpose()
    nb_classification_report_df = nb_classification_report_df.round(2)
    nb_classification_report_df = nb_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
    st.write(nb_classification_report_df)
# XGBoost
if selected_choice == "XGBoost":
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print('Accuracy of XGBoost classifier on test set: {:.2f}'.format(xgb.score(X_test, y_test)))
    # confusion matrix
    xgb_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    xgb_confusion_matrix
    # plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(xgb_confusion_matrix), annot=True, cmap="YlGn", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Tea vs. Coffee: XGBoost Confusion Matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot()
    # classification report
    print(metrics.classification_report(y_test, y_pred))
    st.write("Figure 7: XGBoost Classification Report")
    st.write("The XGBoost model has an accuracy of 0.99. 56 teas were predicted correctly, only 1 drink was incorrectly predicted as a tea, 16 drinks were incorrectly as coffees, and 0 drinks were incorrectly rejected. This is the best model for classifying tea vs. coffee.")
    # put the classification report in a pretty table
    xgb_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    xgb_classification_report_df = pd.DataFrame(xgb_classification_report).transpose()
    xgb_classification_report_df = xgb_classification_report_df.round(2)
    xgb_classification_report_df = xgb_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
    st.write(xgb_classification_report_df)

    


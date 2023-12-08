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
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve
st.sidebar.success("Page Navigation")


starbucks_drinks = pd.read_csv("starbucks_drinks.csv")
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
    starbucks_drinks = starbucks_drinks.drop(columns, axis=1)
    return starbucks_drinks
starbucks_drinks = onehot_encode(
    starbucks_drinks,
    columns=['Beverage_prep'],
    prefixes=['Prep']
)
starbucks_drinks.head()

# Scaling
X = starbucks_drinks.drop(['Tea'], axis=1)
y = starbucks_drinks['Tea']
starbucks_scaler = StandardScaler()
X = starbucks_scaler.fit_transform(X)

# Split the data set
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
    lr_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# Confusion Matrix Plot
    class_names = ['Coffee', 'Tea']
    fig = px.imshow(lr_confusion_matrix,
                    labels=dict(x="Predicted", y="Actual", Count="greens"),
                    x=class_names, y=class_names)

    # Add annotations for the number of observations in each square
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            Count = 'white' if lr_confusion_matrix[i, j] > lr_confusion_matrix.max() / 2 else 'black'
            fig.add_annotation(
                go.layout.Annotation(
                    text=str(lr_confusion_matrix[i, j]),
                    x=class_names[j],
                    y=class_names[i],
                    showarrow=False,
                    font=dict(color=Count)
                )
            )
    # Plotly Layout
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis=dict(colorscale='greens'))
    fig.update_layout(title_text="Confusion Matrix for Logistic Regression", margin=dict(b=10, t=140))
    st.plotly_chart(fig)

# classification report
    print(metrics.classification_report(y_test, y_pred))
    st.write("Figure 1: Logistic Regression Classification Confusion Matrix")
    st.write("The Logistic Regression model has an accuracy of 0.82. 56 coffees were predicted correctly, 13 tea drinks were incorrectly predicted as coffees, 4 tea drinks were correctly predicted as teas, and 0 drinks were incorrectly rejected.")
    if st.checkbox("View Classification Report for Logistic Regression Model") == True:
        lr_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        lr_classification_report_df = pd.DataFrame(lr_classification_report).transpose()
        lr_classification_report_df = lr_classification_report_df.round(2)
        lr_classification_report_df = lr_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
        st.write(lr_classification_report_df)
# plot precision-recall curve
    precision, recall, _ = metrics.precision_recall_curve(y_test, logreg.decision_function(X_test))
    pr_df = pd.DataFrame({'Precision': precision, 'Recall': recall})
    st.altair_chart(
        alt.Chart(pr_df).mark_line(color='green').encode(
            x='Recall:Q',
            y='Precision:Q'
        ).properties(
            title='Precision-Recall Curve for Logistic Regression'
        ).interactive()
    )
    st.subheader("View the nutritional labels and predicted drinks below:")
    starbucks_drinks['Predicted Drink'] = logreg.predict(X)
    starbucks_drinks['Predicted Drink'] = starbucks_drinks['Predicted Drink'].apply(lambda x: 'Tea' if x == 1 else 'Coffee')
    st.write(starbucks_drinks)


#SVM
if selected_choice == 'SVM':
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
# confusion matrix
    svm_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
 # Plot SVM Model
    class_names = ['Coffee', 'Tea']
    fig = px.imshow(svm_confusion_matrix,
                    labels=dict(x="Predicted", y="Actual", Count="greens"),
                    x=class_names, y=class_names)

    # Add annotations for the number of observations in each square
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            Count = 'white' if svm_confusion_matrix[i, j] > svm_confusion_matrix.max() / 2 else 'black'
            fig.add_annotation(
                go.layout.Annotation(
                    text=str(svm_confusion_matrix[i, j]),
                    x=class_names[j],
                    y=class_names[i],
                    showarrow=False,
                    font=dict(color=Count)
                )
            )
    # Plotly Layout
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis=dict(colorscale='greens'))
    fig.update_layout(title_text="Confusion Matrix for Support Vector Machine", margin=dict(b=10, t=140))
    st.plotly_chart(fig)
    

# classification report
    st.write("Figure 2: SVM Classification Confusion Matrix")
    st.write("The Logistic Regression model has an accuracy of 0.78. 56 coffees were predicted correctly, 16 tea drinks were incorrectly predicted as coffees, 1 tea drinks was correctly predicted as a tea, and 0 drinks were incorrectly rejected.")
    if st.checkbox("View Classification Report for SVM Model") == True:
        svm_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        svm_classification_report_df = pd.DataFrame(svm_classification_report).transpose()
        svm_classification_report_df = svm_classification_report_df.round(2)
        svm_classification_report_df = svm_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
        st.write(svm_classification_report_df)
    
    precision, recall, _ = metrics.precision_recall_curve(y_test, svm.decision_function(X_test))
    pr_df = pd.DataFrame({'Precision': precision, 'Recall': recall})
    st.altair_chart(
        alt.Chart(pr_df).mark_line(color='green').encode(
            x='Recall:Q',
            y='Precision:Q'
        ).properties(
            title='Precision-Recall Curve for SVM'
        ).interactive()
    )
    
    st.subheader("View the nutritional labels and predicted drinks below:")
    starbucks_drinks['Predicted Drink'] = svm.predict(X)
    starbucks_drinks['Predicted Drink'] = starbucks_drinks['Predicted Drink'].apply(lambda x: 'Tea' if x == 1 else 'Coffee')
    st.write(starbucks_drinks)

# Decision Tree
if selected_choice == 'Decision Tree':
    dtree=DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtree.score(X_test, y_test)))
    # confusion matrix
    dtree_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    class_names = ['Coffee', 'Tea']
    fig = px.imshow(dtree_confusion_matrix,
                    labels=dict(x="Predicted", y="Actual", Count="greens"),
                    x=class_names, y=class_names)

    probs = dtree.predict_proba(X_test)
    # Add annotations for the number of observations in each square
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            Count = 'white' if dtree_confusion_matrix[i, j] > dtree_confusion_matrix.max() / 2 else 'black'
            fig.add_annotation(
                go.layout.Annotation(
                    text=str(dtree_confusion_matrix[i, j]),
                    x=class_names[j],
                    y=class_names[i],
                    showarrow=False,
                    font=dict(color=Count)
                )
            )
    # Plotly Layout
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis=dict(colorscale='greens'))
    fig.update_layout(title_text="Confusion Matrix for Decision Tree", margin=dict(b=10, t=140))
    st.plotly_chart(fig)

# classification report
    st.write("Figure 3: Decision Tree Classification Confusion Matrix")
    st.write("The Decision Tree model has an accuracy of 0.92. 55 coffees were predicted correctly, 3 tea drinks were incorrectly predicted as coffees, 14 tea drinks was correctly predicted as a tea, and 1 drink was incorrectly rejected.")
    if st.checkbox("View Classification Report for Decision Tree Model") == True:
        dtree_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        dtree_classification_report_df = pd.DataFrame(dtree_classification_report).transpose()
        dtree_classification_report_df = dtree_classification_report_df.round(2)
        dtree_classification_report_df = dtree_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
        st.write(dtree_classification_report_df)
    precision, recall, _ = metrics.precision_recall_curve(y_test, probs[:, 1])
    pr_df = pd.DataFrame({'Precision': precision, 'Recall': recall})

    alt_chart = alt.Chart(pr_df).mark_line(color='green').encode(
    x='Recall:Q',
    y='Precision:Q'
        ).properties(
    title='Precision-Recall Curve for Decision Tree'
    ).interactive()

    st.altair_chart(alt_chart)
    st.subheader("View the nutritional labels and predicted drinks below:")
    starbucks_drinks['Predicted Drink'] = dtree.predict(X)
    starbucks_drinks['Predicted Drink'] = starbucks_drinks['Predicted Drink'].apply(lambda x: 'Tea' if x == 1 else 'Coffee')
    st.write(starbucks_drinks)

# Random Forest
if selected_choice == 'Random Forest':
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))
# confusion matrix
    rf_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig = px.imshow(rf_confusion_matrix,
                    labels=dict(x="Predicted", y="Actual", Count="greens"),
                    x=class_names, y=class_names)

    probs = rf.predict_proba(X_test)
    # Add annotations for the number of observations in each square
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            Count = 'white' if rf_confusion_matrix[i, j] > rf_confusion_matrix.max() / 2 else 'black'
            fig.add_annotation(
                go.layout.Annotation(
                    text=str(rf_confusion_matrix[i, j]),
                    x=class_names[j],
                    y=class_names[i],
                    showarrow=False,
                    font=dict(color=Count)
                )
            )
    # Plotly Layout
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis=dict(colorscale='greens'))
    fig.update_layout(title_text="Confusion Matrix for Random Forest", margin=dict(b=10, t=140))
    st.plotly_chart(fig)

# classification report
    st.write("Figure 4: Random Forest Classification Confusion Matrix")
    st.write("The Random Forest model has an accuracy of 0.85. 55 coffees were predicted correctly, 7 tea drinks were incorrectly predicted as coffees, 10 tea drinks was correctly predicted as a tea, and 1 drink was incorrectly rejected.")
    if st.checkbox("View Classification Report for Random Forest Model") == True:
        rf_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        rf_classification_report_df = pd.DataFrame(rf_classification_report).transpose()
        rf_classification_report_df = rf_classification_report_df.round(2)
        rf_classification_report_df = rf_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
        st.write(rf_classification_report_df)
    precision, recall, _ = metrics.precision_recall_curve(y_test, probs[:, 1])
    pr_df = pd.DataFrame({'Precision': precision, 'Recall': recall})

    alt_chart = alt.Chart(pr_df).mark_line(color='green').encode(
    x='Recall:Q',
    y='Precision:Q'
        ).properties(
    title='Precision-Recall Curve for Random Forest'
    ).interactive()

    st.altair_chart(alt_chart)
    st.subheader("View the nutritional labels and predicted drinks below:")
    starbucks_drinks['Predicted Drink'] = rf.predict(X)
    starbucks_drinks['Predicted Drink'] = starbucks_drinks['Predicted Drink'].apply(lambda x: 'Tea' if x == 1 else 'Coffee')
    st.write(starbucks_drinks)
    
# KNN
if selected_choice == 'KNN':
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
# confusion matrix
    knn_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig = px.imshow(knn_confusion_matrix,
                    labels=dict(x="Predicted", y="Actual", Count="greens"),
                    x=class_names, y=class_names)

    probs = knn.predict_proba(X_test)
    # Add annotations for the number of observations in each square
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            Count = 'white' if knn_confusion_matrix[i, j] > knn_confusion_matrix.max() / 2 else 'black'
            fig.add_annotation(
                go.layout.Annotation(
                    text=str(knn_confusion_matrix[i, j]),
                    x=class_names[j],
                    y=class_names[i],
                    showarrow=False,
                    font=dict(color=Count)
                )
            )
    # Plotly Layout
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis=dict(colorscale='greens'))
    fig.update_layout(title_text="Confusion Matrix for KNN", margin=dict(b=10, t=140))
    st.plotly_chart(fig)

# classification report
    st.write("Figure 5: KNN Classification Confusion Matrix")
    st.write("The KNN model has an accuracy of 0.86. 50 coffees were predicted correctly, 4 tea drinks were incorrectly predicted as coffees, 13 tea drinks was correctly predicted as a tea, and 6 drinks were incorrectly rejected.")
    if st.checkbox("View Classification Report for KNN Model") == True:
        rf_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        rf_classification_report_df = pd.DataFrame(rf_classification_report).transpose()
        rf_classification_report_df = rf_classification_report_df.round(2)
        rf_classification_report_df = rf_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
        st.write(rf_classification_report_df)
    precision, recall, _ = metrics.precision_recall_curve(y_test, probs[:, 1])
    pr_df = pd.DataFrame({'Precision': precision, 'Recall': recall})

    alt_chart = alt.Chart(pr_df).mark_line(color='green').encode(
    x='Recall:Q',
    y='Precision:Q'
        ).properties(
    title='Precision-Recall Curve for KNN'
    ).interactive()

    st.altair_chart(alt_chart)
    st.subheader("View the nutritional labels and predicted drinks below:")
    starbucks_drinks['Predicted Drink'] = knn.predict(X)
    starbucks_drinks['Predicted Drink'] = starbucks_drinks['Predicted Drink'].apply(lambda x: 'Tea' if x == 1 else 'Coffee')
    st.write(starbucks_drinks)
# Naive Bayes
if selected_choice == 'Naive Bayes':
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
# confusion matrix
    nb_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig = px.imshow(nb_confusion_matrix,
                    labels=dict(x="Predicted", y="Actual", Count="greens"),
                    x=class_names, y=class_names)

    probs = nb.predict_proba(X_test)
    # Add annotations for the number of observations in each square
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            Count = 'white' if nb_confusion_matrix[i, j] > nb_confusion_matrix.max() / 2 else 'black'
            fig.add_annotation(
                go.layout.Annotation(
                    text=str(nb_confusion_matrix[i, j]),
                    x=class_names[j],
                    y=class_names[i],
                    showarrow=False,
                    font=dict(color=Count)
                )
            )
    # Plotly Layout
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis=dict(colorscale='greens'))
    fig.update_layout(title_text="Confusion Matrix for Naive Bayes", margin=dict(b=10, t=140))
    st.plotly_chart(fig)

# classification report
    st.write("Figure 6: Naive Bayes Classification Confusion Matrix")
    st.write("The Naive Bayes model has an accuracy of 0.60. 31 coffees were predicted correctly, 4 tea drinks were incorrectly predicted as coffees, 13 tea drinks was correctly predicted as a tea, and 25 drinks were incorrectly rejected.")
    if st.checkbox("View Classification Report for Naive Bayes Model") == True:
        nb_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        nb_classification_report_df = pd.DataFrame(nb_classification_report).transpose()
        nb_classification_report_df = nb_classification_report_df.round(2)
        nb_classification_report_df = nb_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
        st.write(nb_classification_report_df)
    precision, recall, _ = metrics.precision_recall_curve(y_test, probs[:, 1])
    pr_df = pd.DataFrame({'Precision': precision, 'Recall': recall})

    alt_chart = alt.Chart(pr_df).mark_line(color='green').encode(
    x='Recall:Q',
    y='Precision:Q'
        ).properties(
    title='Precision-Recall Curve for Naive Bayes'
    ).interactive()

    st.altair_chart(alt_chart)
    st.subheader("View the nutritional labels and predicted drinks below:")
    starbucks_drinks['Predicted Drink'] = nb.predict(X)
    starbucks_drinks['Predicted Drink'] = starbucks_drinks['Predicted Drink'].apply(lambda x: 'Tea' if x == 1 else 'Coffee')
    st.write(starbucks_drinks)
# XGBoost
if selected_choice == 'XGBoost':
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
# confusion matrix
    xgb_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# plot confusion matrix
    class_names = ['Coffee', 'Tea']
    fig = px.imshow(xgb_confusion_matrix,
                    labels=dict(x="Predicted", y="Actual", Count="greens"),
                    x=class_names, y=class_names)

    probs = xgb.predict_proba(X_test)
    # Add annotations for the number of observations in each square
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            Count = 'white' if xgb_confusion_matrix[i, j] > xgb_confusion_matrix.max() / 2 else 'black'
            fig.add_annotation(
                go.layout.Annotation(
                    text=str(xgb_confusion_matrix[i, j]),
                    x=class_names[j],
                    y=class_names[i],
                    showarrow=False,
                    font=dict(color=Count)
                )
            )
    # Plotly Layout
    fig.update_xaxes(side="top")
    fig.update_layout(coloraxis=dict(colorscale='greens'))
    fig.update_layout(title_text="Confusion Matrix for XGBoost", margin=dict(b=10, t=140))
    st.plotly_chart(fig)

# classification report
    st.write("Figure 7: XGBoost Classification Confusion Matrix")
    st.write("The XGBoost model has an accuracy of 0.99. 56 coffees were predicted correctly, 1 tea drink was incorrectly predicted as coffee, 16 tea drinks was correctly predicted as a tea, and 0 drinks were incorrectly rejected.")
    if st.checkbox("View Classification Report for XGBoost Model") == True:
        xgb_classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        xgb_classification_report_df = pd.DataFrame(xgb_classification_report).transpose()
        xgb_classification_report_df = xgb_classification_report_df.round(2)
        xgb_classification_report_df = xgb_classification_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
        st.write(xgb_classification_report_df)
    precision, recall, _ = metrics.precision_recall_curve(y_test, probs[:, 1])
    pr_df = pd.DataFrame({'Precision': precision, 'Recall': recall})

    alt_chart = alt.Chart(pr_df).mark_line(color='green').encode(
    x='Recall:Q',
    y='Precision:Q'
        ).properties(
    title='Precision-Recall Curve for XGBoost'
    ).interactive()

    st.altair_chart(alt_chart)   
    st.subheader("View the nutritional labels and predicted drinks below:")
    starbucks_drinks['Predicted Drink'] = xgb.predict(X)
    starbucks_drinks['Predicted Drink'] = starbucks_drinks['Predicted Drink'].apply(lambda x: 'Tea' if x == 1 else 'Coffee')
    st.write(starbucks_drinks)


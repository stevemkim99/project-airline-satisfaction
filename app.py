import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Supressing display warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page title and icon
st.set_page_config(page_title = "Airline Satisfaction Dataset Explorer", page_icon = ":airplane:")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "EDA", "Modeling", "Make Predictions!", "Extras"])

df = pd.read_csv('data/train.csv')

# Build a homepage
if page == "Home":
    st.title(":airplane: Airline Satisfaction Dataset Explorer App")
    st.subheader("Welcome to our Airline Satisfaction Dataset explorer app!")
    st.write("This app is designed to make the exploration and analysis of the Airline Satisfaction dataset easy and accessible to all!")
    st.image("https://www.retently.com/wp-content/uploads/2018/08/Airline-satisfaction-cover-1.png")
    st.write("Use the sidebar to navigate between different sections!")


# Build Data Overview page
    
if page == "Data Overview":
    st.title(":1234: Data Overview")
    st.subheader("About the Data")
    st.write("This is one of the datasets that grabbed reports on customer satisfaction based on their recent travels.")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxeUZ7tAhmxhlOldviL8PDfSQZfyZHYHppUQ&usqp=CAU")
    st.link_button("Click here to learn more", "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data", help = "Airline Satisfaction Dataset Kaggle Page")

    st.subheader("Quick Glance at the Data")

    # Display dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df)

    # Column list
    if st.checkbox("Column List"):
        st.code(f"Columns: {df.columns.tolist()}")

        if st.toggle('Further breakdown of columns'):
            num_cols = df.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns{obj_cols}")

    if st.checkbox("Shape"):
        # st.write(f"The shape is {df.shape}")
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")
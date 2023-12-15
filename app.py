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
        st.write(f"There are {df.shape[0]} rows (Customers) and {df.shape[1]} columns (Reports on Flight).")


# Build EDA page
if page == "EDA":
    st.title(":bar_chart: EDA")
    num_cols = df.select_dtypes(include = 'number').columns.tolist()
    obj_cols = df.select_dtypes(include = 'object').columns.tolist()

    eda_type = st.multiselect("What type of EDA are you interested in exploring?",
                              ['Histograms', 'Box Plots', 'Scatterplots', 'Countplots'])
    
    # HISTOGRAMS
    if "Histograms" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for your histogram:", num_cols, index = None)

        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue on Histogram"):
                st.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title, color = 'satisfaction', barmode = 'overlay'))
            else: 
                st.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title))


    # BOXPLOTS
    if "Box Plots" in eda_type:
        st.subheader("Boxplots Visualizing Numerical Distribtutions")
        b_selected_col = st.selectbox("Select a numerical column for your box plot:", num_cols, index = None)
       
        if b_selected_col:           
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue on Box Plot"):
                st.plotly_chart(px.box(df, x = b_selected_col, y = 'satisfaction', title = chart_title, color = 'satisfaction'))
            else:
                st.plotly_chart(px.box(df, x = b_selected_col, title = chart_title))
                


    # SCATTERPLOTS
    if "Scatterplots" in eda_type:
        st.subheader("Visualizing Relationships")

        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index = None)

        

        if selected_col_x and selected_col_y:
            chart_title = f"Relationship of {' '.join(selected_col_x.split('_')).title()} vs. {' '.join(selected_col_y.split('_')).title()}"

            if st.toggle("Satisfaction Hue on Scatterplot"):
                st.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title, color = 'satisfaction'))
            else: 
                st.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title))



# Building our Modeling page

if page == "Modeling":
    st.title(":gear: Modeling")
    st.markdown("On this page, you can see how well different **machine learning models** make predictions on airline satisfaction!")

    # Set up X and y
    features = ['Class', 'Departure Delay in Minutes', 'Ease of Online booking', 'Flight Distance', 'Seat comfort', 'Inflight entertainment', 'Cleanliness']
    X = df[features]
    y = df['satisfaction']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # Model selection
    model_option = st.selectbox("Select a Model", ['KNN', 'Logistic Regression', 'Random Forest'], index = None)

    if model_option:
        # st.write(f"You selected {model_option}")

        if model_option == 'KNN':
            k_value = st.slider("Select the number of neighers (k)", 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors = k_value)
        elif model_option == 'Logistic Regression':
            model = LogisticRegression()
        elif model_option == 'Random Forest':
            model = RandomForestClassifier()

        
        if st.button("Let's see the performance!"):
            model.fit(X_train, y_train)

            # Display Results
            st.subheader(f"{model} Evaluation")
            st.text(f"Training Accuracy: {round(model.score(X_train, y_train)*100, 2)}%")
            st.text(f"Testing Accuracy: {round(model.score(X_test, y_test)*100, 2)}%")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = 'Blues')
            st.pyplot()


# Predictions Page
if page == "Make Predictions!":
    st.title(":rocket: Make Predictions on Airline Satisfaction Dataset")

    # Create sliders for user to input data
    st.subheader("Adjust the sliders to input data:")

    d_d = st.slider("Departure Delay in Minutes", 0.0, 1600.0, 0.0, 1.0)
    e_b = st.slider("Ease of Online booking", 0.0, 5.0, 0.0, 1.0)
    f_d = st.slider("Flight Distance", 0.0, 3500.0, 0.0, 5.0)
    s_c = st.slider("Seat comfort", 0.0, 5.0, 0.0, 1.0)
    i_e = st.slider("Inflight entertainment", 0.0, 5.0, 0.0, 1.0)
    c_c = st.slider("Cleanliness", 0.0, 5.0, 0.0, 1.0)

    # Your features must be in order that the model was trained on
    user_input = pd.DataFrame({
            'Departure Delay in Minutes': [d_d],
            'Ease of Online booking': [e_b],
            'Flight Distance': [f_d],
            'Seat comfort': [s_c],
            'Inflight entertainment': [i_e],
            'Cleanliness': [c_c]
            })

    # Check out "pickling" to learn how we can "save" a model
    # and avoid the need to refit again!
    features = ['Departure Delay in Minutes', 'Ease of Online booking', 'Flight Distance', 'Seat comfort', 'Inflight entertainment', 'Cleanliness']
    X = df[features]
    y = df['satisfaction']

    # Model Selection
    model_option = st.selectbox("Select a Model", ["KNN", "Logistic Regression", "Random Forest"], index = None)

    if model_option:

        # Instantiating & fitting selected model
        if model_option == "KNN":
            k_value = st.slider("Select the number of neighbors (k)", 1, 21, 5, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        
        if st.button("Make a Prediction!"):
            model.fit(X, y)
            prediction = model.predict(user_input)
            st.write(f"{model} predicts you will be {prediction[0]}!")
            st.balloons()
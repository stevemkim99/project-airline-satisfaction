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
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "EDA", "Modeling", "Make Predictions!"])

df_train = pd.read_csv('data/cleaned_train.csv')
df_test = pd.read_csv('data/cleaned_test.csv')

df_train.drop(columns=['Unnamed: 0'], inplace = True)

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
        st.dataframe(df_train)

    # Column list
    if st.checkbox("Column List"):
        st.code(f"Columns: {df_train.columns.tolist()}")

        if st.toggle('Further breakdown of columns'):
            num_cols = df_train.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df_train.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns{obj_cols}")

    if st.checkbox("Shape"):
        # st.write(f"The shape is {df.shape}")
        st.write(f"There are {df_train.shape[0]} rows (Customers) and {df_train.shape[1]} columns (Flight Information).")


# Build EDA page
if page == "EDA":
    st.title(":bar_chart: EDA")
    num_cols = df_train.select_dtypes(include = 'number').columns.tolist()
    obj_cols = df_train.select_dtypes(include = 'object').columns.tolist()

    eda_type = st.multiselect("What type of EDA are you interested in exploring?",
                              ['Histograms', 'Box Plots', 'Scatterplots', 'Countplots'])
    
    # HISTOGRAMS
    if "Histograms" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for your histogram:", num_cols, index = None)

        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue on Histogram"):
                st.plotly_chart(px.histogram(df_train, x = h_selected_col, title = chart_title, color = 'satisfaction', barmode = 'overlay'))
            else: 
                st.plotly_chart(px.histogram(df_train, x = h_selected_col, title = chart_title))


    # BOXPLOTS
    if "Box Plots" in eda_type:
        st.subheader("Boxplots Visualizing Numerical Distribtutions")
        b_selected_col = st.selectbox("Select a numerical column for your box plot:", num_cols, index = None)
       
        if b_selected_col:           
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue on Box Plot"):
                st.plotly_chart(px.box(df_train, x = b_selected_col, y = 'satisfaction', title = chart_title, color = 'satisfaction'))
            else:
                st.plotly_chart(px.box(df_train, x = b_selected_col, title = chart_title))
                


    # SCATTERPLOTS
    if "Scatterplots" in eda_type:
        st.subheader("Visualizing Relationships")

        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index = None)

        

        if selected_col_x and selected_col_y:
            chart_title = f"Relationship of {' '.join(selected_col_x.split('_')).title()} vs. {' '.join(selected_col_y.split('_')).title()}"

            if st.toggle("Satisfaction Hue on Scatterplot"):
                st.plotly_chart(px.scatter(df_train, x = selected_col_x, y = selected_col_y, title = chart_title, color = 'satisfaction'))
            else: 
                st.plotly_chart(px.scatter(df_train, x = selected_col_x, y = selected_col_y, title = chart_title))



# Building our Modeling page

if page == "Modeling":
    st.title(":gear: Modeling")
    st.markdown("On this page, you can see how well different **machine learning models** make predictions on airline satisfaction!")

    # Set up X and y
    features = ['class', 'departure_delay_in_minutes', 'ease_of_online_booking', 'flight_distance', 'seat_comfort', 'inflight_entertainment', 'cleanliness']
    X = df_train[features]
    y = df_train['satisfaction']

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

    c_a = st.slider("class", 0.0, 2.0, 0.0, 1.0)
    d_d = st.slider("departure_delay_in_minutes", 0.0, 1600.0, 0.0, 1.0)
    e_b = st.slider("ease_of_online_booking", 0.0, 5.0, 0.0, 1.0)
    f_d = st.slider("flight_distance", 0.0, 3500.0, 0.0, 5.0)
    s_c = st.slider("seat_comfort", 0.0, 5.0, 0.0, 1.0)
    i_e = st.slider("inflight_entertainment", 0.0, 5.0, 0.0, 1.0)
    c_c = st.slider("cleanliness", 0.0, 5.0, 0.0, 1.0)

    # Your features must be in order that the model was trained on
    user_input = pd.DataFrame({
            'class': [c_a],
            'departure_delay_in_minutes': [d_d],
            'ease_of_online_booking': [e_b],
            'flight_distance': [f_d],
            'seat_comfort': [s_c],
            'inflight_entertainment': [i_e],
            'cleanliness': [c_c]
            })

    # Check out "pickling" to learn how we can "save" a model
    # and avoid the need to refit again!
    features = ['class', 'departure_delay_in_minutes', 'ease_of_online_booking', 'flight_distance', 'seat_comfort', 'inflight_entertainment', 'cleanliness']
    X = df_train[features]
    y = df_train['satisfaction']

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
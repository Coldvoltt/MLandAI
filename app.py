import streamlit as st
import pandas as pd
from train_and_evaluate import train_and_evaluate_model
from ai_chat import ask_openai
from preprocess import prep_data

# Title
st.title("Machine Learning Model Training and Evaluation")
st.caption(
    "The app is designed for users to carry out Machine Learning using a csv data and also ask :blue[ChatGPT] about the result. \nMake sure the csv data is cleaned using excel before uploading for analysis.", )

# Step 1: Upload a file
uploaded_file = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

# Initialize variables globally
if 'results' not in st.session_state:
    st.session_state.results = None
if 'importance' not in st.session_state:
    st.session_state.importance = None
if 'r_squared' not in st.session_state:
    st.session_state.r_squared = None

if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Display data
        st.write("Data Preview:")
        st.write(data.head())

        # Step 2: Select independent variable
        independent_var = st.selectbox(
            "Select the independent variable", data.columns)

        # Step 1.1: Preprocess data
        data = prep_data(data, independent_var)

        # Step 3: Select model to use
        model_choice = st.selectbox(
            "Select the model to use",
            [
                "Linear Regression",
                "Random Forest",
                "Support Vector Machine",
                "Decision Tree",
                "K-Nearest Neighbors",
                "Gradient Boosting"
            ]
        )

        # Initialize variables as None

        # Step 4: Train and evaluate the model
        if st.button("Train and Evaluate Model"):
            with st.spinner("Training the model..."):
                # Perform model training and evaluation
                st.session_state.results, st.session_state.importance, st.session_state.r_squared = train_and_evaluate_model(
                    data, independent_var, model_choice)

            st.success("Model training and evaluation completed!")

            # Display results
            st.subheader("Model Results")
            st.write(st.session_state.results)

            # Display R Squared if available
            if st.session_state.r_squared is not None:
                st.subheader("R Squared")
                st.write(st.session_state.r_squared)

            # Display feature importance if available
            if st.session_state.importance is not None:
                st.subheader("Feature Importance")
                st.write(st.session_state.importance)

    except ValueError as ve:
        st.error(f"Value Error: {ve}")
    except Exception as e:
        st.error(f"Error: {e}")

# Step 5: Ask AI about the results
st.subheader("Ask AI")
query = st.text_input("Enter your query for the AI")


def ask_openai_with_ml_results(text: str):
    global results, importance, r_squared
    sys_msg = """
    You are an assistant specifically designed to respond to questions
    about results obtained from machine learning algorithms and be prepared
    to interpret the results in an academic writing standard.

    If results are not given, you can answer questions regularly. However,
    please ensure that users upload a cleaned CSV file for accurate interpretation.
    """
    if st.session_state.results is not None:
        sys_msg += f"\n\nResults:\n{st.session_state.results}"
    if st.session_state.importance is not None:
        sys_msg += f"\n\nFeature Importance:\n{st.session_state.importance}"
    if st.session_state.r_squared is not None:
        sys_msg += f"\n\nR-squared:\n{st.session_state.r_squared}"

    # Call ask_openai with the combined system message and user query
    try:
        return ask_openai(sys_msg + "\n\n" + text)
    except Exception as e:
        return f"AI Error: {e}"


if st.button("Ask AI"):
    with st.spinner("Asking AI..."):
        ai_response = ask_openai_with_ml_results(query)

    st.subheader("AI Response")
    st.write(ai_response)

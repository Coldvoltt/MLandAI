import streamlit as st
import pandas as pd
from train_and_evaluate import train_and_evaluate_model
from ai_chat import ask_openai
from preprocess import remove_id_columns

# Title
st.title("Machine Learning Model Training and Evaluation")
st.caption(
    "The app is designed for users to carry out Machine Learning using a csv data and also ask :blue[ChatGPT] about the result.", )

# Step 1: Upload a file
uploaded_file = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

results, importance = None, None  # Initialize globally

if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Step 1.1: Preprocess data
        data = remove_id_columns(data)

        # Display data
        st.write("Data Preview:")
        st.write(data.head())

        # Step 2: Select independent variable
        independent_var = st.selectbox(
            "Select the independent variable", data.columns)

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

        results, importance = train_and_evaluate_model(data,
                                                       independent_var, model_choice)

        # Step 4: Train and evaluate the model
        if st.button("Train and Evaluate Model"):
            with st.spinner("Training the model..."):
                # Perform model training and evaluation
                results, importance = results, importance

            st.success("Model training and evaluation completed!")

            # Display results
            st.subheader("Model Results")
            st.write(results)

            # Display feature importance if available
            if importance is not None:
                st.subheader("Feature Importance")
                st.write(importance)

    except Exception as e:
        st.error(f"Error: {e}")

# Step 5: Ask AI about the results
st.subheader("Ask AI")
query = st.text_input("Enter your query for the AI")


def ask_openai_with_ml_results(text: str):
    global results, importance

    sys_msg = """
    You are an assistant specifically designed to respond to questions
    about results obtained from machine learning algorithms and be prepared
    to interpret the results in an academic writing standard.

    If results are not given, you can answer questions regularly. However,
    please ensure that users upload a cleaned CSV file for accurate interpretation.
    """
    if results is not None:
        sys_msg += f"\n\nResults:\n{results}"
    if importance is not None:
        sys_msg += f"\n\nFeature Importance:\n{importance}"

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

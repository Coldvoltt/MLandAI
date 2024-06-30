from ai_chat import ask_openai
from train_and_evaluate import train_and_evaluate_model

# Train and evaluate the model, assuming it sets global variables results and importance
results, importance = train_and_evaluate_model("GradeClass", "Random Forest")

# Function to ask OpenAI with system message including results and importance


def ask_openai_with_ml_results(text: str):
    # Access global variables for results and importance
    global results
    global importance

    # Prepare system message with results and importance if available
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
    return ask_openai(sys_msg + "\n\n" + text)


# Example usage
res = ask_openai_with_ml_results("Interprete this result for me")
print(res)

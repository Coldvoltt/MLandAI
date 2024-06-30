from langchain.chains import ConversationChain
import os
# import json
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from groq import Groq
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# OpenAI API key
openai.api_key = os.getenv(OPENAI_API_KEY)

# System message template
sys_message = """
You are an assistant specifically designed to respond to questions
about results obtained from machine learning algorithms and be prepared
to interpret the results in an academic writing standard.

If results are not given, you can answer questions regularly. However,
please ensure that users upload a cleaned CSV file for accurate interpretation.
"""


def ask_openai(text: str):
    # Access global variables for results and importance if they exist
    global results
    global importance

    # Prepare system message with results and importance if available
    sys_msg = sys_message
    if 'results' in globals() and results is not None:
        sys_msg += f"\n\nResults:\n{results}"
    if 'importance' in globals() and importance is not None:
        sys_msg += f"\n\nFeature Importance:\n{importance}"

    # Initialize OpenAI chat model
    chat = ChatOpenAI(model_name="gpt-4o")

    # Format instructions for AI response
    format_instructions = """
    The output format should always be in a markdown format such that
    there will be appropriate heading sections.
    """

    # Template for AI prompt
    template = 'query: {query}.\nformat instructions: {format_instructions}'

    prompt = PromptTemplate(
        input_variables=['query'],
        template=template,
        partial_variables={'format_instructions': format_instructions}
    )

    # Conversation chain setup with system and human messages
    chain = chat([
        SystemMessage(content=sys_msg),
        HumanMessage(content=prompt.format(query=text))
    ])

    # Get AI response
    result = chain.content

    return result

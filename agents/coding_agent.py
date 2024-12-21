from .llm_agent import GrokLLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API_KEY environment variable
api_key = os.getenv("api_key")


def coding_agent(data_summary, problem_description, features):

    """Given a data summary and a problem description, suggest innovative features
    """

    llm = GrokLLM(api_key = api_key)
    
    prompt = f"""You are a Machine Learning Engineer who will help coding the set of the features suggested. Steps that you need to follow.
    1. Understand the problem statement from the user. This is about building a ML model
    2. Understand the data summary. It contains the features in the sample data that user has submitted. 
        You need to consider this data summary
    3. Understand the suggested features and help code the ML training solution from start to end. Code should include library imports, reading data, preprocessing, cleaning and adding suggested features in Python.
    Problem Description : {problem_description}
    Data Summary : {data_summary}
    Feature Suggestion : {features}.
    Make sure to provide only the code as output and nothing else"""
    
    response = llm._call(prompt).replace("```json", "").replace("```","")
    return {'data' : response}
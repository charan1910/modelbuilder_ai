import pandas as pd
from .llm_agent import GrokLLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API_KEY environment variable
api_key = os.getenv("api_key")

def data_agent(problem_description, df):
    """Given a csv parse, understand and make a clean version for the downstream task
    """
    # df = pd.read_csv(file)
    column_summary = [
        {"column_name": col, "data_type": str(df[col].dtype)}
        for col in df.columns
    ]
    # data_issues = {"missing_values": df.columns[df.isnull().any()].tolist()}
    data = {"data": column_summary}

    llm = GrokLLM(api_key = api_key)
    prompt = """You are a data analyst. Your job is to understand a problem statement from the user and 
                infer the meaning of the columns in the sample data. Problem statement : """ + problem_description +""". User has uploaded a sample data set. I will give you the sample data set in this format. 
                {column_name : 'XXX', data_type : 'YYYY'}, you need to understand the problem statement and infer the meaning of the column
                Your output should strictly follow this template
                {column_name : 'XXX', data_type : 'YYYY', meaning : 'ZZZZ'}
                Sample data : """+ str(data) + """Remember, you should strictly follow the template and output only json object"""
    response = llm._call(prompt).replace("```json", "").replace("```","")
    return {'data' : response}

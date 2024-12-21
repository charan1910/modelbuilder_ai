from .llm_agent import GrokLLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API_KEY environment variable
api_key = os.getenv("api_key")


def feature_engineering_agent(data_summary, problem_description):

    """Given a data summary and a problem description, suggest innovative features
    """

    llm = GrokLLM(api_key = api_key)

    prompt = """You are a data scientist and your job is to suggest a bunch of features for feature engineering for a Machine Learning task.
    You need to follow these 3 steps
    1. Understand the problem statement well, think critically on how to best solve the problem. 
    2. Understand the data columns available for use and identify what can be new features we can add to improve the model
    3. Combine problem statement and data to suggest a bunch of innovative features. Be creative. Also, consider interaction b/w features
    End of it, you should provide me a json string stricly in the below format 
    "{features : 
        {"feature1" : Meaning of feature1,
        "feature2" : Meaning of feature2,
        etc.
        }".
    Remember, you can be as creative as possible
    Problem Statement : """ + problem_description + """ Data Columns available : """ + str(data_summary) + """ Remember, you should strictly follow the template and output only json object"""

    response = llm._call(prompt).replace("```json", "").replace("```","")
    return {'data' : response}



    
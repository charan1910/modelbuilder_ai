import streamlit as st
import pandas as pd
from agents.data_agent import data_agent
from agents.feature_engineering import feature_engineering_agent
from agents.coding_agent import coding_agent
import logging

# Set up logging configuration
logging.basicConfig(filename='feedback_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')


# Set the page configuration to use the wide layout
st.set_page_config(page_title="AI Chatbot for ML Models", layout="wide")

# Title of the app
st.title("AI Agent for Building Machine Learning Models")

# Initialize session state for the steps
if "current_step" not in st.session_state:
    st.session_state.current_step = "upload_file"  # Start with the file upload step

# Step 1: File Upload
if st.session_state.current_step == "upload_file":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        st.write("Preview of the uploaded file:")
        st.session_state.df = pd.read_csv(uploaded_file)  # Save the DataFrame in session state
        st.session_state.df.to_csv(f"data/{uploaded_file.name}", index = False) # Save the dataframe to data folder
        st.dataframe(st.session_state.df.head())
        st.session_state.current_step = "describe_problem"  # Move to the next step

# Step 2: Problem Description
if st.session_state.current_step == "describe_problem":
    problem_description = st.text_area("Describe the problem you are trying to solve")
    if problem_description:
        st.session_state.problem_description = problem_description  # Save description
        if st.button("Run Data Agent"):
            st.session_state.data_summary = data_agent(st.session_state.problem_description, st.session_state.df)['data']
            st.session_state.current_step = "run_feature_engineering"  # Move to the next step

# Step 3: Run Data Agent
if st.session_state.current_step == "run_feature_engineering":
    st.write("Understand data:")
    st.json(st.session_state.get("data_summary", {}))
    if st.button("Run Feature Engineering Agent"):
        st.session_state.features = feature_engineering_agent(st.session_state.data_summary, st.session_state.problem_description)['data']
        st.session_state.current_step = "run_coding_agent"  # Move to the next step

# Step 4: Run Feature Engineering Agent
if st.session_state.current_step == "run_coding_agent":
    st.write("Feature Suggestions:")
    st.json(st.session_state.get("features", {}))
    if st.button("Run Coding Agent"):
        st.session_state.coding_agent_output = coding_agent(st.session_state.data_summary, st.session_state.problem_description, st.session_state.features)['data']
        st.session_state.current_step = "done"  # Final step

# Step 5: Run Coding Agent
if st.session_state.current_step == "done":
    st.write("Coding Agent Output:")
    st.code(st.session_state.get("coding_agent_output", "No output generated"))

    # Feedback Form
    st.write("### We'd love to hear your feedback!")
    feedback = st.text_area("Please share your feedback on the entire process:")

    if feedback:
        # Log the feedback to the file
        logging.info(f"User Feedback: {feedback}")    
        st.write("Thank you for your feedback! We'll use it to improve the app.")

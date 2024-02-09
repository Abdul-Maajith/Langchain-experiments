import os

import streamlit as st
import pandas as pd

from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.globals import set_verbose, get_verbose

set_verbose(True)  # set verbose to True
verbose = get_verbose()  

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY') 

# LLM Model
llm = OpenAI(temperature = 0)

steps_eda = llm("What steps are involved in EDA?")

st.title("AI Assistant for DataScience")
st.write("Hello, I'm your AI Assistant and I'm here to help you in data science eelated problems")

with st.sidebar:
    st.write("**Your Data Science Adventure Begins with a CSV Upload**")
    st.caption('''You may already know that every exciting data science journey starts with a dataset. That's why I'd love for you to upload a CSV file. Once we have your data in hand, we'll dive into understanding it and have some fun exploring it. Then, we'll work together to shape your business challenge into a data science framework. I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?''')

    with st.expander("What steps are involved in EDA?"):
        st.caption(steps_eda)

    st.divider()

    st.caption('<p style="text-align:center">Made with ❤️ by Abdul Maajith</p>', unsafe_allow_html=True)

# Initialise a session state variable
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    st.title("Exploratory Data Analysis(EDA):")
    st.subheader("Solution:")

    data_csv = st.file_uploader("Upload your file data", type="csv ")

    if data_csv is not None:
        data_csv.seek(0)
        df = pd.read_csv(data_csv, low_memory=False)

        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        st.write("**Data Overview**")
        st.write("The first rows of your dataset look like this:")
        st.write(df.head())
        st.write("**Data Cleaning**")
        columns_df = pandas_agent.run("What are the meaning of the columns?")
        st.write(columns_df)
        missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
        st.write(missing_values)
        duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
        st.write(duplicates)
        st.write("**Data Summarisation**")
        st.write(df.describe())
        correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
        st.write(correlation_analysis)
        outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
        st.write(outliers)
        new_features = pandas_agent.run("What new features would be interesting to create?.")
        st.write(new_features)


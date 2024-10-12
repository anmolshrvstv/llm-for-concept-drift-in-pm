!pip install pm4py
!pip install transformers
!pip install joblib

import pm4py
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
from transformers import pipeline
from joblib import Parallel, delayed

from google.colab import drive
drive.mount('/content/drive')
from pm4py.objects.log.importer.xes import importer as xes_importer


# Load the XES file
xes_file_path = "/content/drive/My Drive/cp-5000.xes"
log = xes_importer.apply(xes_file_path)
# Convert the event log to a DataFrame
df = pm4py.convert_to_dataframe(log)
# Print column names to identify the correct ones
print("DataFrame columns:", df.columns)
print(df) #check the dataframe for further processing

import pandas as pd
# Function to create process descriptions
def create_process_descriptions(df, case_id_col, timestamp_col, activity_col):
    process_descriptions = []
    for case_id, case_df in df.groupby(case_id_col):
        case_df = case_df.sort_values(by=timestamp_col)
        description = f"Process {case_id}:\n"
        for _, row in case_df.iterrows():
            # Get the activity name
            activity = row[activity_col] if pd.notna(row[activity_col]) else 'Unknown activity'
            # Format the description as desired
            description += f"In Case {case_id}, Activity \"{activity}\" was performed at {row[timestamp_col]}.\n"
        process_descriptions.append(description)
    return process_descriptions

# Define the column names based on your DataFrame
case_id_col = 'case:concept:name'  # Use case ID column
timestamp_col = 'time:timestamp'
activity_col = 'concept:name'  # This should point to the activity name
# Generate process descriptions
process_descriptions = create_process_descriptions(df, case_id_col, timestamp_col, activity_col)
# Print the first process description
print(process_descriptions[0])

###Approach 1 : Using BUilt-In pm4py function
# Define your OpenAI API key and model
api_key = "api=key"
openai_model = "gpt-3.5-turbo"  # or "gpt-4"
# Assuming process_descriptions is a list of descriptions generated earlier
# Join the process descriptions into a single string
process_descriptions_str = "\n".join(process_descriptions)
# Construct the prompt for concept drift detection
prompt = f"""
You are an expert in process mining and concept drift detection.
I have the following event logs:

{process_descriptions_str}

Can you help me detect any concept drift present in these logs? Please provide the type of drift and specific points in time where the drift occurs.
"""
# Call the openai_query function without specifying api_url
response = pm4py.llm.openai_query(prompt, api_key, openai_model)
# Print the response
print(response)

Approach 2 : Using Huggingface to leverage Bart!
# Load the BART model for text generation
model_name = 'facebook/bart-large-cnn'
bart_model = pipeline('text-generation', model=model_name)
# Create a prompt from the process descriptions
prompt = "Analyze the following process descriptions and Identify the type of concept drift in these process events and give the point where this was detected:\n"
+ "\n".join(process_descriptions)
# Generate a response from BART
response = bart_model(prompt, max_length=320, num_return_sequences=1)
# Output the generated response
print(response[0]['generated_text'])

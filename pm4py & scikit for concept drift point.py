### This is the mathematical approach with an algorithm developed to detect point of concept drift and verify the obtained results

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the event log data from an XES file
log = xes_importer.apply("/content/drive/My Drive/cp-5000.xes")

# Convert the event log to a DataFrame
dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

# Sort the DataFrame by timestamp
dataframe = dataframe.sort_values(by='time:timestamp')

# Define the window size and step size
window_size = 1000  # Adjust based on your data
step_size = 500     # Adjust based on your data

# Create overlapping segments
segments = []
for start in range(0, len(dataframe) - window_size + 1, step_size):
    segments.append(dataframe.iloc[start:start + window_size])

# Function to convert a DataFrame segment to DFG
def dataframe_to_dfg(df_segment):
    log_segment = log_converter.apply(df_segment)
    dfg = dfg_discovery.apply(log_segment)
    return dfg

# Convert segments to DFGs
dfgs = [dataframe_to_dfg(segment) for segment in segments]

# Convert DFGs to textual representations
def dfg_to_text(dfg):
    description = []
    for edge in dfg.keys():
        description.append(f"{edge[0]} -> {edge[1]} ({dfg[edge]})")
    return " ".join(description)

dfg_texts = [dfg_to_text(dfg) for dfg in dfgs]

# Vectorize the DFG texts using TF-IDF
vectorizer = TfidfVectorizer()
dfg_vectors = vectorizer.fit_transform(dfg_texts)

# Compute similarity scores between consecutive segments
similarity_scores = []
for i in range(1, len(dfg_vectors.shape)):
    score = cosine_similarity(dfg_vectors[i-1:i], dfg_vectors[i:i+1])[0][0]
    similarity_scores.append(score)

# Define a threshold to detect concept drift
threshold = 0.7
# Identify points of concept drift
drift_points = []
for i, score in enumerate(similarity_scores):
    if score < threshold:
        drift_points.append(i * step_size + window_size)

# Output the drift points
for point in drift_points:
    print(f"Concept drift detected around event index: {point}")

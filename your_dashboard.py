import streamlit as st
import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt


# To Run the code, Paste the line below in the terminal:-
# streamlit run your_dashboard.py


# Title and description
st.title("PsySense Dashboard")
st.write("Real-time monitoring of student emotions and psychological assessment over time.")

# Connect to the SQLite database
conn = sqlite3.connect("emotion_data.db")
query = "SELECT * FROM aggregated_emotions"
df = pd.read_sql(query, conn)

# Convert the aggregated_confidences column from JSON to a dictionary
def json_to_dict(json_str):
    try:
        return json.loads(json_str)
    except:
        return {}

df["aggregated_confidences_dict"] = df["aggregated_confidences"].apply(json_to_dict)

# Display the data in a table
st.subheader("Aggregated Emotion Data")
st.dataframe(df)

# Plotting example: show the trend of a specific emotion (e.g., 'sad') over time for a selected student.
student_ids = df["student_id"].unique().tolist()
selected_student = st.selectbox("Select Student ID", student_ids)

# Filter data for the selected student
student_df = df[df["student_id"] == selected_student].copy()

# Convert timestamp to datetime
student_df["timestamp"] = pd.to_datetime(student_df["timestamp"])

# Extract a specific emotion's values, e.g., 'sad'
student_df["sad"] = student_df["aggregated_confidences"].apply(lambda x: json.loads(x).get("sad", 0))

st.subheader(f"Trend of 'Sad' Emotion for {selected_student}")
fig, ax = plt.subplots()
ax.plot(student_df["timestamp"], student_df["sad"], marker='o')
ax.set_xlabel("Time")
ax.set_ylabel("Average 'Sad' Percentage")
ax.set_title("Emotion Trend Over Time")
st.pyplot(fig)

# Optionally, display psychological assessment info if stored in the DB.
# For instance, you could have another table with psychological risk scores.
st.write("Add additional widgets and charts as needed for your psychological assessments.")

conn.close()

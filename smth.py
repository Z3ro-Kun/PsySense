from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from typing import List
import json

app = FastAPI()

# Allow access from frontend on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE = "emotion_data.db"

def fetch_all_students():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT student_id FROM aggregated_emotions")
        return [row[0] for row in cursor.fetchall()]

def fetch_aggregated_data():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, timestamp, aggregated_confidences FROM aggregated_emotions")
        return cursor.fetchall()

def fetch_by_student(student_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, aggregated_confidences FROM aggregated_emotions WHERE student_id = ?", (student_id,))
        return cursor.fetchall()

@app.get("/students")
def get_all_students():
    return fetch_all_students()

@app.get("/data")
def get_all_data():
    rows = fetch_aggregated_data()
    return [
        {
            "student_id": r[0],
            "timestamp": r[1],
            "emotions": json.loads(r[2])
        } for r in rows
    ]

@app.get("/students/{student_id}")
def get_student_data(student_id: str):
    rows = fetch_by_student(student_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Student not found")
    return [
        {
            "timestamp": r[0],
            "emotions": json.loads(r[1])
        } for r in rows
    ]

@app.get("/students/{student_id}/latest")
def get_latest_student_data(student_id: str):
    rows = fetch_by_student(student_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Student not found")
    latest = rows[-1]
    return {
        "timestamp": latest[0],
        "emotions": json.loads(latest[1])
    }

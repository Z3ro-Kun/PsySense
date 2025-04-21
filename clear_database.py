import sqlite3

conn = sqlite3.connect("emotion_data.db")
cursor = conn.cursor()

# Delete all rows from your table
cursor.execute("DELETE FROM aggregated_emotions")
conn.commit()

# Reset the auto-increment counter
cursor.execute("DELETE FROM sqlite_sequence WHERE name='aggregated_emotions'")
conn.commit()

print("Database cleared and auto-increment counter reset.")

conn.close()

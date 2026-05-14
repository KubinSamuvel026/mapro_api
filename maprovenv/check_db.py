import sqlite3

conn = sqlite3.connect("mapro.db")

cursor = conn.cursor()

cursor.execute("SELECT * FROM emotion_history")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
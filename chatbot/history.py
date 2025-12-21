import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("storage/history.db")
DB_PATH.parent.mkdir(exist_ok=True)


class ChatHistoryStore:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                question TEXT,
                answer TEXT
            )
        """)
        self.conn.commit()

    def log(self, question: str, answer: str):
        self.conn.execute(
            "INSERT INTO chat_history VALUES (NULL, ?, ?, ?)",
            (datetime.utcnow().isoformat(), question, answer)
        )
        self.conn.commit()

    def fetch_all_questions(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT question FROM chat_history")
        rows = cursor.fetchall()
        return [r[0] for r in rows]
import sqlite3
from datetime import datetime

# Database file
DB_FILE = "my_database.db"

# Function to create tables
def create_tables():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Create user_queries table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create agent_responses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            query TEXT NOT NULL,
            ml_triage_agent_response TEXT,
            nlp_triage_agent_response TEXT,
            ticket_analysis_agent_response TEXT,
            csv_agent_response TEXT,
            web_search_agent_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES user_queries(query_id) ON DELETE CASCADE
        );
    """)

    # Commit changes and close connection
    conn.commit()
    conn.close()

# Function to store a user query
def store_user_query(query):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_queries (query) VALUES (?)", (query,))
    conn.commit()
    conn.close()

# Function to store an agent response
def store_agent_response(query_id, query, ml_response, nlp_response, ticket_analysis_response, csv_response, web_search_response):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO agent_responses (query_id, query, ml_triage_agent_response, nlp_triage_agent_response, ticket_analysis_agent_response, csv_agent_response, web_search_agent_response)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (query_id, query, ml_response, nlp_response, ticket_analysis_response, csv_response, web_search_response))
    conn.commit()
    conn.close()

# Function to fetch all stored responses (for debugging or viewing data)
def fetch_responses():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM agent_responses")
    results = cursor.fetchall()
    conn.close()
    return results

if __name__ == "__main__":
    create_tables()
    print("table_created")

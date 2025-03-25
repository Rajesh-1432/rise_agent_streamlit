import sqlite3

conn = sqlite3.connect('user_accounts.db')
cursor = conn.cursor()

# Create the table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        ohr_id TEXT,
        status INTEGER
    )
''')
conn.commit()

# Sample 20 user records
users_data = [
    ("user1@example.com", "OHR001", 1),
    ("user2@example.com", "OHR002", 0),
    ("user3@example.com", "OHR003", 1),
    ("user4@example.com", "OHR004", 1),
    ("user5@example.com", "OHR005", 0),
    ("user6@example.com", "OHR006", 1),
    ("user7@example.com", "OHR007", 0),
    ("user8@example.com", "OHR008", 1),
    ("user9@example.com", "OHR009", 1),
    ("user10@example.com", "OHR010", 0),
    ("user11@example.com", "OHR011", 1),
    ("user12@example.com", "OHR012", 1),
    ("user13@example.com", "OHR013", 0),
    ("user14@example.com", "OHR014", 1),
    ("user15@example.com", "OHR015", 0),
    ("user16@example.com", "OHR016", 1),
    ("user17@example.com", "OHR017", 1),
    ("user18@example.com", "OHR018", 0),
    ("user19@example.com", "OHR019", 1),
    ("user20@example.com", "OHR020", 0),
    ("sukirtimaskey@gmail.com", "703394911", 1)
]

# Insert data into the table (ignoring duplicates)
cursor.executemany("INSERT OR IGNORE INTO users (email, ohr_id, status) VALUES (?, ?, ?)", users_data)

# Commit the changes
conn.commit()

# Count the number of records
cursor.execute("SELECT COUNT(*) FROM users")
count = cursor.fetchone()[0]
print(f"Number of records: {count}")
# Fetch and display all records
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
print("\n\bUser Records:")
for row in rows:
    print(row)

# Close the connection
conn.close()

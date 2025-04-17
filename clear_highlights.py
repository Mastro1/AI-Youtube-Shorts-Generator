# clear_highlights.py
import sqlite3
import os

db_path = "video_processing.db"

if not os.path.exists(db_path):
    print(f"Error: Database file '{db_path}' not found.")
    exit()

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Confirm with the user (optional but recommended for delete operations)
    confirm = input(f"Are you sure you want to delete ALL entries from the 'highlights' table in {db_path}? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        exit()

    # Delete all rows from the highlights table
    cursor.execute("DELETE FROM highlights")
    deleted_rows = cursor.rowcount # Get the number of deleted rows
    conn.commit()

    print(f"Successfully deleted {deleted_rows} rows from the 'highlights' table.")

except sqlite3.Error as e:
    print(f"SQLite error: {e}")
    if conn:
        conn.rollback() # Rollback changes on error
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if conn:
        conn.close()
        print("Database connection closed.") 
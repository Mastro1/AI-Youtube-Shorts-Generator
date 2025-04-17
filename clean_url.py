# clean_url.py
import sqlite3
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import os

db_path = "video_processing.db"
target_id = 1

if not os.path.exists(db_path):
    print(f"Error: Database file '{db_path}' not found.")
    exit()

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch the current URL
    cursor.execute("SELECT youtube_url FROM videos WHERE id = ?", (target_id,))
    result = cursor.fetchone()

    if result:
        original_url = result[0]
        print(f"Original URL for ID {target_id}: {original_url}")

        if original_url:
            # Simple string manipulation for this specific case
            cleaned_url = original_url
            if '&t' in original_url:
                # Find the position of &t
                t_pos = original_url.find('&t')
                # Check if it's followed by '=' or just the end of the string
                if t_pos != -1:
                    # Check if it's the last parameter or followed by another &
                    next_amp_pos = original_url.find('&', t_pos + 1)
                    if next_amp_pos == -1:
                        # &t is the last part, remove it
                        cleaned_url = original_url[:t_pos]
                        print("Removed trailing '&t' parameter.")
                    else:
                        # &t=... is in the middle, need proper parsing (or more complex string logic)
                        # Fall back to original parsing method just in case
                        print("'&t' found, but potentially with a value or followed by other params. Using URL parsing...")
                        parsed_url = urlparse(original_url)
                        query_params = parse_qs(parsed_url.query)
                        if 't' in query_params:
                            del query_params['t']
                            new_query_string = urlencode(query_params, doseq=True)
                            cleaned_url_parts = parsed_url._replace(query=new_query_string)
                            cleaned_url = urlunparse(cleaned_url_parts)
                            print("Removed 't' parameter using URL parsing.")
                        else:
                            # If parsing didn't find t=, maybe it was just &t. Try removing the segment again.
                            cleaned_url = original_url[:t_pos] + original_url[next_amp_pos:]
                            print("Removed '&t' segment between other params.")
                else: # &t not found by find()
                     print("'&t' not found in the URL string. No update needed.")       

            if cleaned_url != original_url:
                print(f"Cleaned URL: {cleaned_url}")
                cursor.execute("UPDATE videos SET youtube_url = ? WHERE id = ?", (cleaned_url, target_id))
                conn.commit()
                print(f"Database updated successfully for ID {target_id}.")
            else:
                print("URL was not changed. No update needed.")
        else:
             print("URL is empty or null in the database. No update needed.")

    else:
        print(f"Error: No entry found for ID {target_id}.")

except sqlite3.Error as e:
    print(f"SQLite error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if conn:
        conn.close() 
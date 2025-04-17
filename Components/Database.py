# Components/Database.py
import sqlite3
from typing import Optional, List, Tuple
import json
import os
from datetime import datetime


class VideoDatabase:
    def __init__(self, db_path: str = "video_processing.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create videos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    youtube_url TEXT UNIQUE,
                    local_path TEXT,
                    audio_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create transcriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    transcription_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                )
            """)

            # Create highlights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS highlights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    start_time FLOAT,
                    end_time FLOAT,
                    output_path TEXT,
                    segment_text TEXT,
                    caption_with_hashtags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                )
            """)

            conn.commit()

    def add_video(
        self, youtube_url: Optional[str], local_path: str, audio_path: str
    ) -> int:
        """Add a new video entry and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO videos (youtube_url, local_path, audio_path)
                VALUES (?, ?, ?)
            """,
                (youtube_url, local_path, audio_path),
            )
            return cursor.lastrowid

    def update_video_audio_path(self, video_id: int, audio_path: str) -> bool:
        """Update the audio path for an existing video."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE videos 
                    SET audio_path = ?
                    WHERE id = ?
                """,
                    (audio_path, video_id),
                )
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating video audio path: {e}")
            return False

    def get_video(
        self, youtube_url: Optional[str] = None, local_path: Optional[str] = None
    ) -> Optional[tuple]:
        """Get video entry by URL or local path."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if youtube_url:
                cursor.execute(
                    "SELECT * FROM videos WHERE youtube_url = ?", (youtube_url,)
                )
            elif local_path:
                cursor.execute(
                    "SELECT * FROM videos WHERE local_path = ?", (local_path,)
                )
            else:
                return None
            return cursor.fetchone()

    def add_transcription(
        self, video_id: int, transcriptions: List[Tuple[str, float, float]]
    ) -> int:
        """Add transcription data for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            transcription_json = json.dumps(transcriptions)
            cursor.execute(
                """
                INSERT INTO transcriptions (video_id, transcription_data)
                VALUES (?, ?)
            """,
                (video_id, transcription_json),
            )
            return cursor.lastrowid

    def get_transcription(
        self, video_id: int
    ) -> Optional[List[Tuple[str, float, float]]]:
        """Get transcription data for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT transcription_data FROM transcriptions WHERE video_id = ?",
                (video_id,),
            )
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
            return None

    def add_highlight(
        self, video_id: int, start_time: float, end_time: float, output_path: str,
        segment_text: Optional[str] = None, caption_with_hashtags: Optional[str] = None
    ) -> int:
        """Add a highlight segment for a video with enriched data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO highlights (
                    video_id, start_time, end_time, output_path, 
                    segment_text, caption_with_hashtags
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (video_id, start_time, end_time, output_path, 
                 segment_text, caption_with_hashtags),
            )
            return cursor.lastrowid

    def get_highlights(self, video_id: int) -> List[Tuple[float, float, str, str, str]]:
        """Get all highlights for a video including enriched data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT start_time, end_time, output_path, segment_text, caption_with_hashtags
                FROM highlights 
                WHERE video_id = ?
            """,
                (video_id,),
            )
            return cursor.fetchall()

    def video_exists(
        self, youtube_url: Optional[str] = None, local_path: Optional[str] = None
    ) -> bool:
        """Check if a video exists in the database."""
        return self.get_video(youtube_url, local_path) is not None

    def get_cached_processing(
        self, youtube_url: Optional[str] = None, local_path: Optional[str] = None
    ) -> Optional[dict]:
        """Get all cached processing data for a video."""
        video = self.get_video(youtube_url, local_path)
        if not video:
            return None

        video_id = video[0]
        transcription = self.get_transcription(video_id)
        highlights = self.get_highlights(video_id)

        return {
            "video": video,
            "transcription": transcription,
            "highlights": highlights,
        }

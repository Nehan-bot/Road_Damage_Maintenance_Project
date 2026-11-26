
from pathlib import Path
import sqlite3
import json
import pandas as pd
from datetime import datetime, timezone


ROOT_DIR = Path(__file__).resolve().parent.parent

DB_PATH = ROOT_DIR / "database" / "road_damage.db"


def get_connection():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return sqlite3.connect(DB_PATH)


def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                analyzed_at TEXT NOT NULL,

                image_filename TEXT NOT NULL,

                total_detections INTEGER NOT NULL,

                damage_counts TEXT NOT NULL,

                predicted_priority TEXT NOT NULL,

                probabilities TEXT NOT NULL,

                extracted_features TEXT NOT NULL
            )
        """)


def save_prediction(
    image_filename,
    counts,
    feature_df,
    prediction,
    probabilities
):

    features = feature_df.iloc[0].to_dict()

    with get_connection() as connection:

        connection.execute("""

            INSERT INTO predictions (

                analyzed_at,
                image_filename,
                total_detections,
                damage_counts,
                predicted_priority,
                probabilities,
                extracted_features

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

        """, (

            datetime.now(timezone.utc).isoformat(),

            image_filename,

            int(features["total_detections"]),

            json.dumps(counts),

            str(prediction),

            json.dumps(probabilities),

            json.dumps(features)

        ))


def get_prediction_history(limit=100):

    with get_connection() as connection:

        return pd.read_sql_query("""

            SELECT

                id,
                analyzed_at,
                image_filename,
                total_detections,
                predicted_priority

            FROM predictions

            ORDER BY id DESC

            LIMIT ?

        """, connection, params=(limit,))


def get_prediction_details(record_id):

    with get_connection() as connection:

        cursor = connection.execute("""

            SELECT *

            FROM predictions

            WHERE id = ?

        """, (record_id,))

        return cursor.fetchone()

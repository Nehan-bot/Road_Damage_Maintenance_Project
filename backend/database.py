import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "backend" / "inspection_history.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create the inspections table and safely migrate existing databases."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                detection_count INTEGER NOT NULL,
                detections_json TEXT NOT NULL,
                image_path TEXT,
                priority TEXT,
                priority_confidence REAL,
                priority_probabilities_json TEXT
            )
            """
        )

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(inspections)"
            ).fetchall()
        }

        migrations = {
            "image_path": "TEXT",
            "priority": "TEXT",
            "priority_confidence": "REAL",
            "priority_probabilities_json": "TEXT",
        }

        for column_name, column_type in migrations.items():
            if column_name not in columns:
                connection.execute(
                    f"ALTER TABLE inspections "
                    f"ADD COLUMN {column_name} {column_type}"
                )


def save_inspection(
    filename,
    image_width,
    image_height,
    detections,
    image_path=None,
    priority=None,
    priority_confidence=None,
    priority_probabilities=None,
):
    """Save a completed detection and priority prediction."""
    created_at = datetime.now(timezone.utc).isoformat()

    probabilities_json = (
        json.dumps(priority_probabilities)
        if priority_probabilities is not None
        else None
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO inspections (
                filename,
                created_at,
                image_width,
                image_height,
                detection_count,
                detections_json,
                image_path,
                priority,
                priority_confidence,
                priority_probabilities_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                created_at,
                image_width,
                image_height,
                len(detections),
                json.dumps(detections),
                image_path,
                priority,
                priority_confidence,
                probabilities_json,
            ),
        )

        inspection_id = cursor.lastrowid

    return {
        "id": inspection_id,
        "created_at": created_at,
        "priority": priority,
        "priority_confidence": priority_confidence,
        "priority_probabilities": priority_probabilities,
    }


def list_inspections(limit=50):
    """Return recent inspection summaries."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                created_at,
                image_width,
                image_height,
                detection_count,
                priority,
                priority_confidence
            FROM inspections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_inspection(inspection_id):
    """Return one inspection with detections and priority probabilities."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM inspections
            WHERE id = ?
            """,
            (inspection_id,),
        ).fetchone()

    if row is None:
        return None

    inspection = dict(row)

    inspection["detections"] = json.loads(
        inspection.pop("detections_json")
    )

    probabilities_json = inspection.pop(
        "priority_probabilities_json",
        None,
    )

    inspection["priority_probabilities"] = (
        json.loads(probabilities_json)
        if probabilities_json
        else None
    )

    return inspection
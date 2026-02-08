CREATE TABLE IF NOT EXISTS songs (
    song_id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_name TEXT,
    file_path TEXT
);

CREATE TABLE IF NOT EXISTS landmarks (
    landmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER,
    f1 REAL,
    f2 REAL,
    delta_t REAL,
    time_offset REAL
);

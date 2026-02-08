import numpy as np
import soundfile as sf
from scipy.signal import stft, find_peaks, resample
import sqlite3
import os

# ==============================
# DATABASE SETUP
# ==============================
conn = sqlite3.connect("audio_landmarks.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS songs (
    song_id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_name TEXT,
    file_path TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS landmarks (
    landmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER,
    f1 REAL,
    f2 REAL,
    delta_t REAL,
    time_offset REAL
)
""")

conn.commit()

# ==============================
# LANDMARK EXTRACTION FUNCTION
# ==============================
def extract_landmarks(audio_path):
    y, sr = sf.read(audio_path)

    if y.ndim > 1:
        y = y.mean(axis=1)

    TARGET_SR = 11025
    if sr != TARGET_SR:
        y = resample(y, int(len(y) * TARGET_SR / sr))
        sr = TARGET_SR

    # Select middle 25 seconds
    start = int(30 * sr)
    end = int(55 * sr)
    y = y[start:end]

    # STFT
    f, t, Z = stft(y, fs=sr, nperseg=1024, noverlap=768)
    S = np.abs(Z)

    # Humming frequency range
    mask = (f >= 80) & (f <= 1000)
    f_sel = f[mask]
    S_sel = S[mask, :]

    # Peak picking
    peaks = []
    for ti in range(S_sel.shape[1]):
        col = S_sel[:, ti]
        threshold = np.percentile(col, 95)
        idx, _ = find_peaks(col, height=threshold)
        for i in idx:
            peaks.append((t[ti], f_sel[i]))

    # Landmark generation
    peaks.sort(key=lambda x: x[0])
    landmarks = []

    MAX_DT = 0.5
    MAX_PAIRS = 20

    for i, (t1, f1) in enumerate(peaks):
        for j in range(i + 1, min(i + MAX_PAIRS, len(peaks))):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if 0 < dt <= MAX_DT:
                landmarks.append((round(f1,1), round(f2,1), round(dt,3), round(t1,3)))

    return landmarks

# ==============================
# MAIN LOOP
# ==============================
while True:
    choice = input("\nDo you want to add a song? (yes/no): ").strip().lower()

    if choice != "yes":
        print("Exiting. All landmarks saved.")
        break

    audio_path = input("Enter path to audio file: ").strip()

    if not os.path.exists(audio_path):
        print("❌ File not found. Try again.")
        continue

    song_name = os.path.basename(audio_path)

    print("⏳ Processing song...")
    landmarks = extract_landmarks(audio_path)

    cursor.execute(
        "INSERT INTO songs (song_name, file_path) VALUES (?, ?)",
        (song_name, audio_path)
    )
    song_id = cursor.lastrowid

    for lm in landmarks:
        cursor.execute(
            "INSERT INTO landmarks (song_id, f1, f2, delta_t, time_offset) VALUES (?, ?, ?, ?, ?)",
            (song_id, lm[0], lm[1], lm[2], lm[3])
        )

    conn.commit()
    print(f"✅ Added '{song_name}' with {len(landmarks)} landmarks")

conn.close()

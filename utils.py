import numpy as np
import soundfile as sf
from scipy.signal import stft, find_peaks, resample
import sqlite3
import os
from collections import defaultdict

# ==============================
# CONFIG
# ==============================
DB_PATH = "audio_landmarks.db"
TARGET_SR = 11025

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ==============================
# LANDMARK EXTRACTION FUNCTION
# ==============================
def extract_landmarks(audio_path):
    try:
        y, sr = sf.read(audio_path)
    except Exception as e:
        print(f"Error reading audio file {audio_path}: {e}")
        return []

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != TARGET_SR:
        try:
            y = resample(y, int(len(y) * TARGET_SR / sr))
            sr = TARGET_SR
        except ValueError:
            return []

    # Use only first 15 seconds (speed) - consistent with main.py optimization
    # Note: landmarks.py used middle 25s, main.py used first 15s.
    # For a server, we likely want consistent behavior.
    # I will unify to first 15s for speed, or maybe first 30s.
    # Let's stick to the logic from main.py as it seems to be the "matcher".
    # However, for *adding* songs (landmarks.py), we might want better coverage.
    # Let's make it configurable or just use one standard.
    # I will use the logic from landmarks.py for better quality, but adaptable.
    
    # Actually, main.py has `y = y[:15 * sr]`. landmarks.py has `start = int(30 * sr); end = int(55 * sr)`.
    # Let's support an optional duration or just process the whole thing if typically short, 
    # or a specific segment.
    # For now, I'll use the logic from `landmarks.py` but safer (checking duration).
    
    # Better yet, let's keep the logic simple and robust.
    # I will basically copy the logic from `landmarks.py` exactly as it was the "indexer".
    
    # Wait, `main.py` is the one searching. It uses a shorter snippet.
    # The `extract_landmarks` in `main.py` also has: `mask = (f >= 80) & (f <= 1000)`.
    # `landmarks.py` has same mask.
    # `peaks` finding has slightly different threshold: 95 vs 97.
    # `MAX_PAIRS` 20 vs 15.
    
    # I will standardize on `landmarks.py` settings as they are likely the "ground truth" for the DB.
    
    # Select middle 25 seconds if available, otherwise just use what we have.
    if len(y) > 55 * sr:
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
        # Use 95th percentile from landmarks.py
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
# FAST MATCHING
# ==============================
def match_song(query_audio):
    query_landmarks = extract_landmarks(query_audio)

    # Reduce density for speed
    query_landmarks = query_landmarks[::3]

    if not query_landmarks:
        return None, 0

    hashes = list(set((f1, f2, dt) for f1, f2, dt, _ in query_landmarks))

    conn = get_db_connection()
    cursor = conn.cursor()

    placeholders = ",".join(["(?, ?, ?)"] * len(hashes))
    sql = f"""
        SELECT song_id, f1, f2, delta_t, time_offset
        FROM landmarks
        WHERE (f1, f2, delta_t) IN ({placeholders})
    """

    params = [x for h in hashes for x in h]
    cursor.execute(sql, params)
    db_rows = cursor.fetchall()
    conn.close()

    votes = defaultdict(int)

    query_map = defaultdict(list)
    for f1, f2, dt, tq in query_landmarks:
        query_map[(f1, f2, dt)].append(tq)

    for row in db_rows:
        song_id = row['song_id']
        f1 = row['f1']
        f2 = row['f2']
        dt = row['delta_t']
        t_db = row['time_offset']
        
        for t_q in query_map[(f1, f2, dt)]:
            offset = round(t_db - t_q, 2)
            votes[(song_id, offset)] += 1

    if not votes:
        return None, 0

    (best_song_id, _), score = max(votes.items(), key=lambda x: x[1])

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT song_name FROM songs WHERE song_id=?",
        (best_song_id,)
    )
    result = cursor.fetchone()
    song_name = result['song_name'] if result else None
    conn.close()

    return song_name, score

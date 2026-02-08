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

# ==============================
# LANDMARK EXTRACTION
# ==============================
def extract_landmarks(audio_path):
    y, sr = sf.read(audio_path)

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != TARGET_SR:
        y = resample(y, int(len(y) * TARGET_SR / sr))
        sr = TARGET_SR

    # Use only first 15 seconds (speed)
    y = y[:15 * sr]

    f, t, Z = stft(y, fs=sr, nperseg=1024, noverlap=768)
    S = np.abs(Z)

    mask = (f >= 80) & (f <= 1000)
    f_sel = f[mask]
    S_sel = S[mask, :]

    peaks = []
    for ti in range(S_sel.shape[1]):
        col = S_sel[:, ti]
        threshold = np.percentile(col, 97)
        idx, _ = find_peaks(col, height=threshold)
        for i in idx:
            peaks.append((t[ti], f_sel[i]))

    peaks.sort(key=lambda x: x[0])

    landmarks = []
    MAX_DT = 0.5
    MAX_PAIRS = 15

    for i, (t1, f1) in enumerate(peaks):
        for j in range(i + 1, min(i + MAX_PAIRS, len(peaks))):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if 0 < dt <= MAX_DT:
                landmarks.append(
                    (round(f1, 1), round(f2, 1), round(dt, 3), round(t1, 3))
                )

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

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
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

    for song_id, f1, f2, dt, t_db in db_rows:
        for t_q in query_map[(f1, f2, dt)]:
            offset = round(t_db - t_q, 2)
            votes[(song_id, offset)] += 1

    if not votes:
        return None, 0

    (best_song_id, _), score = max(votes.items(), key=lambda x: x[1])

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT song_name FROM songs WHERE song_id=?",
        (best_song_id,)
    )
    song_name = cursor.fetchone()[0]
    conn.close()

    return song_name, score

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    query_path = input("Enter query audio path (song/humming): ").strip()

    if not os.path.exists(query_path):
        print("❌ File not found")
        exit()

    print("🔍 Matching...")
    song, score = match_song(query_path)

    if song and score >= 15:
        print("\n🎵 MATCH FOUND")
        print("Song:", song)
        print("Confidence score:", score)
    else:
        print("\n❌ No confident match found")


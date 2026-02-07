from flask import Flask, request, jsonify, render_template
import os
import sqlite3
from utils import extract_landmarks, get_db_connection, match_song
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save to temp file to ensure sf.read works as expected with paths
        # and to handle any format quirks.
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            song_name, score = match_song(tmp_path)
            result = {
                "song": song_name,
                "confidence": score,
                "match_found": bool(song_name and score >= 15) # Using 15 as threshold from main.py
            }
            return jsonify(result)
        finally:
            os.remove(tmp_path)

@app.route('/add_song', methods=['POST'])
def add_song_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Logic from landmarks.py to add song
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if song exists?
            # landmarks.py didn't check.
            
            cursor.execute(
                "INSERT INTO songs (song_name, file_path) VALUES (?, ?)",
                (filename, file_path)
            )
            song_id = cursor.lastrowid
            
            landmarks = extract_landmarks(file_path)
            
            for lm in landmarks:
                cursor.execute(
                    "INSERT INTO landmarks (song_id, f1, f2, delta_t, time_offset) VALUES (?, ?, ?, ?, ?)",
                    (song_id, lm[0], lm[1], lm[2], lm[3])
                )
            
            conn.commit()
            conn.close()
            
            return jsonify({
                "status": "success",
                "song_id": song_id,
                "song_name": filename,
                "landmarks_count": len(landmarks)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

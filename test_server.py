import requests
import numpy as np
import soundfile as sf
import time
import os

# Generate a minimal WAV file
def generate_sine_wave(filename, duration=5, samplerate=44100, frequency=440):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    # Add some harmonics to make it interesting enough for landmarks?
    # Landmarks need peaks. A pure sine might have only one peak/frequency.
    # Let's add a few frequencies.
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio += 0.3 * np.sin(2 * np.pi * (frequency * 1.5) * t)
    audio += 0.2 * np.sin(2 * np.pi * (frequency * 2.0) * t)
    sf.write(filename, audio, samplerate)
    print(f"Generated {filename}")

def test_server():
    server_url = "http://127.0.0.1:5000"
    test_file = "test_audio.wav"
    generate_sine_wave(test_file)

    try:
        # 1. Add Song
        print("Testing /add_song...")
        with open(test_file, 'rb') as f:
            r = requests.post(f"{server_url}/add_song", files={'file': f})
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json()}")
        
        if r.status_code != 200:
            print("Failed to add song.")
            return

        # 2. Match Song
        print("\nTesting /match...")
        with open(test_file, 'rb') as f:
            r = requests.post(f"{server_url}/match", files={'file': f})
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json()}")
        
        # Verify match
        data = r.json()
        if data.get('match_found'):
            print("SUCCESS: Song matched!")
        else:
            print("FAILURE: Song not matched (might be due to low complexity audio or thresholds)")

    except Exception as e:
        print(f"Error: {e}")
        print("Is the server running?")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_server()

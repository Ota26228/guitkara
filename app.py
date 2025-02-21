from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import numpy as np
import subprocess
import wave
from pydub import AudioSegment
import librosa

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
RECORDED_FOLDER = os.path.join(app.root_path, 'static', 'recorded')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORDED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECORDED_FOLDER'] = RECORDED_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play/<filename>')
def play(filename):
    songs = [ 
        {
        'title': 'y2mate.com - ABC Riff.mp3',
        'tab': '''
E: --------------------------------sl------------sl-----------------------------sl-----------p-----------------------------------T---------T----------------
B: -----------------------------10----12-----10--11--10-----8---------------10------11---10-----8---------x------10----------------------------------------
G: ------9/------------x-------------------------------------------9--------8-----------------------------x------11---------------------------------------
D: -------------------------------------------------------------------------------------------------------x------13--------------21-----------------------
A: ------7/------------x----------------------------------------------------6----------------------------------------------------------------------------------
E: ------8/------------x----------------------------------------------------7--------------------------------------------------------------7----------------'''
        },
        {
        'title': 'y2mate.com - Polyphia Album 4 Teaser.mp3',
        'tab': '''
E: ------------
B: ------4-----
G: ------2-----
D: --0--------3-
A: ---------------------------------
E: ------------'''
        },
        {
        'title': 'y2mate.com - Bloodbath Solo.mp3',
        'tab': '''6th String: ------------
5th String: --0--3-----
4th String: ------4-----
3rd String: ------2-----
2nd String: --0--------3-
1st String: ------------'''
        }
]


    song = next((song for song in songs if song['title'] == filename), None)

    if song is None:
        return f"Song '{filename}' not found", 404

    return render_template('play.html', filename=filename, song=song)

@app.route('/uploads/<filename>')
def uploaded_audio_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/recorded/<filename>')
def recorded_audio_file(filename):
    return send_from_directory(app.config['RECORDED_FOLDER'], filename)

def read_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        num_frames = wf.getnframes()
        audio = wf.readframes(num_frames)
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    return audio

def analyze_pitch(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None, mono=True)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[magnitudes[:, t].argmax(), t] > 0]
        return np.mean(pitch_values) if pitch_values else 0
    except Exception:
        return 0

def calculate_score(recorded_audio, target_audio):
    y = read_wav(recorded_audio)
    x = read_wav(target_audio)

    if np.all(y == 0):
        return 40  # 無音なら最低スコア

    min_length = min(len(x), len(y))
    y, x = y[:min_length], x[:min_length]
    
    correlation = np.corrcoef(y, x)[0, 1] if np.std(y) > 0 and np.std(x) > 0 else 0
    score_from_correlation = (correlation + 1) * 10  # 最大20点

    pitch_score = (analyze_pitch(recorded_audio) / 100) * 3  # 最大3点

    final_score = 40 + score_from_correlation + pitch_score
    return max(40, min(85, int(final_score)))  # 40~85点に制限


@app.route('/save_recording', methods=['POST'])
def save_recording():
    if 'audio_file' not in request.files:
        return 'No file uploaded', 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return 'No file selected', 400

    webm_path = os.path.join(app.config['RECORDED_FOLDER'], 'recorded_audio.webm')
    wav_path = os.path.join(app.config['RECORDED_FOLDER'], 'recorded_audio.wav')
    audio_file.save(webm_path)

    try:
        AudioSegment.from_file(webm_path, format="webm").export(wav_path, format="wav")
        os.remove(webm_path)  # WebM削除
        return 'Recording saved successfully', 200
    except Exception as e:
        return f"Error processing audio file: {str(e)}", 500

@app.route('/result/<filename>')
def result(filename):
    recorded_audio = os.path.join(app.config['RECORDED_FOLDER'], 'recorded_audio.wav')
    target_audio_mp3 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    target_audio_wav = target_audio_mp3.replace(".mp3", ".wav")

    try:
        if not os.path.isfile(target_audio_wav):
            subprocess.run(['ffmpeg', '-i', target_audio_mp3, target_audio_wav], check=True)

        if not os.path.isfile(recorded_audio):
            return "Recording file does not exist", 404

        score = calculate_score(recorded_audio, target_audio_wav)
        return render_template('result.html', score=int(score), recorded_audio_file=url_for('recorded_audio_file', filename='recorded_audio.wav'))
    except subprocess.CalledProcessError:
        return "Error converting audio file", 500
    except Exception as e:
        return f"Error processing audio files: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)


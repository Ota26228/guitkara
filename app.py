from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import subprocess
import wave

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

@app.route('/play_audio/<filename>')
def play_audio(filename):
    return render_template('play.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_audio_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/recorded/<filename>')
def recorded_audio_file(filename):
    return send_from_directory(app.config['RECORDED_FOLDER'], filename)

@app.route('/save_recording', methods=['POST'])
def save_recording():
    if 'audio_file' not in request.files:
        print('No file uploaded')
        return 'No file uploaded', 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        print('No file selected')
        return 'No file selected', 400

    recording_path = os.path.join(app.config['RECORDED_FOLDER'], 'recorded_audio.wav')
    audio_file.save(recording_path)
    print(f'Recording saved at: {recording_path}')

    return 'Recording saved successfully'

@app.route('/result/<filename>')
def result(filename):
    recorded_audio_path = os.path.join(app.config['RECORDED_FOLDER'], 'recorded_audio.wav')
    specific_audio_path_mp3 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    specific_audio_path_wav = specific_audio_path_mp3.replace(".mp3", ".wav")

    try:
        # MP3からWAVに変換
        if not os.path.exists(specific_audio_path_wav):
            command = ['ffmpeg', '-i', specific_audio_path_mp3, specific_audio_path_wav]
            subprocess.run(command, check=True)

        def read_wav(filepath):
            with wave.open(filepath, 'rb') as wf:
                num_frames = wf.getnframes()
                audio = wf.readframes(num_frames)
                audio = np.frombuffer(audio, dtype=np.int16)
            return audio

        print(f"Recorded audio path: {recorded_audio_path}")
        print(f"Specific audio path: {specific_audio_path_wav}")

        # 録音された音声ファイルを読み込み
        y = read_wav(recorded_audio_path)
        print(f"Length of recorded audio: {len(y)}")

        # 特定の音声ファイルを読み込み
        x = read_wav(specific_audio_path_wav)
        print(f"Length of specific audio: {len(x)}")

        # データを正規化
        y = (y - np.mean(y)) / np.std(y)
        x = (x - np.mean(x)) / np.std(x)

        # 両方の音声ファイルの長さを揃える
        min_length = min(len(x), len(y))
        y = y[:min_length]
        x = x[:min_length]

        # 相関係数を計算
        correlation = np.corrcoef(y, x)[0, 1]

        # スコアを100点満点に正規化
        score = max(0, min(100, (correlation + 1) / 2 * 100))

        return render_template('result.html', score=int(score), recorded_audio_file=url_for('recorded_audio_file', filename='recorded_audio.wav'))

    except Exception as e:
        return f"Error processing audio files: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)


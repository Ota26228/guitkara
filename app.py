from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import subprocess
import wave
from pydub import AudioSegment  # 追加
import librosa  # 音声解析ライブラリ

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


def analyze_pitch_and_timing(audio_file):
    # 音声ファイルをロード
    y, sr = librosa.load(audio_file)

    # ピッチ（基本周波数）の推定
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    # ピッチが最も強い位置を検出（最初のフレームのみ使用）
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_value = pitches[index, t]
        if pitch_value > 0:  # ピッチが0の場合は無視
            pitch.append(pitch_value)

    # ピッチの一致度を計算（ここでは簡易的にピッチの平均で計算）
    pitch_score = np.mean(pitch) if pitch else 0
    return pitch_score


def read_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        num_frames = wf.getnframes()
        audio = wf.readframes(num_frames)
        audio = np.frombuffer(audio, dtype=np.int16)
    return audio


def calculate_score(recorded_audio_path, specific_audio_path):
    # 録音された音声と比較対象の音声を読み込む
    y = read_wav(recorded_audio_path)
    x = read_wav(specific_audio_path)

    # 音声が無音の場合、スコアを60点に設定
    if np.all(y == 0):
        return 60  # 無音の場合はスコア60点

    # データを正規化
    y = (y - np.mean(y)) / np.std(y)
    x = (x - np.mean(x)) / np.std(x)

    # 両方の音声ファイルの長さを揃える
    min_length = min(len(x), len(y))
    y = y[:min_length]
    x = x[:min_length]

    # 相関係数を計算
    correlation = np.corrcoef(y, x)[0, 1]

    # ピッチスコアを計算
    pitch_score = analyze_pitch_and_timing(recorded_audio_path)

    # 相関係数をスコアに変換（-1から1の範囲を0から100に変換）
    score_from_correlation = (correlation + 1) / 2 * 40  # 60点をベースに相関係数で40点分調整

    # ピッチスコアを10点加算（ピッチが無音の場合は0点）
    score_from_pitch = (pitch_score / 100) * 10

    # 最終スコア（60点ベース）
    score = 60 + score_from_correlation + score_from_pitch

    # スコアを100点満点に制限
    return max(60, min(100, score))


@app.route('/save_recording', methods=['POST'])
def save_recording():
    if 'audio_file' not in request.files:
        print('No file uploaded')
        return 'No file uploaded', 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        print('No file selected')
        return 'No file selected', 400

    # WebM の一時保存
    webm_path = os.path.join(app.config['RECORDED_FOLDER'], 'recorded_audio.webm')
    wav_path = os.path.join(app.config['RECORDED_FOLDER'], 'recorded_audio.wav')

    audio_file.save(webm_path)
    print(f'Recording saved at: {webm_path}')

    try:
        # WebM → WAV 変換
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")
        print(f'Converted to WAV: {wav_path}')

        return 'Recording saved successfully', 200

    except Exception as e:
        return f"Error processing audio file: {str(e)}", 500

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

        print(f"Recorded audio path: {recorded_audio_path}")
        print(f"Specific audio path: {specific_audio_path_wav}")

        # 録音された音声ファイルを読み込み
        if not os.path.exists(recorded_audio_path):
            return "Recording file does not exist", 404
        
        # スコア計算
        score = calculate_score(recorded_audio_path, specific_audio_path_wav)

        return render_template('result.html', score=int(score), recorded_audio_file=url_for('recorded_audio_file', filename='recorded_audio.wav'))

    except Exception as e:
        return f"Error processing audio files: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)


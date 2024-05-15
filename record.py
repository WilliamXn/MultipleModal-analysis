import cv2
import mediapipe as mp
import librosa
import numpy as np
import pandas as pd
import os
import subprocess
from utils import process_line

def chordgram(y_, sr_, hop_length):
    y_harmonic, y_percussive = librosa.effects.hpss(y_)
    y_harmonic_processed = np.where(np.isfinite(y_harmonic), y_harmonic, 0)
    _chromagram = librosa.feature.chroma_cqt(y=y_harmonic_processed, sr=sr_, hop_length=hop_length)
    chroma_vectors = np.transpose(_chromagram)
    return chroma_vectors

def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path]
        subprocess.call(command)
    else:
        print(f"Audio file already exists: {audio_path}")

def record_data(chord: str, chord_path: str, video_folder: str, hop_length: int):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()

    volume_threshold = -20
    dataset = []

    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_folder, filename)
            print(f"Processing video: {video_path}")
            audio_name = os.path.splitext(filename)[0] + ".mp3"
            audio_path = os.path.join(video_folder, audio_name)
            y, sr = librosa.load(audio_path, sr=44100)
            chroma = chordgram(y, sr, hop_length)
            frame_rate = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
            frame_period = int(1 / frame_rate * sr / hop_length)  # 计算音频帧周期

            cap = cv2.VideoCapture(video_path)
            audio_frame_index = 0  # 当前音频帧索引

            try:
                while True:
                    success, img = cap.read()
                    if not success:
                        break

                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(imgRGB)

                    if np.max(librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024)), ref=np.max)) > volume_threshold:
                        if results.multi_hand_landmarks:
                            for handLms in results.multi_hand_landmarks:
                                w, h, _ = img.shape
                                hand_data = [[lm.x, lm.y] for lm in handLms.landmark] # [[12, 33], [33, 121], ...]
                                print("处理前: ", hand_data)
                                hand_data = process_line(hand_data) # [1.2, 2.2, ...] 
                                hand_data = np.array(hand_data)
                                print("处理后: ", hand_data)
                                # 获取对应时间的三帧音频数据
                                audio_data = chroma[audio_frame_index:audio_frame_index + 3].flatten()

                                combined_data = np.concatenate([hand_data, audio_data])
                                dataset.append(combined_data)

                    # 更新音频帧索引
                    audio_frame_index += frame_period

            finally:
                cap.release()

    df = pd.DataFrame(dataset)
    df.to_csv(chord_path, index=False)
    print(f"Data saved at: {chord_path}")

# 示例用法
video_folder = "./video/C"
chord = "C"
chord_path = "csv/C/C.csv"
hop_length = 512

record_data(chord, chord_path, video_folder, hop_length)
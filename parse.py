import cv2
import moviepy.editor as mp
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 视频文件路径
video_path = 'your_video.mp4'

# 目标帧率
target_fps = 1  # 每秒一帧

# 使用 MoviePy 读取视频并提取音频
video = mp.VideoFileClip(video_path)
audio = video.audio

# 使用 OpenCV 读取视频
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的原始帧率
duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

# 设置读取频率
frame_period = int(fps / target_fps)  # 每 frame_period 帧读取一帧
current_frame = 0

while True:
    ret = cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        break

    # 处理视频帧（示例：仅显示帧编号）
    print("Processing frame number: ", current_frame)
    # 你可以在这里加入你的视频帧处理代码

    # 对应的音频处理
    start_time = current_frame / fps
    end_time = (current_frame + frame_period) / fps if (current_frame + frame_period) < duration * fps else duration
    audio_segment = audio.subclip(start_time, end_time)
    audio_segment.write_audiofile(f'audio_{current_frame}.wav')  # 将音频片段保存为文件

    # 使用 Librosa 分析音频
    y, sr = librosa.load(f'audio_{current_frame}.wav')
    # 在这里添加音频分析代码，如显示波形等
    librosa.display.waveshow(y, sr=sr)
    plt.show()

    # 更新帧编号
    current_frame += frame_period

cap.release()
audio.close()
video.close()

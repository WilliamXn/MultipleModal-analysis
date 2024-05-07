import os
import moviepy.editor as mp


def extract_audio(videos_folder_path):
    for file_name in os.listdir(videos_folder_path):
        if file_name.endswith(".mp4"):
            video_file_path = os.path.join(videos_folder_path, file_name)
            audio_file_path = os.path.splitext(video_file_path)[0] + ".mp3"

            my_clip = mp.VideoFileClip(video_file_path)
            my_clip.audio.write_audiofile(audio_file_path)


# 指定要处理的文件夹路径
folder_path = r'D:\PycharmProjects\pythonProject\MultipleModal\video\C'

extract_audio(folder_path)
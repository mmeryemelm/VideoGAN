import cv2
import os
from glob import glob


def extract_and_save_frames(video_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"frame_{frame_count:02d}.png"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted and saved {frame_count} frames from {video_path} to {output_folder}")


def process_all_videos(video_folder, output_base_folder):

    gesture_folders = glob(os.path.join(video_folder, "gesture*"))
    for folder in gesture_folders:
        video_files = glob(os.path.join(folder, "*.avi"))
        for video_file in video_files:
            video_name = os.path.basename(video_file)
            gesture_index, subject_index, repetition_index = video_name[1:3], video_name[4:6], video_name[7:8]
            #output_folder = os.path.join(output_base_folder, subject_index, gesture_index, repetition_index)
            #extract_and_save_frames(video_file, output_folder)

            output_folder = os.path.join(output_base_folder, f"subject{int(subject_index):03d}", f"gesture{int(gesture_index):03d}", f"take{int(repetition_index):03d}")
            extract_and_save_frames(video_file, output_folder)


def main():
    video_folder = r'C:\Users\DELL\Desktop\pfe\2'
    output_base_folder = r'C:\Users\DELL\Desktop\pfe\3'
    process_all_videos(video_folder, output_base_folder)


if __name__ == '__main__':
    main()

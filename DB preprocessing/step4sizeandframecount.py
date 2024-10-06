import cv2
import numpy as np
import os
import glob

def interpolate_frames(frames, target_length):
    index = np.linspace(0, len(frames) - 1, num=target_length)
    interpolated_frames = [frames[int(i)] for i in index]
    return interpolated_frames

def downsample_frames(frames, target_length):
    step = len(frames) / target_length
    downsampled_frames = [frames[int(i * step)] for i in range(target_length)]
    return downsampled_frames

def process_folder(folder_path, target_frame_count, output_folder):
    frames = []
    # Load all frames from the folder
    for filename in sorted(os.listdir(folder_path)):
        frame = cv2.imread(os.path.join(folder_path, filename))
        # Resize frame to 64x64 pixels
        frame = cv2.resize(frame, (64, 64))
        frames.append(frame)

    # Interpolate or downsample frames
    if len(frames) > target_frame_count:
        frames = downsample_frames(frames, target_frame_count)
    elif len(frames) < target_frame_count:
        frames = interpolate_frames(frames, target_frame_count)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed frames with JPEG compression to minimize file size
    for idx, frame in enumerate(frames):
        # Adjust the quality parameter to achieve the desired file size
        cv2.imwrite(os.path.join(output_folder, f'frame_{idx:03}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

def main():
    input_base_path = r"C:\Users\DELL\Desktop\pfe\3"
    output_base_path = r"C:\Users\DELL\Desktop\pfe\downsize"
    target_frame_count = 32

    # Find all folders matching the structure
    paths = glob.glob(input_base_path + "\\*\\*\\*")

    for folder_path in paths:
        if os.path.isdir(folder_path):
            # Construct the corresponding output directory
            relative_path = os.path.relpath(folder_path, input_base_path)
            output_folder = os.path.join(output_base_path, relative_path)
            process_folder(folder_path, target_frame_count, output_folder)
            print(f"Processed and saved frames for {output_folder}")

if __name__ == "__main__":
    main()

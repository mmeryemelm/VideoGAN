import cv2
import os


def parse_segmentation_file(filepath):
    print(f"Opening segmentation file: {filepath}")
    with open(filepath, 'r') as file:
        lines = file.readlines()
    data = []
    current_subject = None
    current_gesture = None
    for line in lines:
        line = line.strip()
        if line.startswith('// Subject'):
            current_subject = line.split()[2]
        elif ',' in line and line[0].isdigit() and line.count(',') == 2:
            parts = line.split(',')
            if len(parts) == 3:
                # Only set current_subject and current_gesture if the parts can be verified as purely numeric
                try:
                    int(parts[0])  # Check if subject part is an integer
                    int(parts[1])  # Check if gesture part is an integer
                    current_subject, current_gesture, _ = parts
                except ValueError:
                    continue  # Skip this line if the values are not purely numeric
        elif ':' in line and line[0].isdigit():
            parts = line.split(':')
            if len(parts) == 2:
                repeat_no, indices = parts
                start_idx, end_idx = indices.split(',')
                data.append({
                    'subject_no': current_subject,
                    'gesture_no': current_gesture,
                    'repeat_no': repeat_no,
                    'start_idx': int(start_idx.strip()),  # Ensure no whitespace issues
                    'end_idx': int(end_idx.strip())        # Ensure no whitespace issues
                })
    return data



def process_video(subject_no, gesture_no, repeat_no, start_frame, end_frame, source_dir, dest_dir):
    # Ensure gesture and subject numbers have leading zeros if they are single digits
    formatted_gesture_no = f"{int(gesture_no):02d}"
    formatted_subject_no = f"{int(subject_no):02d}"

    video_path = os.path.join(source_dir, f'gesture{formatted_gesture_no}',
                              f'g{formatted_gesture_no}s{formatted_subject_no}.avi')
    dest_path = os.path.join(dest_dir, f'gesture{formatted_gesture_no}')
    output_filename = f'g{formatted_gesture_no}s{formatted_subject_no}r{repeat_no}.avi'
    output_filepath = os.path.join(dest_path, output_filename)

    # Check if the output file already exists
    if os.path.exists(output_filepath):
        print(f"Skipping existing file: {output_filepath}")
        return

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error opening video file {video_path}')
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filepath, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = start_frame
    while frame_count <= end_frame and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1
        else:
            break

    cap.release()
    out.release()
    print(f"Processed video for gesture {formatted_gesture_no}, subject {formatted_subject_no}, repeat {repeat_no}")



def main():
    source_dir = r'C:\Users\DELL\Desktop\pfe\1'
    dest_dir = r'C:\Users\DELL\Desktop\pfe\2'
    data_file_path = r'C:\Users\DELL\Desktop\pfe\n\segmentation.txt'
    segmentation_data = parse_segmentation_file(data_file_path)
    for entry in segmentation_data:
        process_video(entry['subject_no'], entry['gesture_no'], entry['repeat_no'], entry['start_idx'],
                      entry['end_idx'], source_dir, dest_dir)
    print("Video segmentation completed.")


if __name__ == "__main__":
    main()

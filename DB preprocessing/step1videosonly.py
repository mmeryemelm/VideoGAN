import os
import glob
import shutil


def load_data(source_base, destination_base):
    # Ensure the base destination directory exists, if not, create it
    if not os.path.exists(destination_base):
        os.makedirs(destination_base)

    # Debug: print all found gesture folders
    gesture_folders = glob.glob(os.path.join(source_base, 'gesture[0-9][0-9]'))
    print("Found gesture folders:", gesture_folders)

    for gesture_folder in gesture_folders:
        # Debug: print current gesture folder being processed
        print("Processing folder:", gesture_folder)

        # Find all .avi files in the current gesture folder that match the pattern g[0-9]+s[0-9]+.avi
        #video_files = glob.glob(os.path.join(gesture_folder, 'g[0-9]+s[0-9]+.avi'))
        video_files = glob.glob(os.path.join(gesture_folder + "/*.avi"))

        # Debug: print found video files
        print("Found video files:", video_files)

        for video_file in video_files:
            # Ensure the file does not end with 'd.avi' or 'm.avi'
            if video_file.endswith('d.avi') or video_file.endswith('m.avi'):
                print("Skipping file:", video_file)
                continue  # Skip copying this file

            # Prepare the destination directory structure
            relative_path = os.path.relpath(gesture_folder, source_base)
            dest_folder = os.path.join(destination_base, relative_path)

            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
                print("Created directory:", dest_folder)

            # Define the destination file path
            dest_file = os.path.join(dest_folder, os.path.basename(video_file))

            # Copy the video file to the new location
            shutil.copy(video_file, dest_file)
            print(f'Copied: {video_file} to {dest_file}')


def main():
    source_base = r'C:\Users\DELL\Desktop\pfe\database'
    destination_base = r'C:\Users\DELL\Desktop\pfe\1'
    load_data(source_base, destination_base)


if __name__ == "__main__":
    main()

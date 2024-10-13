import PIL
import PIL.Image
import skvideo.io
import os

skvideo.setFFmpegPath(r"C:\Users\DELL\Desktop\ffmpeg-2024-05-13-git-37db0454e4-full_build\bin")



import skimage.transform
import numpy as np
# import imageio

"""
Largely from TF ver https://github.com/GV1028/videogan
"""

# def save_gen(generated_images, n_ex = 36, epoch = 0, iter = 0):
#     for i in range(generated_images.shape[0]):
#         # cv2.imwrite('/root/code/Video_Generation/gen_images/image_' + str(epoch) + '_' + str(iter) + '_' + str(i) + '.jpg', generated_images[i, :, :, :])
#         PIL.Image.fromarray(np.around(generated_images[i, :, :, :]).astype(np.uint8)).save('/root/code/Video_Generation/gen_images/image_' + str(epoch) + '_' + str(iter) + '_' + str(i) + '.jpg')

def process_and_write_image(images,name):
    images = np.array(images).transpose((0,2,3,4,1))
    images = (images + 1)*127.5

    # Create folder for frames if it doesn't exist

    frame_folder = os.path.splitext(name.strip(".mp4"))[0]
    print(f"Saving frames to folder: {frame_folder}")

    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    for i in range(images.shape[0]):
        frame = np.around(images[i, 0, :, :, :]).astype(np.uint8)
        frame_path = os.path.join(frame_folder, f"{i}.jpg")
        PIL.Image.fromarray(frame).save(frame_path)

    #for i in range(images.shape[0]):
        #PIL.Image.fromarray(np.around(images[i,0,:,:,:]).astype(np.uint8)).save("./genvideos/" + name + ".jpg")



def read_and_process_video(files,size,nof):
    videos = np.zeros((size,nof,64,64,3))
    counter = 0
    for file in files:
        vid = skvideo.io.vreader(file)
        curr_frames = []
        i = 0

        nr = np.random.randint(20)
        for frame in vid:
            i = i + 1
            if i <= nr:
                continue

            frame = skimage.transform.resize(frame,[64,64])
            curr_frames.append(frame)

            if i >= nr+nof:
                break

        curr_frames = np.array(curr_frames)
        curr_frames = curr_frames*255.0
        curr_frames = curr_frames/127.5 - 1
        videos[counter,:,:,:,:] = curr_frames
        counter = counter + 1

    return videos.transpose((0,4,1,2,3)).astype(np.float32)

def process_and_write_video(videos,name):
    videos = np.array(videos)
    videos = np.reshape(videos,[-1,3,32,64,64]).transpose((0,2,3,4,1))
    vidwrite = np.zeros((32,64,64,3))

    output_path = os.path.join(name)
    #frame_folder = output_path.strip(".mp4")  # Create folder path based on video name
    #frame_folder = os.path.splitext(output_path.strip(".mp4") )[0]

    #if not os.path.exists(frame_folder):
        #os.makedirs(frame_folder)

    for i in range(videos.shape[0]):
        vid = videos[i,:,:,:,:]
        vid = (vid + 1)*127.5

        for j in range(vid.shape[0]):
            frame = vid[j,:,:,:]
            vidwrite[j,:,:,:] = frame
            #frame_filename = f"{j}.jpg"
            #frame_path = os.path.join(frame_folder, frame_filename)  # Combine folder and filename

            #os.makedirs(frame_folder, exist_ok=True)  # Create folder if it doesn't exist
            #frame_path = os.path.join(frame_folder, f"{name}_0_0.jpg")  # Frame filename within folder
            #PIL.Image.fromarray(np.around(frame).astype(np.uint8)).save(frame_path)

        skvideo.io.vwrite(output_path, vidwrite, inputdict={'-r': '24'})



        # Specify desired FPS directly as '24'


        #     print("About to write video to:", output_path)
        #    print("Video data shape:", vidwrite.shape)
        #  print("Data type:", vidwrite.dtype)
 # Convert back to 0-255 range

        #   print("Video written successfully")
        #   print("Video shape:", vidwrite.shape)
        #   print("Data type:", vidwrite.dtype)
        #    print("Pixel range:", np.min(vidwrite), np.max(vidwrite))

        #        skvideo.io.vwrite("./"+ name + ".mp4",vidwrite)
        #     imageio.mimsave("./" + name, vidwrite)  # Adjust fps for your video frame rate



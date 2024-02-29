import cv2
import os

# Directory containing your images (update this path)
# image_folder = 'output/colorpeel_e4c_maroon2olive/chair'

folder_name='colorpeel_e4c_maroon2navy_10000steps_multi_vertex_25_42_6.0_3_personlized'

image_folder = f'output/{folder_name}/chair'
# Video file name and format
video_name = f'output/{folder_name}/chair.mp4'

# Frame rate (change this value as needed)
frame_rate = 4

# Get the list of image files in the directory
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# Sort the images to ensure they are in the correct order
images.sort()

# Get the dimensions of the first image (assuming all images have the same dimensions)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Create a VideoWriter object to write the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 format
video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# Loop through the images and add them to the video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    video.write(img)

# Release the video writer
video.release()

print(f"Video '{video_name}' created successfully.")

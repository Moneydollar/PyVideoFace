import glob
import json
import os
import re


import cv2 as cv
import face_recognition
import numpy as np
from cv2 import VideoCapture




totalFaces = 0


def setup(video_path):

    videoName = video_path.split('/')[-1]

    frames_output_dir = os.path.join("/../.cache/output", videoName)
    frames_with_boxes_dir = os.path.join("/../.cache/boxed", videoName)
    faces_dir = os.path.join("/../faces", videoName)
    print(f"--------------{faces_dir}--------------")

    os.makedirs(frames_output_dir, exist_ok=True)
    os.makedirs(frames_with_boxes_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    return frames_output_dir, frames_with_boxes_dir, faces_dir


def natural_sort_frames(frame_list):
    """
    Sort frames naturally by their numbers (frame1 before frame2 before frame10)

    Args:
        frame_list: List of frame filenames

    Returns:
        List of sorted frame filenames
    """

    # Extract numbers from filenames using regex
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    # Sort based on the extracted numbers
    return sorted(frame_list, key=extract_number)


def cleanUp(frames_output_dir, frames_with_boxes_dir):
    # Remove all files from the frames output and boxed directories
    for dir_path in [frames_output_dir, frames_with_boxes_dir]:
        files = glob.glob(os.path.join(dir_path, "*"))
        for f in files:
            os.remove(f)


def videoFrames(video_path, frames_output_dir):
    vidcap: VideoCapture = cv.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    success, image = vidcap.read()
    count = 0

    while success:
        frame_path = os.path.join(frames_output_dir, f"frame{count}.jpg")
        cv.imwrite(frame_path, image)
        success, image = vidcap.read()
        print(f"Read a new frame: {success}")
        count += 1

    vidcap.release()


def detectFaces(frames_output_dir, frames_with_boxes_dir, faces_dir):
    global totalFaces
    frameList = natural_sort_frames(os.listdir(frames_output_dir))
    os.makedirs(frames_with_boxes_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)
    batch_size = 25
    current_batch = 0
    batch_metadata = []

    for frame_file in frameList:
        frame_path = os.path.join(frames_output_dir, frame_file)
        frame = face_recognition.load_image_file(frame_path)

        if frame is None:
            print(f"Error loading image {frame_path}. Skipping...")
            continue

        face_locations = face_recognition.face_locations(frame, model="hog")
        print(f"Detected {len(face_locations)} faces in {frame_file}")
        if len(face_locations) <= 0:
            save = False
        else:
            save = True


        totalFaces += len(face_locations)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        for top, right, bottom, left in face_locations:
            cv.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            face_image = frame[top:bottom, left:right]
            frame_number = frame_file.split("frame")[-1].split(".")[0]
            metadata = {
                "frame": {
                    "frameNumber": frame_number
                },
                "bounding_box": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }
            }
            batch_metadata.append(metadata)

            if save:
                output_path = os.path.join(faces_dir, frame_file)
                cv.imwrite(output_path, face_image)

        if len(batch_metadata) == batch_size:
            batch_filename = f"metadata_batch_{current_batch}.json"
            batch_path = os.path.join(faces_dir, batch_filename)
            with open(batch_path, "w") as f:
                json.dump(batch_metadata, f, indent=4)

                batch_metadata = []
                current_batch += 1

        if batch_metadata:
            batch_filename = f"metadata_batch_{current_batch}.json"
            batch_path = os.path.join(faces_dir, batch_filename)
            with open(batch_path, "w") as f:
                json.dump(batch_metadata, f, indent=4)


    if frameList.size > 0:
        print(f"Face detection rating = {round((totalFaces / np.size(frameList)), 3)}")
    print(f"Processed frames saved in directory: {faces_dir}")


def main():
    setup()
    videoFrames()
    detectFaces()
    cleanUp()


if __name__ == "__main__":
    video_path = r"../../videos/Sitting and smiling robbery.mp4"
    main()

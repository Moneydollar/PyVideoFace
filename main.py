import glob
import json
import os
from linecache import cache

import cv2 as cv
import face_recognition
import numpy as np
from cv2 import VideoCapture


VIDEO_PATH = r"videos/people.mp4"
FRAMES_OUTPUT_DIR = "./.cache/output" + VIDEO_PATH.removeprefix("videos")
FRAMES_WITH_BOXES_DIR = "./.cache/boxed" + VIDEO_PATH.removeprefix("videos")
FACES_DIR = "./faces" + VIDEO_PATH.removeprefix("videos")


print(FACES_DIR)
totalFaces = 0
def setup():
    os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAMES_WITH_BOXES_DIR, exist_ok=True)
    save = False

def cleanUp():

    files = glob.glob(os.path.join(FRAMES_OUTPUT_DIR, "*"))

    for f in files:
        os.remove(f)

    boxFiles = glob.glob(os.path.join(FRAMES_WITH_BOXES_DIR, "*"))

    for f in boxFiles:
        os.remove(f)



def videoFrames():
    vidcap: VideoCapture = cv.VideoCapture(VIDEO_PATH)
    if not vidcap.isOpened():
        print(f"Error opening video file: {VIDEO_PATH}")
        return

    success, image = vidcap.read()
    count = 0

    while success:
        frame_path = os.path.join(FRAMES_OUTPUT_DIR, f"frame{count}.jpg")
        cv.imwrite(frame_path, image)
        success, image = vidcap.read()
        print(f"Read a new frame: {success}")
        count += 1

    vidcap.release()


def detectFaces():
    global totalFaces
    frameList = np.sort(os.listdir(FRAMES_OUTPUT_DIR), kind="heapsort")
    os.makedirs(FRAMES_WITH_BOXES_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    batch_size = 25
    current_batch = 0
    batch_metadata = []

    for frame_file in frameList:
        frame_path = os.path.join(FRAMES_OUTPUT_DIR, frame_file)
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

            metadata = {
                "bounding_box": {
                    "frame": frame_file,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }
            }
            batch_metadata.append(metadata)

            if len(batch_metadata) == batch_size:
                batch_filename = f"metadata_batch_{current_batch}.json"
                batch_path = os.path.join(FACES_DIR, batch_filename)
                with open(batch_path, "w") as f:
                    json.dump(batch_metadata, f, indent=4)

                    batch_metadata = []
                    current_batch += 1

            if batch_metadata:
                batch_filename = f"metadata_batch_{current_batch}.json"
                batch_path = os.path.join(FACES_DIR, batch_filename)
                with open(batch_path, "w") as f:
                    json.dump(batch_metadata, f, indent=4)
            if save:
                output_path = os.path.join(FACES_DIR, frame_file)
                cv.imwrite(output_path, face_image)

    if frameList.size > 0:
        print(f"Face detection rating = {round((totalFaces / np.size(frameList)), 3)}")
    print(f"Processed frames saved in directory: {FACES_DIR}")


def main():
    setup()
    videoFrames()
    detectFaces()
    cleanUp()


if __name__ == "__main__":
    main()

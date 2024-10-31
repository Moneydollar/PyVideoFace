import cv2 as cv
import os
import numpy as np
import glob
import face_recognition

VIDEO_PATH = r"videos/woman_test.mp4"
FRAMES_OUTPUT_DIR = "./frames_output"
FRAMES_WITH_BOXES_DIR = "./frames_with_boxes"

totalFaces = 0

def setup():
    os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(FRAMES_OUTPUT_DIR, '*'))
    
    for f in files:
        os.remove(f)

    boxFiles = glob.glob(os.path.join(FRAMES_WITH_BOXES_DIR, '*'))

    for f in boxFiles:
        os.remove(f)

def videoFrames():
    vidcap = cv.VideoCapture(VIDEO_PATH)
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
    frameList = np.sort(os.listdir(FRAMES_OUTPUT_DIR))
    os.makedirs(FRAMES_WITH_BOXES_DIR, exist_ok=True)

    for frame_file in frameList:
        frame_path = os.path.join(FRAMES_OUTPUT_DIR, frame_file)
        frame = face_recognition.load_image_file(frame_path)
        
        if frame is None:
            print(f"Error loading image {frame_path}. Skipping.")
            continue

        face_locations = face_recognition.face_locations(frame)
        print(f"Detected {len(face_locations)} faces in {frame_file}")
        totalFaces += len(face_locations)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        for top, right, bottom, left in face_locations:
            cv.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        output_path = os.path.join(FRAMES_WITH_BOXES_DIR, frame_file)
        cv.imwrite(output_path, frame)

    if frameList.size > 0:
        print(f"Face detection rating = {round((totalFaces / np.size(frameList)), 3)}")
    print(f"Processed frames saved in directory: {FRAMES_WITH_BOXES_DIR}")

def main():
    setup()
    videoFrames()
    detectFaces()

if __name__ == "__main__":
    main()

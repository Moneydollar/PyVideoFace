import cv2 as cv
import os
import numpy as np
import glob

VIDEO_PATH = r"videos/Sitting and smiling robbery.mp4"
CASCADE_PATH = r"/home/cashc/Documents/Projects/PyVideoFace/cascades/haarcascade_frontalcatface_extended.xml"
FRAMES_OUTPUT_DIR = "./frames_output"
FRAMES_WITH_BOXES_DIR = "./frames_with_boxes"

totalFaces = 0 
def setup():
    os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(FRAMES_OUTPUT_DIR, '*'))
    for f in files:
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
        cv.imwrite(frame_path, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print(f"Read a new frame: {success}")
        count += 1

    vidcap.release()

def detectFaces():
    global totalFaces
    face_cascade = cv.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error loading cascade file: {CASCADE_PATH}")
        return

    frameList = np.sort(os.listdir(FRAMES_OUTPUT_DIR))
    os.makedirs(FRAMES_WITH_BOXES_DIR, exist_ok=True)

    for frame_file in frameList:
        frame_path = os.path.join(FRAMES_OUTPUT_DIR, frame_file)
        frame = cv.imread(frame_path)
        if frame is None:
            print(f"Error loading image {frame_path}. Skipping.")
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        print(f"Detected {len(faces)} faces in {frame_file}")
        totalFaces += len(faces)
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        output_path = os.path.join(FRAMES_WITH_BOXES_DIR, frame_file)
        cv.imwrite(output_path, frame)
    print(f"Face detection rating = {round((totalFaces/np.size(frameList)), 3)}")
    print(f"Processed frames saved in directory: {FRAMES_WITH_BOXES_DIR}")

def main():
    setup()
    videoFrames()
    detectFaces()

if __name__ == "__main__":
    main()
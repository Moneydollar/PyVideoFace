import cv2 as cv
import os
import numpy as np

def setup():
    os.makedirs(r"./frames_output", exist_ok= True)

def videoFrames():
    vidcap = cv.VideoCapture('/home/cashc/Documents/Projects/PyVideoFace/videos/woman_test.mp4')
    success,image = vidcap.read()
    count = 0

    while success:
        cv.imwrite("./frames_output/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def detectFaces():
    # Load multiple Haar cascades
    cascades = [
        '/home/cashc/Documents/Projects/PyVideoFace/cascades/haarcascade_frontalcatface_extended.xml',
    ]

    
    frameList = np.sort(os.listdir("./frames_output"))
    output_dir = "./frames_with_boxes"
    
    
    os.makedirs(output_dir, exist_ok=True)
    
   
    for frame_file in frameList:
        frame_path = os.path.join("./frames_output", frame_file)
        
        
        frame = cv.imread(frame_path)
        if frame is None:
            print(f"Error loading image {frame_path}. Skipping.")
            continue
        
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        all_faces = []  
        
       
        for cascade_path in cascades:
            face_cascade = cv.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            all_faces.extend(faces)  # Combine results
            
        # Remove duplicates (optional, if multiple cascades detect the same face)
        all_faces = np.unique(all_faces, axis=0).tolist()
        
        
        for (x, y, w, h) in all_faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
        output_path = os.path.join(output_dir, frame_file)
        cv.imwrite(output_path, frame)
    
    print(f"Processed frames saved in directory: {output_dir}")

setup()

videoFrames()

detectFaces()

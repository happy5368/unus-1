import cv2
import numpy as np

class libDeeplearning(object):
    def preprocess_video(video_path):                          # preprocess the video
        cap = cv2.VideoCapture(video_path)                     # load video
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # get total number of frames
        processed_frames = []                                  # create an empty array to store preprocessed 

        for i in range(num_frames):                          # loop through each frame
            ret, frame = cap.read()                          # read the next frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # convert to grayscale
            resized = cv2.resize(gray, (120, 120))           # resize frame to (120, 120)
            normalized = resized / 120.0                     # normalize values to range between 0 and 1
            processed_frames.append(normalized)              # add to list of processed frames

        
        processed_frames = np.array(processed_frames)                     # convert processed_frames to numpy array
        processed_frames = np.expand_dims(processed_frames, axis = -1)    # add an extra dimension for time (number of frames)
        return processed_frames                                           # return processed frames
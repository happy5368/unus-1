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

    def inverse_perspective_mapping(img, src, dst):
        # Define source and destination points for IPM
        src = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
        dst = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    def inverse_perspective_mapping(frame):
        # Corners of Top Left , Bottom Left, Top Right, Bottom Right
        tl = (790, 560)
        bl = (400, 1080)
        tr = (1520, 560)
        br = (1920, 1080)

        # Corner 포인트들 파란 점으로 표시
        # cv2.circle(frame, tl, 5, (255, 0, 0), -1)
        # cv2.circle(frame, bl, 5, (255, 0, 0), -1)
        # cv2.circle(frame, tr, 5, (255, 0, 0), -1)
        # cv2.circle(frame, br, 5, (255, 0, 0), -1)

        # Apply Geometrical Transformation
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 1080], [1920, 0], [1920, 1080]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # print("Bird-eye-view Matrix: \n", matrix)
        t_frame = cv2.warpPerspective(frame, matrix, (1920, 1080))

        return t_frame
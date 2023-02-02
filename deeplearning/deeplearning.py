import cv2
import numpy as np
import tensorflow as tf

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


def inverse_perspective_mapping(img, src_points, dst_points): # inverse perspective mapping
    src_points = np.float32() # fix me 
    dst_points = np.float32() # fix me

    M = cv2.getPerspectiveTransform(src_points, dst_points)                                       # get the transform matrix from source to destination points
    Minv = np.linalg.inv(M)                                                                       # invert the transform matrix to get the inverse perspective mapping
    inverse_perspective_mapping = cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]))    # apply the inverse perspective mapping to the image
    return inverse_perspective_mapping

video = preprocess_video("./test_videos/challenge.mp4") # video는 전처리가 완료 되었음
model = tf.keras.models.load_model(video) # load the pre-trained CNN model
real_video = cv2.VideoCapture("./test_videos/1.mp4")


while True:
    ret, frame = real_video.read()
    if not ret:
        break
    # preprocess the frame
    predictions = model.predict(video)
    cv2.imshow("processed video", predictions)


video.release()
cv2.destroyAllWindows()
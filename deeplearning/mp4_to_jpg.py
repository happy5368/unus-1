import cv2

video = cv2.VideoCapture("./test_videos/challenge.mp4")  # fix me # Load the video
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))   # Get the number of frames in the video
frame_number = 0                                         # Set the starting frame number


while True:                                              # Iterate over the frames in the video
    ret, frame = video.read()                            # Read the next frame from the video
    if not ret:                                          # Break the loop if we have reached the end of the video
        break

    cv2.imwrite("frame_{}.jpg".format(frame_number), frame)    # Save the current frame as an image
    frame_number += 1                                          # Increment the frame number

video.release()                                                # Release the video

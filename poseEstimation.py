import cv2
import mediapipe as mp
import os
from utils import checkFileType, readVideo
from termcolor import colored


def main():
    filename = "./assets/IpVsWan0.mp4"
    filename = os.path.normpath(filename)
    write = True

    fileType = checkFileType(filename)

    if fileType == "other":
        print(colored("Unsupported file format", "red"))
        return

    # get the hands recognition object
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    frameWidth = None
    frameHeight = None

    frames = []

    if fileType == "image":
        img = cv2.imread(filename)
        frames.append(img)
    else:
        (frames, (frameWidth, frameHeight)) = readVideo(filename, True)

    print(colored(f"Finish reading {fileType}", "green"))

    for frame in frames:
        currResult = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = currResult.pose_landmarks
        if landmarks:
            mpDraw.draw_landmarks(frame, landmarks, mpPose.POSE_CONNECTIONS)
            for ind, landmark in enumerate(landmarks.landmark):
                h, w, _c = frame.shape()
                yPosition, xPosition = int(landmark.x * w), int(landmark.y * h)

    print(colored("Finish processing pose estimation", "green"))

    if not write:
        if fileType == "video":
            for frame in frames:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
        else:
            cv2.imshow("Frame", frames[0])
            cv2.waitKey(0)
        return

    outputFilename = f"out/pose/{filename.split(os.sep)[-1]}"

    if fileType == "video":
        outputVideo = cv2.VideoWriter(outputFilename, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frameWidth, frameHeight))
        for frame in frames:
            outputVideo.write(frame)
        outputVideo.release()

    else:
        if write:
            cv2.imwrite(outputFilename, frames[0])

    print(colored(f"Output written to {outputFilename}", "green"))


if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
from src.modules.utils import outputWrite, readVideo, checkFileType
from termcolor import colored
from src.modules.poseEstimation import PoseDetector

CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480


def main():
    filename = "./assets/maLong1.jpg"
    filename = os.path.normpath(filename)
    write = True

    fileType = checkFileType(filename)

    if fileType == "other":
        print(colored("Unsupported file format", "red"))
        return

    frameWidth = None
    frameHeight = None

    frames = []
    fps = None
    audio = None

    if fileType == "image":
        img = cv2.imread(filename)
        frames.append(img)
    else:
        ((frames, audio), (frameWidth, frameHeight, fps)) = readVideo(filename, True, True)

    print(colored(f"Finish reading {fileType}", "green"))

    detector = PoseDetector()
    (img, landmarks) = detector.findPose(img)
    detector.findLandmarksPositions(img, landmarks)

    print(colored("Finish processing face detection", "green"))

    if not write:
        for frame in frames:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
        return

    outputFilename = outputWrite(frames, filename, fileType, "AITrainer", fps, (frameWidth, frameHeight), audio)

    print(colored(f"Output written to {outputFilename}", "green"))


if __name__ == "__main__":
    main()

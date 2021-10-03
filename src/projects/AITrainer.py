import os
import cv2
import numpy as np
from src.modules.utils import outputWrite, readVideo, checkFileType
from termcolor import colored
from src.modules.poseEstimation import PoseDetector

CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480


def main():
    filename = "./assets/weightLifting0.mp4"
    filename = os.path.normpath(filename)
    write = True
    curlsCount = 0
    direction = 0  # 0 == up and 1 == down

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
    allPosesInFrames = detector.findPoseInFrames(frames, False)
    frames = []
    for frame, poses in allPosesInFrames:
        for pose in poses:
            # # right arm
            # (frame, angle) = detector.findAndComputeAngle(frame, pose, 12, 14, 16)

            # left arm
            (frame, angle) = detector.findAndComputeAngle(frame, pose, 11, 13, 15)
            percentage = np.interp(angle, (210, 310), (0, 100))

            # check for a curl
            if percentage == 100:
                if direction == 0:
                    # if arm reaches 100% upwards
                    curlsCount += 0.5
                    direction = 1
            elif percentage == 0:
                if direction == 1:
                    # if arm reaches 0% downwards
                    curlsCount += 0.5
                    direction = 0

            cv2.putText(frame, f"Count: {curlsCount}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        frames.append(frame)

    print(colored("Finish processing pose detection", "green"))

    if not write:
        for frame in frames:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
        return

    outputFilename = outputWrite(frames, filename, fileType, "AITrainer", fps, (frameWidth, frameHeight), audio)

    print(colored(f"Output written to {outputFilename}", "green"))


if __name__ == "__main__":
    main()

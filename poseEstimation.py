import cv2
import mediapipe as mp
import os
from utils import checkFileType, readVideo
from termcolor import colored


class PoseDetector:
    def __init__(
        self,
        staticImageMode=False,
        modelComplexity=1,
        smoothLandmarks=True,
        enableSegmentation=False,
        smoothSegmentation=True,
        minDetectionConfidence=0.5,
        minTrackingConfidence=0.5,
    ):
        self.staticImageMode = staticImageMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        # get the hands recognition object
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            staticImageMode,
            modelComplexity,
            smoothLandmarks,
            enableSegmentation,
            smoothSegmentation,
            minDetectionConfidence,
            minTrackingConfidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        currResult = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        landmarks = currResult.pose_landmarks
        if landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, landmarks, self.mpPose.POSE_CONNECTIONS)
            # for ind, landmark in enumerate(landmarks.landmark):
            #     h, w, _c = img.shape()
            #     yPosition, xPosition = int(landmark.x * w), int(landmark.y * h)
        return img

    def findPoseInFrames(self, frames: list, draw=True):
        newFrames = []
        for frame in frames:
            newFrames.append(self.findPose(frame, draw))
        return newFrames


def main():
    filename = "./assets/chain punch.mp4"
    filename = os.path.normpath(filename)
    write = True

    fileType = checkFileType(filename)

    if fileType == "other":
        print(colored("Unsupported file format", "red"))
        return

    frameWidth = None
    frameHeight = None

    frames = []

    if fileType == "image":
        img = cv2.imread(filename)
        frames.append(img)
    else:
        (frames, (frameWidth, frameHeight)) = readVideo(filename, True)

    print(colored(f"Finish reading {fileType}", "green"))

    detector = PoseDetector()
    frames = detector.findPoseInFrames(frames, True)

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

import cv2
import mediapipe as mp
import os
from src.modules.utils import checkFileType, readVideo
from termcolor import colored
import ffmpeg


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
            return (img, landmarks.landmark)
        return (img, [])

    def findPoseInFrames(self, frames: list, draw=True):
        """
        Returns a list of tuples where `list[i] = (frame, landmarks)`
        """
        res = []
        for frame in frames:
            res.append(self.findPose(frame, draw))
        return res

    def findLandmarksPositions(self, img, landmarks, draw=True):
        """
        Returns a list of tuples where `list[i] = (landmarkId, xPosition, yPosition)`
        """
        positions = []
        if landmarks:
            for ind, landmark in enumerate(landmarks):
                h, w, _c = img.shape
                # now the x and y position are in pixels rather than ratios
                xPosition, yPosition = int(landmark.x * w), int(landmark.y * h)
                positions.append([ind, xPosition, yPosition])
                if draw:
                    cv2.circle(img, (xPosition, yPosition), 5, (255, 0, 0), cv2.FILLED)
        return positions


def main():
    filename = "./assets/IpVsWan0.mp4"
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
        ((frames, audio), (frameWidth, frameHeight)) = readVideo(filename, True, True)

    print(colored(f"Finish reading {fileType}", "green"))

    detector = PoseDetector()
    poses = detector.findPoseInFrames(frames, True)
    for pose in poses:
        frame, landmarks = pose
        landmarksPositions = detector.findLandmarksPositions(frame, landmarks)
        print(landmarksPositions if 1 < 1 else "")

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

    outputFilename = f"out/pose/buffer-{filename.split(os.sep)[-1]}"

    if fileType == "video":
        outputVideo = cv2.VideoWriter(outputFilename, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frameWidth, frameHeight))
        for frame in frames:
            outputVideo.write(frame)
        outputVideo.release()

        processedFfmpegVideo = ffmpeg.input(outputFilename)

        ffmpeg.concat(processedFfmpegVideo, audio, v=1, a=1).output(outputFilename.replace("buffer-", "", 1)).run()
        os.remove(outputFilename)

    else:
        if write:
            cv2.imwrite(outputFilename, frames[0])

    print(colored(f"Output written to {outputFilename}", "green"))


if __name__ == "__main__":
    main()

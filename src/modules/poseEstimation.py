import cv2
import mediapipe as mp
import os
from src.modules.utils import checkFileType, readVideo
from termcolor import colored
import ffmpeg
from src.modules.handTracking import EMPHASIS_COLOR


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
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.drawingSpecLine = self.mpDraw.DrawingSpec((20, 255, 0), 7)
        self.drawingSpecLandmark = self.mpDraw.DrawingSpec((20, 20, 255), 3, 15)

    def findPose(self, img, draw=True):
        """
        Returns a tuples structured as `(img, poses)`
        And "poses" is:
        ```python
        list[list[tuple[landmarkId: int, x: int, y: int]]]
        ```
        """
        currResult = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        landmarks = currResult.pose_landmarks
        poses = []
        if landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    self.mpDrawingStyles.get_default_pose_landmarks_style(),
                    self.drawingSpecLine,
                )
            pose = []
            for ind, landmark in enumerate(landmarks.landmark):
                imageH, imageW, _ = img.shape
                x, y = int(landmark.x * imageW), int(landmark.y * imageH)
                pose.append((ind, x, y))
            poses.append(pose)

        return (img, poses)

    def findPoseInFrames(self, frames: list, draw=True):
        """
        Returns a list of tuples where `list[i] = (frame, poses)`
        And "poses" is:
        ```python
        list[list[tuple[landmarkId: int, x: int, y: int]]]
        ```
        """
        res = []
        for frame in frames:
            res.append(self.findPose(frame, draw))
        return res

    def highlightLandmark(self, img, poses, landmarkId, circleRadius=12):
        """Given a list of poses, highlight the landmark where `landmark.id = landmarkId` across all poses.

        Args:
            img: cv2 img
            poses (List[List[Tuple[int, int]]]): A list of hands
            landmarkId ([type]): [description]
        """
        x, y = None, None

        for pose in poses:
            for ind, currX, currY in pose:
                if ind == landmarkId:
                    x, y = currX, currY
                    break

        if x is None or y is None:
            raise ValueError("Invalid landmark id")

        cv2.circle(img, (x, y), circleRadius, EMPHASIS_COLOR, cv2.FILLED)

        return img

    def findAndComputeAngle(self, img, pose: list, landmarkAId, landmarkBId, landmarkCId, draw=True):
        """Computes and returns the value of ∠ABC

        Args:
            landmarkA: A
            landmarkB: B
            landmarkC: C
            draw (bool, optional): [description]. Defaults to True.
        """
        # trigonometry
        x1, y1 = pose[landmarkAId][1:]
        x2, y2 = pose[landmarkBId][1:]
        x3, y3 = pose[landmarkCId][1:]

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 5)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

        return img


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

    allPosesInFrames = detector.findPoseInFrames(frames, True)
    frames = []
    for frame, _ in allPosesInFrames:
        frames.append(frame)

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

import cv2
import mediapipe as mp
import os
from utils import checkFileType, readVideo
from termcolor import colored
import ffmpeg


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    def findFace(self, img, draw=True):
        currResult = self.face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detections = currResult.detections
        if detections:
            if draw:
                for detection in detections:
                    self.mpDraw.draw_detection(img, detection)
            return (img, detections)
        return (img, [])

    def findFaceInFrames(self, frames: list, draw=True):
        """
        Returns a list of tuples where `list[i] = (frame, landmarks)`
        """
        res = []
        for frame in frames:
            res.append(self.findFace(frame, draw))
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
    filename = "./assets/oldmanThumbsUp.jpg"
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

    detector = FaceDetector()
    faces = detector.findFaceInFrames(frames, True)
    # for face in faces:
    # frame, landmarks = face
    # landmarksPositions = detector.findLandmarksPositions(frame, landmarks)
    # print(landmarksPositions if 1 > 4 else "")

    print(colored("Finish processing face detection", "green"))

    if not write:
        if fileType == "video":
            for frame in frames:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
        else:
            cv2.imshow("Frame", frames[0])
            cv2.waitKey(0)
        return

    outputFilename = f"out/face/buffer-{filename.split(os.sep)[-1]}"

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

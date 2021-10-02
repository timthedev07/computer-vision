import os
import cv2
import mediapipe as mp
from src.modules.utils import checkFileType, readVideo
from termcolor import colored
import ffmpeg

ANNOTATION_COLOR = (157, 155, 24)


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(min_detection_confidence, model_selection)
        self.mpDraw = mp.solutions.drawing_utils

    def findFace(self, img, draw=True):
        currResult = self.face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detections = currResult.detections
        boundingBoxes = []
        if detections:
            for ind, detection in enumerate(detections):
                bboxC = detection.location_data.relative_bounding_box
                imageH, imageW, _trash = img.shape
                boundingBox = (
                    int(bboxC.xmin * imageW),
                    int(bboxC.ymin * imageH),
                    int(bboxC.width * imageW),
                    int(bboxC.height * imageH),
                )

                # a tuple containing (id, boundingBox, detectionScore)
                boundingBoxes.append((ind, boundingBox, detection.score))
                if draw:
                    img = self.customDraw(img, boundingBox)
                    cv2.putText(
                        img,
                        f"{int(detection.score[0] * 100)}%",
                        (boundingBox[0], boundingBox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        ANNOTATION_COLOR,
                        2,
                    )
        return (img, boundingBoxes)

    def findFaceInFrames(self, frames: list, draw=True):
        """
        Returns a list of tuples where `list[i] = (frame, landmarks)`
        """
        res = []
        for frame in frames:
            res.append(self.findFace(frame, draw))
        return res

    def customDraw(self, img, boundingBox, cornerMarkerLength=30, cornerMarkerThickness=10, rectangleThickness=1):
        xStart, yStart, w, h = boundingBox
        xEnd, yEnd = xStart + w, yStart + h

        cv2.rectangle(img, boundingBox, ANNOTATION_COLOR, rectangleThickness)
        # top left corner
        cv2.line(img, (xStart, yStart), (xStart + cornerMarkerLength, yStart), ANNOTATION_COLOR, cornerMarkerThickness)
        cv2.line(img, (xStart, yStart), (xStart, yStart + cornerMarkerLength), ANNOTATION_COLOR, cornerMarkerThickness)

        # top right corner
        cv2.line(img, (xEnd, yStart), (xEnd - cornerMarkerLength, yStart), ANNOTATION_COLOR, cornerMarkerThickness)
        cv2.line(img, (xEnd, yStart), (xEnd, yStart + cornerMarkerLength), ANNOTATION_COLOR, cornerMarkerThickness)

        # bottom right corner
        cv2.line(img, (xEnd, yEnd), (xEnd - cornerMarkerLength, yEnd), ANNOTATION_COLOR, cornerMarkerThickness)
        cv2.line(img, (xEnd, yEnd), (xEnd, yEnd - cornerMarkerLength), ANNOTATION_COLOR, cornerMarkerThickness)

        # bottom left corner
        cv2.line(img, (xStart, yEnd), (xStart + cornerMarkerLength, yEnd), ANNOTATION_COLOR, cornerMarkerThickness)
        cv2.line(img, (xStart, yEnd), (xStart, yEnd - cornerMarkerLength), ANNOTATION_COLOR, cornerMarkerThickness)

        return img


def main():
    filename = "./assets/IP MAN 0.mp4"
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
        ((frames, audio), (frameWidth, frameHeight, fps)) = readVideo(filename, True, True)

    print(colored(f"Finish reading {fileType}", "green"))

    detector = FaceDetector()
    detector.findFaceInFrames(frames, True)

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

    if fileType == "video":
        outputFilename = f"out/face/{filename.split(os.sep)[-1]}"
        bufferOutputFilename = f"out/face/buffer-{filename.split(os.sep)[-1]}"
        outputVideo = cv2.VideoWriter(
            bufferOutputFilename, cv2.VideoWriter_fourcc(*"MP4V"), fps, (frameWidth, frameHeight)
        )
        for frame in frames:
            outputVideo.write(frame)
            cv2.waitKey(1)
        outputVideo.release()

        processedFfmpegVideo = ffmpeg.input(bufferOutputFilename)

        ffmpeg.concat(processedFfmpegVideo, audio, v=1, a=1).output(outputFilename, loglevel="quiet").run()
        os.remove(bufferOutputFilename)

    else:
        outputFilename = f"out/face/{filename.split(os.sep)[-1]}"
        if write:
            cv2.imwrite(outputFilename, frames[0])
    outputFilename = outputFilename.replace("buffer-", "")

    print(colored(f"Output written to {outputFilename}", "green"))


if __name__ == "__main__":
    main()

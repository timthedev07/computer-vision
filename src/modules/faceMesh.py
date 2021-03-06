import os
import cv2
import mediapipe as mp
from src.modules.utils import checkFileType, readVideo
from termcolor import colored
import ffmpeg

ANNOTATION_COLOR = (26, 246, 0)


class FaceMeshDetector:
    def __init__(self, staticImageMode=False, maxNumFaces=3, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.mpFaceMesh = mp.solutions.face_mesh
        self.face = self.mpFaceMesh.FaceMesh(
            staticImageMode, maxNumFaces, minDetectionConfidence, minTrackingConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=ANNOTATION_COLOR)

    def findFaceMesh(self, img, draw=True):
        """
        Returns tuple consisting of (frame, landmarks)
        """
        currResult = self.face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        multiFacelandmarks = currResult.multi_face_landmarks
        faces = []
        if multiFacelandmarks:
            for faceLandmarks in multiFacelandmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec)
                face = []
                for ind, landmark in enumerate(faceLandmarks.landmark):
                    imageH, imageW, _ = img.shape
                    x, y = int(landmark.x * imageW), int(landmark.y * imageH)
                    face.append((ind, x, y))
                faces.append(face)
        return (img, faces)

    def findFaceMeshInFrames(self, frames: list, draw=True):
        """
        Returns a list of tuples where `list[i] = (frame, landmarks)`
        """
        res = []
        for frame in frames:
            res.append(self.findFaceMesh(frame, draw))
        return res


def main():
    filename = "./assets/techPeople0.jpg"
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

    detector = FaceMeshDetector(maxNumFaces=10, minDetectionConfidence=0.3)
    detector.findFaceMeshInFrames(frames, True)

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
        outputFilename = f"out/faceMesh/{filename.split(os.sep)[-1]}"
        bufferOutputFilename = f"out/faceMesh/buffer-{filename.split(os.sep)[-1]}"
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
        outputFilename = f"out/faceMesh/{filename.split(os.sep)[-1]}"
        if write:
            cv2.imwrite(outputFilename, frames[0])
    outputFilename = outputFilename.replace("buffer-", "")

    print(colored(f"Output written to {outputFilename}", "green"))


if __name__ == "__main__":
    main()

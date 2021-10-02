import os
import cv2
import mediapipe as mp
from utils import checkFileType, readVideo
from termcolor import colored
import ffmpeg

# False, 4, 0.65


class HandDetector:
    def __init__(self, staticImageMode=False, maxNumHands=2, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        # get the hands recognition object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(staticImageMode, maxNumHands, minDetectionConfidence, minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        currResult = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        allHandLandmarks = currResult.multi_hand_landmarks
        hands = []
        if allHandLandmarks:
            for handLandmarks in allHandLandmarks:
                hand = []
                for ind, landmark in enumerate(handLandmarks.landmark):
                    imageH, imageW, _ = img.shape
                    x, y = int(landmark.x * imageW), int(landmark.y * imageH)
                    hand.append((ind, x, y))
                hands.append(hand)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return (img, hands)

    def findHandsInFrames(self, frames: list, draw=True):
        """
        Returns a list of tuples where `list[i] = (frame, landmarks)`
        """
        res = []
        for frame in frames:
            res.append(self.findHands(frame, draw))
        return res


def main():
    """Main function"""
    filename = "./assets/hand0.jpg"
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

    detector = HandDetector()
    detector.findHandsInFrames(frames)

    print(colored("Finish processing hand detection", "green"))

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
        outputFilename = f"out/hands/{filename.split(os.sep)[-1]}"
        bufferOutputFilename = f"out/hands/buffer-{filename.split(os.sep)[-1]}"
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
        outputFilename = f"out/hands/{filename.split(os.sep)[-1]}"
        if write:
            cv2.imwrite(outputFilename, frames[0])
    outputFilename = outputFilename.replace("buffer-", "")

    print(colored(f"Output written to {outputFilename}", "green"))


if __name__ == "__main__":
    main()

import os
import cv2
import mediapipe as mp
from src.modules.utils import checkFileType, readVideo
from termcolor import colored

if __name__ == "__main__":
    import ffmpeg

ANNOTATION_COLOR = (26, 246, 0)
EMPHASIS_COLOR = (26, 0, 246)


class HandDetector:
    def __init__(self, staticImageMode=False, maxNumHands=2, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        # get the hands recognition object
        self.maxNumHands = maxNumHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(staticImageMode, maxNumHands, minDetectionConfidence, minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawingSpec = self.mpDraw.DrawingSpec(ANNOTATION_COLOR)
        self.mpDrawingStyles = mp.solutions.drawing_styles

    def findHands(self, img, draw=True):
        currResult = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        allHandLandmarks = currResult.multi_hand_landmarks
        hands = []
        if allHandLandmarks:
            counter = 0
            for handLandmarks in allHandLandmarks:
                if counter == self.maxNumHands:
                    break
                hand = []
                for ind, landmark in enumerate(handLandmarks.landmark):
                    imageH, imageW, _ = img.shape
                    x, y = int(landmark.x * imageW), int(landmark.y * imageH)
                    hand.append((ind, x, y))
                hands.append(hand)

                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLandmarks,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDrawingStyles.get_default_hand_landmarks_style(),
                        self.drawingSpec,
                    )
                counter += 1
        return (img, hands)

    def findHandsInFrames(self, frames: list, draw=True):
        """
        Returns a list of tuples where `list[i] = (frame, landmarks)`
        """
        res = []
        for frame in frames:
            res.append(self.findHands(frame, draw))
        return res

    def highlightLandmark(self, img, hands, landmarkId, circleRadius=12):
        """Given a list of hands, highlight the landmark where `landmark.id = landmarkId` across all hands.

        Args:
            img: cv2 img
            hands (List[List[Tuple[int, int]]]): A list of hands
            landmarkId ([type]): [description]
        """
        x, y = None, None

        for hand in hands:
            for ind, currX, currY in hand:
                if ind == landmarkId:
                    x, y = currX, currY
                    break

        if x is None or y is None:
            raise ValueError("Invalid landmark id")

        cv2.circle(img, (x, y), circleRadius, EMPHASIS_COLOR, cv2.FILLED)

        return img

    def connectLandmarks(self, img, hands, landmarkAId, landmarkBId, thickness=4):
        """Given a list of hands, highlight the landmark where `landmark.id = landmarkId` across all hands.

        Args:
            img: cv2 img
            hands (List[List[Tuple[int, int]]]): A list of hands
            landmarkId ([type]): [description]
        """
        x1, y1 = None, None
        x2, y2 = None, None

        for hand in hands:
            for ind, currX, currY in hand:
                if ind == landmarkAId:
                    x1, y1 = currX, currY
                elif ind == landmarkBId:
                    x2, y2 = currX, currY

        if x1 is None or y1 is None:
            raise ValueError("Id of landmark A is invalid")
        if x2 is None or y2 is None:
            raise ValueError("Id of landmark B is invalid")

        cv2.line(img, (x1, y1), (x2, y2), EMPHASIS_COLOR, thickness)

        return img


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
    framesWithHands = detector.findHandsInFrames(frames)
    frames = []
    for frameWithHands in framesWithHands:
        (frame, hands) = frameWithHands
        detector.highlightLandmark(frame, hands, 4)
        detector.highlightLandmark(frame, hands, 8)
        frames.append(frame)

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

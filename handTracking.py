import cv2
import mediapipe as mp
from utils import checkFileType
import os


def main():
    """Main function"""
    filename = "./assets/hands/labrador.jpg"
    filename = os.path.normpath(filename)
    write = True

    fileType = checkFileType(filename)

    if fileType == "other":
        print("Unsupported file format")
        return

    # get the hands recognition object
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(False, 4, 0.65)
    mpDraw = mp.solutions.drawing_utils

    frameWidth = None
    frameHeight = None

    frames = []

    if fileType == "image":
        img = cv2.imread(filename)
        frames.append(img)
    else:
        # define capture source
        cap = cv2.VideoCapture(filename)
        frameWidth = int(cap.get(3))
        frameHeight = int(cap.get(4))

        if not cap.isOpened():
            print(f"Failed to open file {filename}")
            return

        while cap.isOpened():
            # read frame
            success, frame = cap.read()

            # if success then add frame to the frames array and show it
            if success is True:
                frames.append(frame)
            else:
                # reached the end
                break
        cap.release()

    print(f"Finish reading {fileType}")

    for frame in frames:
        currResult = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if currResult.multi_hand_landmarks:
            for handLandmark in currResult.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLandmark, mpHands.HAND_CONNECTIONS)

    print("Finish processing hand detection")

    if not write:
        if fileType == "video":
            for frame in frames:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
        else:
            cv2.imshow("Frame", frames[0])
            cv2.waitKey(0)
        return

    outputFilename = f"out/hands/{filename.split(os.sep)[-1]}"

    if fileType == "video":
        outputVideo = cv2.VideoWriter(outputFilename, cv2.VideoWriter_fourcc(*"MP4V"), 30, (frameWidth, frameHeight))
        for frame in frames:
            outputVideo.write(frame)
        outputVideo.release()

    else:
        if write:
            cv2.imwrite(outputFilename, frames[0])

    print(f"Output written to {outputFilename}")


if __name__ == "__main__":
    main()

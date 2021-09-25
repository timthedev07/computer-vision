import cv2
import mediapipe as mp
import time


def main():
    # define capture source
    cap = cv2.VideoCapture("assets/chain punch.mp4")

    # get the hands recognition object
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()

    frames = []

    while cap.isOpened():
        # read frame
        success, frame = cap.read()

        # if success then add frame to the frames array and show it
        if success:
            frames.append(frame)
        else:
            # reached the end
            break

    cap.release()

    for frame in frames:
        print(frame.tolist())

    print("Finish reading video")


if __name__ == "__main__":
    main()

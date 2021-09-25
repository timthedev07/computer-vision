import cv2
import mediapipe as mp
import time


def main():
    filename = "./assets/chain punch.mp4"
    # define capture source
    cap = cv2.VideoCapture(filename)

    # get the hands recognition object
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()

    frames = []

    if not cap.isOpened():
        print(f"Failed to open file {filename}")
        return

    while cap.isOpened():
        # read frame
        success, frame = cap.read()

        # if success then add frame to the frames array and show it
        if success == True:
            frames.append(frame)
        else:
            # reached the end
            break
    cap.release()

    print("Finish reading video")

    for frame in frames:
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import time


def main():
    filename = "./assets/IP MAN 0.mp4"
    # define capture source
    cap = cv2.VideoCapture(filename)

    # get the hands recognition object
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(False, 4, 0.65)
    mpDraw = mp.solutions.drawing_utils

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

    result = []

    for frame in frames:
        currResult = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if currResult.multi_hand_landmarks:
            for handLandmark in currResult.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLandmark, mpHands.HAND_CONNECTIONS)

    print("Finish processing hand detection")

    for frame in frames:
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

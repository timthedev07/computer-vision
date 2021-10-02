import cv2
from src.modules import handTracking as ht
import math

CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)

    detector = ht.HandDetector(minDetectionConfidence=0.7)

    while True:
        success, img = cap.read()

        # do stuff here

        (img, hands) = detector.findHands(img)
        if hands and len(hands) > 0:
            hands = [hands[0]]
            hand = hands[0]

            img = detector.highlightLandmark(
                img,
                hands,
                4,
            )
            img = detector.highlightLandmark(img, hands, 8)
            img = detector.connectLandmarks(img, hands, 4, 8, 3)
            _, x1, y1 = hand[4]
            _, x2, y2 = hand[8]
            midX, midY = (x1 + x2) // 2, (y1 + y2) // 2

            length = math.hypot(x2 - x1, y2 - y1)

            midCircleColor = ht.EMPHASIS_COLOR

            if length <= 40:
                midCircleColor = (255, 0, 0)

            cv2.circle(img, (midX, midY), 12, midCircleColor, cv2.FILLED)

        # ===========

        if success is False:
            break

        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()

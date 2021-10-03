import os
import cv2
from src.modules import handTracking as ht


CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)

    detector = ht.HandDetector(minDetectionConfidence=0.7, maxNumHands=1)

    fingerImages = []
    fingerImagesDirectory = "assets/fingers"
    directoryContent = os.listdir(fingerImagesDirectory)
    directoryContent.sort()
    for imageFilename in directoryContent:
        fingerImage = cv2.imread(f"{fingerImagesDirectory}/{imageFilename}")
        fingerImages.append(fingerImage)

    fingertipIds = [4, 8, 12, 16, 20]
    rightHand = False

    while True:
        success, img = cap.read()

        # do stuff here
        (img, hands) = detector.findHands(img)

        # 1 is up and 0 is down
        fingerStates = []

        # only take the first hand
        if len(hands) > 0:
            hands = [hands[0]]
            hand = hands[0]

            # if the index finger metacarpophalangeal joint
            # is on the right hand side of the pinky MCP joint,
            # it is the right hand
            rightHand = hand[5][1] > hand[17][1]

            # edge case => the thumb
            if hand[4][1] < hand[3][1] if rightHand else hand[4][1] > hand[3][1]:
                fingerStates.append(0)
            else:
                fingerStates.append(1)

            for fingertipId in fingertipIds[1:]:
                if hand[fingertipId][2] < hand[fingertipId - 2][2]:
                    # current finger is up
                    fingerStates.append(1)
                else:
                    fingerStates.append(0)

        totalFingers = fingerStates.count(1)
        fingerImage = fingerImages[totalFingers]
        fingerImageH, fingerImageW, _ = fingerImage.shape
        img[0:fingerImageH, 0:fingerImageW] = fingerImage
        # ===========

        if success is False:
            break

        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()

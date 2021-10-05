import os
import cv2
import numpy as np
import src.modules.handTracking as ht


def main():
    menusDirectory = "assets/virtualPainterMenus"
    menuFilenames = os.listdir(menusDirectory)

    menus = []

    for menuFilename in menuFilenames:
        img = cv2.imread(f"{menusDirectory}/{menuFilename}")
        menus.append(img)

    menu = menus[0]
    menuHeight, menuWidth, _ = menus[0].shape
    detector = ht.HandDetector(minDetectionConfidence=0.85)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(3, 720)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # do stuff here

        # find hand landmarks
        img, hands = detector.findHands(img)
        if len(hands) > 0:
            hand = hands[0]
            hands = [hand]

            # tip of index and middle finger
            x1, y1 = hand[8][1:]
            x2, y2 = hand[12][1:]

        # check which fingers are up

        # if selection mode => two fingers click on an object in the menu

        # if drawing mode => index finger is up

        # placing the menu
        img[0:menuHeight, 0:menuWidth] = menu
        # =============

        if success is False:
            break

        cv2.imshow("Camera", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()

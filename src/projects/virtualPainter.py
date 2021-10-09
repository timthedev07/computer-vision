import os
import cv2
import numpy as np
import src.modules.handTracking as ht


def main():
    menusDirectory = "assets/virtualPainterMenus"
    menuFilenames = os.listdir(menusDirectory)
    menuFilenames.sort()
    penThickness = 15
    prevX, prevY = -1, -1
    CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720

    menus = []

    for menuFilename in menuFilenames:
        img = cv2.imread(f"{menusDirectory}/{menuFilename}")
        menus.append(img)

    menu = menus[0]
    menuHeight, menuWidth, _ = menus[0].shape
    detector = ht.HandDetector(minDetectionConfidence=0.85)

    cap = cv2.VideoCapture(0)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)
    color = (255, 216, 0)

    imgCanvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), np.uint8)

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
            _, fingerStates = detector.getFingersStates(hand)

            # if selection mode => two fingers click on an object in the menu
            if fingerStates[1] == 1 and fingerStates[2] == 1:
                # checking for any clicks on the pens/eraser
                if y1 < 140:
                    if 100 < x1 < 250:
                        menu = menus[0]
                        color = (255, 216, 0)
                    elif 350 < x1 < 500:
                        menu = menus[1]
                        color = (0, 0, 255)
                    elif 700 < x1 < 850:
                        menu = menus[2]
                        color = (0, 255, 0)
                    elif 950 < x1 < 1200:
                        color = (0, 0, 0)
                        menu = menus[3]
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), color, cv2.FILLED)

            # if drawing mode => index finger is up
            elif fingerStates[1] == 1 and fingerStates[2] == 0:
                cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
                if prevX == -1 and prevY == -1:
                    xp, yp = x1, y1
                if color == (0, 0, 0):
                    cv2.line(img, (prevX, prevY), (x1, y1), color, penThickness + 20)
                    cv2.line(imgCanvas, (prevX, prevY), (x1, y1), color, penThickness + 20)
                else:
                    cv2.line(img, (prevX, prevY), (x1, y1), color, penThickness)
                    cv2.line(imgCanvas, (prevX, prevY), (x1, y1), color, penThickness)

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

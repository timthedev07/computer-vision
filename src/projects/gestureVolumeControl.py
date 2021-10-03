import platform
import cv2
from src.modules import handTracking as ht
import math
import numpy as np

system = platform.system()
if system == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
elif system == "Darwin":
    import osascript


CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)
    volumeBar = 400
    volumePercentage = 0

    detector = ht.HandDetector(minDetectionConfidence=0.7)

    if system == "Windows":
        # volume control initialization for windows
        devices = AudioUtilities.GetSpeakers()
        # pylint: disable=protected-access
        volume = cast(devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None), POINTER(IAudioEndpointVolume))
        volumeRange = volume.GetVolumeRange()
        minVolume = volumeRange[0]
        maxVolume = volumeRange[1]

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

            # Finger(index and thumb) distance range: 50 - 300
            length = math.hypot(x2 - x1, y2 - y1)

            if system == "Windows":
                # update volume for windows
                newVolume = np.interp(length, [50, 250], [minVolume, maxVolume])
                volume.SetMasterVolumeLevel(newVolume, None)
            elif system == "Darwin":
                # update volume for mac
                newVolume = int(np.interp(length, [50, 250], [0, 100]))
                volumePercentage = newVolume
                osascript.osascript(f"set volume output volume {newVolume}")

            midCircleColor = ht.EMPHASIS_COLOR

            volumeBar = np.interp(length, [50, 250], [400, 150])
            if system == "Windows":
                volumePercentage = int(np.interp(length, [50, 250], [0, 100]))

            if length <= 40:
                midCircleColor = (255, 0, 0)

            cv2.circle(img, (midX, midY), 12, midCircleColor, cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volumeBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f"{int(volumePercentage)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        # ===========

        if success is False:
            break

        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()

import os
from sys import argv
import cv2
import ffmpeg
import tempfile


def main():
    if len(argv) != 2:
        print("Usage: python flipVideo.py <path>")
    path = argv[1]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frameShape = (int(cap.get(3)), int(cap.get(4)))
    audio = ffmpeg.input(path)
    while True:
        success, frame = cap.read()
        if success is False:
            break

        frame = cv2.flip(frame, 1)
        frames.append(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    os.remove(path)
    cap.release()

    # store the video with no sound in the system temp folder
    bufferOutputFilename = f"{tempfile.gettempdir()}/someBuffer.mp4"
    outputVideo = cv2.VideoWriter(bufferOutputFilename, cv2.VideoWriter_fourcc(*"MP4V"), fps, frameShape)
    for frame in frames:
        outputVideo.write(frame)
        cv2.waitKey(1)
    outputVideo.release()

    # remove the existing video
    os.remove(path)
    processedFfmpegVideo = ffmpeg.input(bufferOutputFilename)
    # , loglevel="quiet"
    ffmpeg.concat(processedFfmpegVideo, audio, v=1, a=1).output(path).run()
    os.remove(bufferOutputFilename)


if __name__ == "__main__":
    main()

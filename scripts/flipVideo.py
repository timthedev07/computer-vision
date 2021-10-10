import os
from sys import argv
import cv2
import ffmpeg
import tempfile
from shutil import copyfile


def main():
    if len(argv) != 2:
        print("Usage: python flipVideo.py <path>")
    path = os.path.normpath(argv[1])
    path = path.replace(os.sep, "/")
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

    cap.release()

    filename = path.split("/")[-1] if "/" in path else path

    # store the video with no sound in the system temp folder
    bufferOutputFilename = f"{tempfile.gettempdir()}/{filename}"
    outputVideo = cv2.VideoWriter(bufferOutputFilename, cv2.VideoWriter_fourcc(*"mp4v"), fps, frameShape)
    for frame in frames:
        outputVideo.write(frame)
        cv2.waitKey(1)
    outputVideo.release()

    processedFfmpegVideo = ffmpeg.input(bufferOutputFilename)
    try:
        ffmpeg.concat(processedFfmpegVideo, audio, v=1, a=1).output(path, loglevel="quiet").run()
        os.remove(bufferOutputFilename)
    # pylint: disable=bare-except
    except:
        os.remove(path)
        copyfile(bufferOutputFilename, path)


if __name__ == "__main__":
    main()

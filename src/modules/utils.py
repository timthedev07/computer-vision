import cv2
import ffmpeg
import os


def checkFileType(filename: str) -> str:

    images = [
        "jpg",
        "png",
        "gif",
        "webp",
        "tiff",
        "psd",
        "raw",
        "bmp",
        "heif",
        "indd",
        "jpeg",
    ]

    for imageFileExtension in images:
        if filename.endswith(imageFileExtension):
            return "image"

    videos = [
        ".webm",
        ".mpg",
        ".mp2",
        ".mpeg",
        ".mpe",
        ".mpv",
        ".ogg",
        ".mp4",
        ".m4p",
        ".m4v",
        ".avi",
        ".wmv",
        ".mov",
        ".qt",
        ".flv",
        ".swf",
    ]

    for videoFileExtension in videos:
        if filename.endswith(videoFileExtension):
            return "video"

    return "other"


def readVideo(path: str, getSize=False, withAudio=False):
    """
    Given a relative path to a video file, read the video and return a list of unprocessed frames.
    to determine the output variable `video` based on the withAudio parameter, if it is true, then the `video` variable would contain a tuple with (a list of frames, a ffmpeg audio)
    and finally `if getSize == true` then its return value would be like `(video, (frameWidth, frameHeight, fps))` and `video` otherwise
    """
    # define capture source
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"Failed to open file {path}")
        return []
    frames = []
    frameWidth = int(cap.get(3))
    frameHeight = int(cap.get(4))
    audio = ffmpeg.input(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        # read frame
        success, frame = cap.read()

        # if success then add frame to the frames array
        if success is True:
            frames.append(frame)
        else:
            # reached the end
            break
    cap.release()

    video = (frames, audio) if withAudio else frames

    return (video, (frameWidth, frameHeight, fps)) if getSize else video


def outputWrite(frames: list, filename: str, fileType: str, fps: int, frameShape: tuple[int, int], audio=None):
    """Handles outputing video/image

    Args:
        frames (list): A list of frames which the output video is going to contain.
        filename (str): File name
        fileType (str): File type
        fps (int): FPS
        frameShape (tuple[int, int]): (frameWidth, frameHeight)
        audio (???, optional): The ffmpeg audio. Defaults to None and thus the video would be "muted" by default.

    Returns:
        str: The output filename
    """
    # if not write:
    #     for frame in frames:
    #         cv2.imshow("Frame", frame)
    #         cv2.waitKey(1)
    #     return

    if fileType == "video":
        outputFilename = f"out/faceMesh/{filename.split(os.sep)[-1]}"
        if not audio:
            outputVideo = cv2.VideoWriter(outputFilename, cv2.VideoWriter_fourcc(*"MP4V"), fps, frameShape)
            outputVideo.release()
        else:
            bufferOutputFilename = f"out/faceMesh/buffer-{filename.split(os.sep)[-1]}"
            outputVideo = cv2.VideoWriter(bufferOutputFilename, cv2.VideoWriter_fourcc(*"MP4V"), fps, frameShape)
            for frame in frames:
                outputVideo.write(frame)
                cv2.waitKey(1)
            outputVideo.release()

            processedFfmpegVideo = ffmpeg.input(bufferOutputFilename)

            ffmpeg.concat(processedFfmpegVideo, audio, v=1, a=1).output(outputFilename, loglevel="quiet").run()
            os.remove(bufferOutputFilename)

    else:
        outputFilename = f"out/faceMesh/{filename.split(os.sep)[-1]}"
        cv2.imwrite(outputFilename, frames[0])
    outputFilename = outputFilename.replace("buffer-", "")

    return outputFilename

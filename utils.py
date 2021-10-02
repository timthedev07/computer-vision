import cv2
import ffmpeg


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

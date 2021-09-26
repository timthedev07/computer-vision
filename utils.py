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
        ".webm,",
        ".mpg,",
        ".mp2,",
        ".mpeg,",
        ".mpe,",
        ".mpv,",
        ".ogg,",
        ".mp4,",
        ".m4p,",
        ".m4v,",
        ".avi,",
        ".wmv,",
        ".mov,",
        ".qt,",
        ".flv",
        ".swf",
    ]

    for videoFileExtension in videos:
        if filename.endswith(videoFileExtension):
            return "video"

    return "other"

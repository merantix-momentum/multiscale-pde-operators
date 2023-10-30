import shutil
from pathlib import Path

import gdown


def download_electromagnetics():
    folder = Path("datasets/electromagnetics")

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        output = "datasets/electromagnetics.zip"
        url = "https://drive.google.com/uc?id=1x8xddySJO9sEnyc_qBUYcDDrSA7z5sV2"

        gdown.download(url, output, quiet=False)
        print("extracting..")
        shutil.unpack_archive(output, folder)

    return folder


def download_darcy():
    folder = Path("datasets/darcy")

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        output = "datasets/darcy.zip"
        url = "https://drive.google.com/uc?id=1OYoXIFhRbY_TJmce0cb0nSs5n9vO8z68"

        gdown.download(url, output, quiet=False)
        print("extracting..")
        shutil.unpack_archive(output, folder)

    return folder


def download_motordata():
    folder = Path("datasets/motor")

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        output = "datasets/motor.zip"
        url = "https://drive.google.com/uc?id=1Af0bJyR3SNEMeak0mXFQC_7Dg97nVmlf"

        gdown.download(url, output, quiet=False)
        print("extracting..")
        shutil.unpack_archive(output, folder)

    return folder


if __name__ == "__main__":
    download_electromagnetics()
    download_darcy()
    download_motordata()

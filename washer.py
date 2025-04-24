# coding=utf-8
from PIL import Image
import os
import yaml


class WasherConfig:

    def __init__(self):
        with open("./config.yml", "r") as f:
            conf = yaml.safe_load(f)
            if conf.get("washer") is None:
                raise FileNotFoundError("No config found for washer")

            conf = conf.get("washer")
            self.location = conf["location"]
            self.target_format = conf["target_format"]
            self.filter_format = tuple(conf["filter_format"])

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


_CONFIG = WasherConfig()


def remove_metadata(img: Image.Image) -> Image.Image:
    """移除图片metadata"""
    data = list(img.getdata())
    image_without_exif = Image.new(img.mode, img.size)
    image_without_exif.putdata(data)
    return image_without_exif


def recursive_search(path: str):
    """递归搜索文件夹里的图片"""
    for file_path in os.listdir(path):
        file_path = os.path.join(path, file_path)

        if os.path.isdir(file_path):
            recursive_search(file_path)

        if os.path.isfile(file_path) and file_path.lower().endswith(
            _CONFIG.filter_format
        ):
            img = Image.open(file_path)
            img = remove_metadata(img)
            img.save(file_path, _CONFIG.target_format)


if __name__ == "__main__":
    recursive_search(os.path.abspath(_CONFIG.location))

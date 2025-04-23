# coding=utf-8
from PIL import Image
import os

FILE_FORMAT = "png"  # 图片格式


def remove_metadata(img: Image.Image) -> Image.Image:
    """移除图片metadata"""
    data = list(img.getdata())
    image_without_exif = Image.new(img.mode, img.size)
    image_without_exif.putdata(data)
    return image_without_exif


def recursive_search(path: str):
    """递归搜索文件夹里的png"""
    for file_path in os.listdir(path):
        file_path = os.path.join(path, file_path)

        if os.path.isdir(file_path):
            recursive_search(file_path)

        if os.path.isfile(file_path) and file_path.lower().endswith(FILE_FORMAT):
            img = Image.open(file_path)
            img = remove_metadata(img)
            img.save(file_path, FILE_FORMAT)


if __name__ == "__main__":
    recursive_search(os.path.abspath("./data"))

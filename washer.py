# coding=utf-8
from PIL import ImageFile, Image
import os

def remove_metadata(img: ImageFile.ImageFile):
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

        if os.path.isfile(file_path) and file_path.endswith("png"):
            img = Image.open(file_path)
            img = remove_metadata(img)
            img.save(file_path, "png")


if __name__ == "__main__":
    recursive_search(os.path.abspath("./data"))
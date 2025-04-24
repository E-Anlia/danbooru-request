# coding=utf-8
import requests
import bs4
import logging
import os
import io
import math
from PIL import Image
from typing import List
from collections import OrderedDict
import yaml
from pathlib import Path

from washer import remove_metadata


class SpiderConfig:

    def __init__(self):
        with open("./config.yml", "r") as f:
            conf = yaml.safe_load(f)
            if conf.get("spider") is None:
                raise FileNotFoundError("No config found for spider")

            conf = conf.get("spider")
            self.domain = conf["domain"]
            self.protocal = conf["protocol"]
            self.save_location = conf["file_save_location"]
            self.max_res = conf["max_res"]
            self.latest_id = conf["latest_id"]
            self.max_id = conf["max_id"]
            self.target_format = conf["target_format"]

            if self.max_id < self.latest_id:
                raise ValueError("max_id must be larger than latest_id")

            Path(self.save_location).mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


logging.basicConfig(
    level=logging.INFO,
    filename="run.log",
    filemode="a",
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

_CONFIG = SpiderConfig()


def compress_img(img: Image.Image) -> Image.Image:
    """压缩图片"""
    res = img.width * img.height
    if res <= _CONFIG.max_res:
        return img
    ratio = math.sqrt(_CONFIG.max_res / res)
    return img.resize((int(img.width * ratio), int(img.height * ratio)))


def save_img(link: str, path: str):
    """保存图片"""
    file_content = requests.get(link).content
    img = Image.open(io.BytesIO(file_content))
    img = remove_metadata(img)
    img = compress_img(img)
    img.save(f"{path}.{_CONFIG.target_format}", _CONFIG.target_format)


def save_tags(tags: List[str], path: str):
    """保存标签"""
    file_content = ",".join(tags)
    with open(path, "w") as f:
        f.write(file_content)


def run(id: int):
    try:
        response = requests.get(f"{_CONFIG.protocal}://{_CONFIG.domain}/posts/{id}")
    except Exception as e:
        logging.error(f"id: {id} request error: {repr(e)}.")
        return

    if response.status_code != 200:
        logging.error(f"id: {id} return {response.status_code}.")
        return

    content = bs4.BeautifulSoup(response.content, "html.parser")

    try:
        # 图片链接
        orig_link_node = content.find("a", class_="image-view-original-link")
        if orig_link_node:
            img_link = orig_link_node["href"]
        else:
            img_link = (
                content.find(id="image")["src"]
                .replace("sample-", "")
                .replace("/sample/", "/original/")
            )

        # 作者
        artists = [
            node.text
            for node in content.find("ul", class_="artist-tag-list").find_all(
                class_="search-tag"
            )
        ]
        # 标签
        tags = [
            node.text.replace("_", " ")
            for node in content.find(class_="tag-list categorized-tag-list").find_all(
                class_="search-tag"
            )
        ]
        tags = list(OrderedDict.fromkeys(tags))  # 去重
    except Exception as e:
        logging.error(f"id: {id} html parse error: {repr(e)}.")
        return

    logging.info(
        f"id: {id}, artist: {artists}, tags count: {len(tags)}, link: {img_link}."
    )

    if len(artists) >= 1:
        folder_name = artists[0]
    else:
        folder_name = "unkown"

    folder = os.path.join(_CONFIG.save_location, folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    try:
        save_img(img_link, os.path.join(folder, f"{id}"))
        save_tags(tags, os.path.join(folder, f"{id}.txt"))
    except Exception as e:
        logging.error(f"id: {id} save file error: {repr(e)}.")
        return

    logging.info(f"id: {id} done.")


if __name__ == "__main__":
    for id in range(_CONFIG.latest_id, _CONFIG.max_id):
        run(id)

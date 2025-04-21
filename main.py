# coding=utf-8
import requests
import bs4
import logging
import os
import io
import math
from PIL import Image, ImageFile
from typing import List

from washer import remove_metadata

DOMAIN = "danbooru.donmai.us"
PROTOCOL = "https"

FILE_SAVE_LOCATION = "./data"

MAX_RES = 2048*2048

LATEST_ID = 9182170 # 最后图片在danbooru上的id
MAX_ID = 9182175 # 最大id

logging.basicConfig(level=logging.INFO,filename='run.log',filemode='a',format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def compress_img(img: ImageFile.ImageFile):
    """压缩图片"""
    res = img.width * img.height
    if res <= MAX_RES:
        return img
    ratio =  math.sqrt(MAX_RES / res)
    return img.resize((int(img.width*ratio), int(img.height*ratio)))


def save_img(link: str,path: str):
    """保存图片"""
    file_content = requests.get(link).content
    img = Image.open(io.BytesIO(file_content))
    img = remove_metadata(img)
    img = compress_img(img)
    img.save(path, "png")


def save_tags(tags: List[str],path: str):
    """保存标签"""
    file_content = ",".join(tags)
    with open(path,"w") as f:
        f.write(file_content)

def run(id: int):
    try:
        response =  requests.get(f"{PROTOCOL}://{DOMAIN}/posts/{id}")
    except Exception as e:
        logging.error(f"id: {id} request error: {repr(e)}.")
        return
    
    if response.status_code != 200:
        logging.error(f"id: {id} return {response.status_code}.")
        return

    content = bs4.BeautifulSoup(response.content,"html.parser")

    try:
        # 图片链接
        orig_link_node = content.find("a", class_="image-view-original-link")
        if orig_link_node:
            img_link = orig_link_node["href"]
        else:
            img_link = content.find(id="image")["src"].replace("sample-","").replace("/sample/","/original/")

        # 作者
        artists = [node.text for node in content.find("ul",class_="artist-tag-list").find_all(class_="search-tag")]
        # 标签
        tags = [node.text.replace("_", " ") for node in content.find(class_="tag-list categorized-tag-list").find_all(class_="search-tag")]
    except Exception as e:
        logging.error(f"id: {id} html parse error: {repr(e)}.")
        return
    
    logging.info(f"id: {id}, artist: {artists}, tags count: {len(tags)}, link: {img_link}.")

    if len(artists) >= 1:
        folder_name = artists[0]
    else:
        folder_name = "unkown"

    if not os.path.exists(f"{FILE_SAVE_LOCATION}/{folder_name}"):
        os.mkdir(f"{FILE_SAVE_LOCATION}/{folder_name}")

    try:
        save_img(img_link, f"{FILE_SAVE_LOCATION}/{folder_name}/{id}.png")
        save_tags(tags, f"{FILE_SAVE_LOCATION}/{folder_name}/{id}.txt")
    except Exception as e: 
        logging.error(f"id: {id} save file error: {repr(e)}.")
        return

    logging.info(f"id: {id} done.")


if __name__ == "__main__":
    for id in range(LATEST_ID, MAX_ID):
        run(id)
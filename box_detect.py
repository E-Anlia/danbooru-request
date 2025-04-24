import os
from pathlib import Path
from typing import List, Tuple
import yaml
from ultralytics import YOLO
from PIL import Image
import logging


class BBoxConfig:

    def __init__(self):
        with open("./config.yml", "r") as f:
            conf = yaml.safe_load(f)
            if conf.get("bbox") is None:
                raise FileNotFoundError("No config found for bbox")

            conf = conf.get("bbox")
            self.image_folder = conf["image_folder"]
            self.txt_folder = conf["txt_folder"]
            self.model_path = conf["model_path"]
            self.nlp_out = conf["nlp_out"]
            self.tag_out = conf["tag_out"]
            self.filter_format = tuple(conf["filter_format"])

            Path(self.nlp_out).mkdir(parents=True, exist_ok=True)
            Path(self.tag_out).mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


logging.basicConfig(
    level=logging.INFO,
    filename="run.log",
    filemode="a",
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

_CONFIG = BBoxConfig()


class PersonDetector:

    def __init__(self):
        self.YOLO = YOLO(_CONFIG.model_path)  # 加载检测模型

    def detect_person_bbox(
        self, image_path: str
    ) -> Tuple[int, int, int, int]:  # x1, y1, x2, y2
        """
        使用 BBOX 模型检测画面中人物，返回最主要人物的边界框
        # TODO: 没检测到人物
        """
        logging.info(f"Trying detecting {image_path}")
        results = self.YOLO(image_path)
        # 过滤出 'person' 类别
        persons = [r for r in results[0].boxes if int(r.cls) == 0]
        if not persons:
            logging.error(f"No person detected in {image_path}")
            return

        # 取最大 bbox
        box = max(persons, key=lambda b: b.xyxy[0][2] * b.xyxy[0][3])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        logging.info(f"Person detected in {image_path}, {[x1, y1, x2, y2]}")
        return x1, y1, x2, y2

    def compute_grid_cell(
        self, image_size: Tuple[int, int], bbox: Tuple[int, int, int, int]
    ) -> str:
        """
        将图像分为 3x3 网格，根据 bbox 中心点返回格子标签（A1~C3）
        """
        w, h = image_size
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        col = min(cx * 3 // w + 1, 3)
        row = min(cy * 3 // h, 2)
        row_label = ["A", "B", "C"][row]
        return f"{row_label}{col}"

    # 第二步：读取已有 txt，并调用自然语言打标器 -->
    def call_nlp_tagger(self, image_path: str, txt_path: str) -> str:
        """
        TODO: 调用自然语言打标服务，返回打标结果文本
        """
        # 这里填入具体 API 调用
        # response = nlp_tagger.tag(image=image_path, prompt=open(txt_path).read())
        response = "[NATURAL LANGUAGE TAGS]"
        # 如果有默认前缀则去除 -->
        return response.lstrip("DefaultPrefix:")

    # 第三步：调用 Danbooru 打标器，并去重/排序 -->
    def call_danbooru_tagger(self, image_path: str, txt: str) -> List[str]:
        """
        TODO: 返回 Danbooru 风格的标签列表
        """
        # response_tags = danbooru_tagger.tag(image=image_path, prompt=txt)
        response_tags = ["tag1", "tag2", "tag1"]  # 示例
        # 语义去重 (waifuset)
        uniq = list(dict.fromkeys(response_tags))
        return uniq

    def fetch_and_sort_tags(self, tags: List[str]) -> List[str]:
        """
        TODO: 通过 Danbooru API 获取标准化标签，并按特定格式排序 -->
        """
        # 示例：调用 API 获取每个 tag 的类别/权重
        # sorted_tags = sort_by_formula(tags, metadata)
        return tags

    # 第四步：输出最终 txt 并清理原始 -->
    def write_final_tags(self, out_path: str, tags: List[str]):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(",".join(tags))

    def recursive_search(self, path: str):
        """递归搜索文件夹里的图片"""
        for file_path in os.listdir(path):
            file_path = os.path.join(path, file_path)

            if os.path.isdir(file_path):
                self.recursive_search(file_path)

            if os.path.isfile(file_path) and file_path.lower().endswith(
                _CONFIG.filter_format
            ):
                self.detect_img(file_path)

    def detect_img(self, file_path: str):
        base = os.path.splitext(file_path)[0]
        img_path = file_path
        txt_src = os.path.join(_CONFIG.txt_folder, f"{base}.txt")

        # 人物检测 & 网格位置
        bbox = pd.detect_person_bbox(img_path)
        if bbox is None:
            return

        img = Image.open(img_path)
        pos = pd.compute_grid_cell(img.size, bbox)

        logging.info(f"Person pos: {pos} in {img_path}")

        # 自然语言打标 -->
        nlp_tags = pd.call_nlp_tagger(img_path, txt_src)
        with open(
            os.path.join(_CONFIG.nlp_out, f"{base}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(nlp_tags)

        # Danbooru 打标 & 去重 & 排序 -->
        tags = pd.call_danbooru_tagger(img_path, nlp_tags)
        tags = pd.fetch_and_sort_tags(tags)
        # 添加位置标签到最前
        tags.insert(0, pos)

        # 输出最终标签并删除中间文件 -->
        pd.write_final_tags(os.path.join(_CONFIG.tag_out, f"{base}.txt"), tags)
        os.remove(os.path.join(_CONFIG.nlp_out, f"{base}.txt"))

    def run(self):
        self.recursive_search(os.path.abspath(_CONFIG.image_folder))


if __name__ == "__main__":
    pd = PersonDetector()
    pd.run()

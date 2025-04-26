import os
import re
from waifuset import WaifuScorer as WScorer
from PIL import Image
import yaml
import logging
from typing import List


class ScorerConfig:

    def __init__(self):
        with open("./config.yml", "r") as f:
            conf = yaml.safe_load(f)
            if conf.get("scorer") is None:
                raise FileNotFoundError("No config found for scorer")

            conf = conf.get("scorer")
            self.batch_size = conf["batch_size"]
            self.model_path = conf["model_path"]
            self.image_folder = conf["image_folder"]
            self.filter_format = tuple(conf["filter_format"])

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


logging.basicConfig(
    level=logging.INFO,
    filename="scorer.log",
    filemode="a",
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

_CONFIG = ScorerConfig()


class WaifuScorer:
    def __init__(self):
        self.scorer = WScorer.from_pretrained(
            pretrained_model_name_or_path=_CONFIG.model_path,
            emb_cache_dir=None,
        )

    def get_score(self, images: List[Image.Image]):
        # 批量评分
        return self.scorer(images)

    def covert_quality(self, txt: str, score: float):
        """根据分数修改质量tag"""
        if score >= 0.8:
            q_tag = "masterpiece"
        elif 0.5 <= score < 0.8:
            q_tag = "best quality"
        elif 0.3 <= score < 0.5:
            q_tag = "normal quality"
        else:
            q_tag = "worst quality"

        tags = [
            tag
            for tag in txt.split(",")
            if not tag
            in ("masterpiece", "best quality", "normal quality", "worst quality")
        ]
        tags.append(q_tag)
        return ",".join(tags)

    def write_score(self, txt_path: str, score: float):
        """智能写入评分到标签文件"""
        content = ""

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

        new_content = self.covert_quality(content, score)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(new_content.strip())
            logging.info(f"new score written in: {txt_path}")

    def run(self, path: str):
        base_folder = os.path.abspath(path)
        image_files = []

        for f_name in os.listdir(base_folder):
            abs_path = os.path.join(base_folder, f_name)

            # 递归子文件夹
            if os.path.isdir(abs_path):
                self.run(abs_path)
                continue

            if f_name.lower().endswith(_CONFIG.filter_format):
                image_files.append(abs_path)

        for i in range(0, len(image_files), _CONFIG.batch_size):
            batch_files = image_files[i : i + _CONFIG.batch_size]
            images = [Image.open(p) for p in batch_files]

            try:
                scores = self.get_score(images)
            except Exception as e:
                logging.error(f"batch score failed: {repr(e)}")
                continue

            for score, img_path in zip(scores, batch_files):
                txt_path = img_path.rsplit(".", 1)[0] + ".txt"
                logging.info(f"score: {score} for {img_path}")
                self.write_score(txt_path, score)


if __name__ == "__main__":
    WaifuScorer().run(_CONFIG.image_folder)

from safetensors.torch import load_file, save_file
import yaml
from pathlib import Path
import logging
import os


class CombineConfig:

    def __init__(self):
        with open("./config.yml", "r") as f:
            conf = yaml.safe_load(f)
            if conf.get("combine") is None:
                raise FileNotFoundError("No config found for combine")

            conf = conf.get("combine")
            self.model_folder = conf["model_folder"]
            self.output_file = conf["output_file"]

            Path(self.output_file).mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


logging.basicConfig(
    level=logging.INFO,
    filename="combine.log",
    filemode="a",
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

_CONFIG = CombineConfig()


def run(model_folder: str, output_file: str):
    """
    合并模型
    """
    base_path = os.path.abspath(model_folder)

    shard_files = [
        os.path.join(base_path, file)
        for file in os.listdir(base_path)
        if file.endswith(".safetensors")
    ]

    logging.info(f"Trying to combine {shard_files}")
    combined = {}
    for f in shard_files:
        sd = load_file(f)
        combined.update(sd)
    # 输出为单文件
    save_file(combined, output_file)
    logging.info(f"Combined model saved to {output_file}")


if __name__ == "__main__":
    run(_CONFIG.model_folder, _CONFIG.output_file)

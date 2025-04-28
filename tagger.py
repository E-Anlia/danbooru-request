# coding=utf-8
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import logging
import yaml


class TaggerConfig:

    def __init__(self):
        with open("./config.yml", "r") as f:
            conf = yaml.safe_load(f)
            if conf.get("tagger") is None:
                raise FileNotFoundError("No config found for tagger")

            conf = conf.get("tagger")
            self.model_path = conf["model_path"]
            self.image_folder = conf["image_folder"]
            self.output_folder = conf["output_folder"]
            self.batch_size = conf["batch_size"]
            self.overwrite = conf["overwrite"]
            self.filter_format = tuple(conf["filter_format"])

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


logging.basicConfig(
    level=logging.INFO,
    filename="tagger.log",
    filemode="a",
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

_CONFIG = TaggerConfig()


class NaturalTagger:
    """自然语言打标器"""

    def __init__(self):
        # 模型加载信息
        logging.info(f"Loading ToriiGate-v0.4-7B model file from {_CONFIG.model_path}")
        self.processor = Qwen2VLProcessor.from_pretrained(
            _CONFIG.model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
            padding_side="left",
            use_fast=True,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            _CONFIG.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        logging.info(f"Loaded ToriiGate-v0.4-7B")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        logging.info("all done / 识别完成")
        logging.info("Unloaded ToriiGate-v0.4-7B")

    # def load_metadata(self, image_path):

    # meta = {"booru_tags": None, "existing_caption": None}
    # base_name = os.path.splitext(image_path)[0]

    # # 加载booru标签
    # tags_path = f"{base_name}_tags.txt"
    # if os.path.exists(tags_path):
    #     with open(tags_path, "r") as f:
    #         meta["booru_tags"] = f.read().strip()

    # # 加载现有描述
    # caption_path = f"{base_name}.txt"
    # if os.path.exists(caption_path):
    #     with open(caption_path, "r") as f:
    #         meta["existing_caption"] = f.read().strip()

    # return meta

    def load_tags(self, image_path):
        """从txt加载tag"""
        tags_path = os.path.splitext(image_path)[0] + ".txt"

        if not os.path.exists(tags_path):
            logging.error(f"tags not exists for {image_path}")
            return ""

        with open(tags_path) as f:
            return "".join(f.readlines())

        return ""

    # def build_prompt(self, metadata):
    #     base_prompt = """
    #     You need to compare given caption with the picture and using chain of thought to write a medium-short and convenient caption for the picture.
    #     """

    #     if metadata["existing_caption"]:
    #         return f"""
    #         {base_prompt}
    #         Existing reference: {metadata['existing_caption']}
    #         Cross-validate with image content and refine if needed
    #         """
    #     else:
    #         return base_prompt

    def run(self, base_folder: str):
        # ====== 修改文件列表生成逻辑（核心修复点） ======
        image_files = []
        base_folder = os.path.abspath(base_folder)

        for f_name in os.listdir(base_folder):
            abs_path = os.path.join(base_folder, f_name)

            # 递归子文件夹
            if os.path.isdir(abs_path):
                self.run(abs_path)
                continue

            parent_name = os.path.split(os.path.dirname(abs_path))[-1]

            if f_name.lower().endswith(_CONFIG.filter_format):
                txt_path = os.path.join(
                    _CONFIG.output_folder, parent_name, f_name.split(".")[0] + ".txt"
                )  # 图片同名txt文件
                if not os.path.exists(txt_path) or _CONFIG.overwrite:
                    image_files.append(abs_path)
                else:
                    logging.info(f"skip existed: {txt_path}")

        for i in range(0, len(image_files), _CONFIG.batch_size):
            batch_files = image_files[i : i + _CONFIG.batch_size]
            texts = []
            images = []

            for image_path in batch_files:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": self.load_tags(image_path)},
                        ],
                    }
                ]

                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_input, _ = process_vision_info(messages)

                texts.append(text_input)
                images.append(image_input)

            inputs = self.processor(
                text=texts, images=images, padding=True, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.001,
                    top_p=0.01,
                    top_k=1,
                    repetition_penalty=1.0,
                )

            output_texts = self.processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # ====== 新增显存清理逻辑 ======
            del inputs, generated_ids  # 删除大张量
            torch.cuda.empty_cache()  # 强制清理缓存 [[8]][[10]]

            # 核心修复部分：仅去除特殊符号
            for idx, output_text in enumerate(output_texts):
                # 仅处理回车符和首尾空格
                output_text = output_text.replace("\r", " ").strip()

                # 验证处理结果
                # logging.info(f"Processed output sample: {output_text[:100]}...")

                # 保存处理后的文本
                parent_name = os.path.split(os.path.dirname(batch_files[idx]))[-1]
                file_name = os.path.split(batch_files[idx])[-1]

                txt_path = os.path.join(
                    _CONFIG.output_folder,
                    parent_name,
                    os.path.splitext(file_name)[0] + ".txt",
                )
                os.makedirs(
                    os.path.join(_CONFIG.output_folder, parent_name), exist_ok=True
                )

                with open(txt_path, "w", encoding="utf-8", newline="") as f:
                    f.write(output_text)  # 直接写入未处理的原始文本

            # 额外清理
            del output_texts, texts, images  # 清理批次数据


if __name__ == "__main__":
    with NaturalTagger() as tagger:
        tagger.run(_CONFIG.image_folder)

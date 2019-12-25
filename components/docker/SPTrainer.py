# coding=utf-8
from __future__ import absolute_import, print_function

import os
import itertools
import numpy as np
from suanpan.app.arguments import Folder
from suanpan.utils import image, json
from suanpan.app import app
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator
from arguments import CFG



def get_balloon_dicts(img_dir, json_file):

    imgs_anns = json.load(json_file)

    dataset_dicts = []
    imagefile = [i.split(".jpg")[0] + ".jpg" for i in imgs_anns["metadata"].keys()]
    for idx, v in enumerate(set(imagefile)):
        record = {}

        indices = [i for i, x in enumerate(imagefile) if x == v]

        filename = os.path.join(img_dir, v)
        height, width = image.read(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for index in indices:
            data = list(imgs_anns["metadata"].values())[index]
            xy = data["xy"][1:]
            px = xy[::2]
            py = xy[1::2]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0,
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


@app.input(CFG(key="inputModel"))
@app.input(Folder(key="inputTrainImage"))
@app.input(Folder(key="inputTrainJson"))
@app.input(Folder(key="inputTestImage"))
@app.input(Folder(key="inputTestJson"))
@app.output(Folder(key="outputModel"))
def SPTrainer(context):
    args = context.args
    cfg = args.inputModel

    trainJson = os.path.join(args.inputTrainJson, "data.json")
    testJson = os.path.join(args.inputTestJson, "data.json")

    for d, image_dir, json_file in zip(
        ["train", "val"],
        [args.inputTrainImage, args.inputTestImage],
        [trainJson, testJson],
    ):
        DatasetCatalog.register(
            "dataset_" + d, lambda d=d: get_balloon_dicts(image_dir, json_file)
        )
        MetadataCatalog.get("dataset_" + d).set(thing_classes=["balloon"])
    cfg.MODEL.DEVICE = "cpu"
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ("dataset_val",)
    cfg.MODEL.WEIGHTS = "/home/yanqinghao/code/AiLab-detectron2/dataset/input/weight/model_final_f10217.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = (
        300  # 300 iterations seems good enough, but you can certainly train longer
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for this toy dataset
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.OUTPUT_DIR = args.outputModel
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    evaluator = COCOEvaluator("dataset_val", cfg, False, output_dir="./output/")
    trainer.test(cfg, trainer.model, evaluators=evaluator)

    return args.outputModel


if __name__ == "__main__":
    SPTrainer()

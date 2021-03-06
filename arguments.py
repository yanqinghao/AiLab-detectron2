# coding=utf-8
from __future__ import absolute_import, print_function

import os
from suanpan.model import Model as BaseModel
from suanpan.storage import storage, StorageProxy
from suanpan.utils import image
from suanpan import path
from suanpan import g
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from utils.annotation import get_balloon_dicts
from detectron2.utils.logger import setup_logger

setup_logger()


class CFGModel(BaseModel):
    def __init__(self):
        super(CFGModel, self).__init__()

    def load(self, path):
        self.output = storage.getPathInTempStore(
            os.path.join("studio", g.userId, g.appId, g.nodeId, "model")
        )
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(path, "model.cfg"))
        self.cfg = cfg
        self.set_weights()
        return path

    def save(self, path):
        if hasattr(self, "output"):
            if len(os.listdir(self.output)) > 0:
                storage.upload(
                    os.path.join("studio", g.userId, g.appId, g.nodeId, "model"),
                    self.output,
                )
                self.cfg.MODEL.WEIGHTS = os.path.join(
                    "studio", g.userId, g.appId, g.nodeId, "model"
                )
        with open(os.path.join(path, "model.cfg"), "wb") as f:
            f.write(self.cfg.dump().encode("ascii"))
        return path

    def train(self):
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        if os.path.exists(self.output):
            path.remove(self.output)
            os.makedirs(self.output)
        else:
            os.makedirs(self.output)
        trainer.train()
        return trainer

    def evaluate(self, trainer, output_val):
        evaluator = COCOEvaluator("dataset_val", self.cfg, False, output_dir=output_val)
        trainer.test(self.cfg, trainer.model, evaluators=evaluator)

    def predict(self, imgs):
        predictor = DefaultPredictor(self.cfg)
        outputs = []
        for img in imgs:
            im = image.read(img)
            output = predictor(im)
            outputs.append(output)
        return outputs

    def visualizer(self, imgs, datas, output_dir):
        for img, data in zip(imgs, datas):
            im = image.read(img)
            v = Visualizer(
                im[:, :, ::-1],
                MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                scale=1.2,
            )
            v = v.draw_instance_predictions(data["instances"].to("cpu"))
            image.save(
                os.path.join(output_dir, os.path.basename(img)),
                v.get_image()[:, :, ::-1],
            )

    def load_model(self, model_cfg, model_weights):
        cfg = get_cfg()
        cfg.merge_from_file(model_cfg)
        self.cfg = cfg
        self.cfg.MODEL.WEIGHTS = model_weights

    def device(self, device_type):
        self.cfg.MODEL.DEVICE = device_type

    def set_weights(self):
        weight_path = self.cfg.MODEL.WEIGHTS
        if weight_path.split(storage.delimiter)[0] in ["studio", "common"]:
            if weight_path.split(storage.delimiter)[0] == "studio":
                storage.download(weight_path, self.output)
                if storage.type == "local":
                    self.output = storage.getPathInTempStore(weight_path)
            else:
                if storage.type == "local":
                    storage_oss = StorageProxy(None, None)
                    storage_oss.setBackend(type="oss")
                else:
                    storage_oss = storage
                storage_oss.download(
                    weight_path,
                    os.path.join(self.output, os.path.basename(weight_path)),
                )
            file_list = os.listdir(self.output)
            file_name = [i for i in file_list if i.endswith(("pkl", "pth"))]
            self.cfg.MODEL.WEIGHTS = os.path.join(self.output, file_name[0])
        else:
            pass

    def set_params(self, **kwd):
        self.cfg.DATASETS.TRAIN = ("dataset_train",)
        self.cfg.DATASETS.TEST = ("dataset_val",)
        self.cfg.DATALOADER.NUM_WORKERS = kwd.pop("NUM_WORKERS")
        self.cfg.SOLVER.IMS_PER_BATCH = kwd.pop("IMS_PER_BATCH")
        self.cfg.SOLVER.BASE_LR = kwd.pop("BASE_LR")
        self.cfg.SOLVER.MAX_ITER = kwd.pop("MAX_ITER")
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = kwd.pop("BATCH_SIZE_PER_IMAGE")
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = kwd.pop("NUM_CLASSES")
        self.cfg.OUTPUT_DIR = self.output

    def dataset_register(self, trainImg, testImg, trainJson, testJson, classes):
        for d, image_dir, json_file in zip(
            ["train", "val"], [trainImg, testImg], [trainJson, testJson],
        ):
            DatasetCatalog.register(
                "dataset_" + d, lambda d=d: get_balloon_dicts(image_dir, json_file)
            )
            MetadataCatalog.get("dataset_" + d).set(thing_classes=classes)

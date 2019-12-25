# coding=utf-8
from __future__ import absolute_import, print_function

import os
from suanpan.app.arguments import Folder, Json
from suanpan.utils import image
from suanpan.app import app
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from arguments import CFG


@app.input(CFG(key="inputModel"))
@app.input(Folder(key="inputData"))
@app.output(Json(key="outputData1"))
@app.output(Folder(key="outputData2"))
def SPPredictor(context):
    args = context.args
    cfg = args.inputModel
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "/home/yanqinghao/code/AiLab-detectron2/dataset/input/weight/model_final_f10217.pkl"
    predictor = DefaultPredictor(cfg)
    im = image.read(os.path.join(args.inputData, os.listdir(args.inputData)[0]))
    outputs = predictor(im)
    v = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image.save(
        os.path.join(args.outputData2, os.listdir(args.inputData)[0]),
        v.get_image()[:, :, ::-1],
    )
    pred_data = {
        "image_size": outputs["instances"].image_size,
        "pred_boxes": outputs["instances"].pred_boxes.tensor.tolist(),
        "pred_classes": outputs["instances"].pred_classes.tolist(),
        "pred_masks": outputs["instances"].pred_masks.tolist(),
        "scores": outputs["instances"].scores.tolist(),
    }
    return pred_data, args.outputData2


if __name__ == "__main__":
    SPPredictor()

# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from suanpan.app.arguments import Folder, Json, Model, Int, Float
from suanpan.app import app
from suanpan.log import logger
from arguments import CFGModel
from utils.tools import find_files


@app.input(Model(key="inputModel", type=CFGModel))
@app.input(Folder(key="inputData"))
@app.param(Int(key="__gpu", default=0))
@app.param(Float(key="ScoreThreshTest", default=0.5))
@app.output(Json(key="outputData1"))
@app.output(Folder(key="outputData2"))
def SPPredictor(context):
    args = context.args
    model = args.inputModel
    device = "cuda" if args.__gpu else "cpu"
    logger.info("**********Use {} Device for Predicting**********".format(device))
    model.device(device)
    model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.ScoreThreshTest

    fileType = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )
    images = find_files(args.inputData, fileType)

    outputs = model.predict(images)
    model.visualizer(images, outputs, args.outputData2)
    pred_data = []
    for output in outputs:
        pred_data.append(
            {
                "image_size": getattr(output["instances"], "image_size", ()),
                "pred_boxes": getattr(
                    getattr(output["instances"], "pred_boxes", None),
                    "tensor",
                    torch.tensor([]),
                ).tolist(),
                "pred_classes": getattr(
                    output["instances"], "pred_classes", torch.tensor([])
                ).tolist(),
                "pred_masks": getattr(
                    output["instances"], "pred_masks", torch.tensor([])
                ).tolist(),
                "scores": getattr(
                    output["instances"], "scores", torch.tensor([])
                ).tolist(),
                "pred_keypoints": getattr(
                    output["instances"], "pred_keypoints", torch.tensor([])
                ).tolist(),
            }
        )
    return pred_data, args.outputData2


if __name__ == "__main__":
    SPPredictor()

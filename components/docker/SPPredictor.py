# coding=utf-8
from __future__ import absolute_import, print_function

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

    images = find_files(args.inputData)

    outputs = model.predict(images)
    model.visualizer(images, outputs, args.outputData2)
    pred_data = []
    for output in outputs:
        pred_data.append(
            {
                "image_size": output["instances"].image_size,
                "pred_boxes": output["instances"].pred_boxes.tensor.tolist(),
                "pred_classes": output["instances"].pred_classes.tolist(),
                "pred_masks": output["instances"].pred_masks.tolist(),
                "scores": output["instances"].scores.tolist(),
            }
        )
    return pred_data, args.outputData2


if __name__ == "__main__":
    SPPredictor()

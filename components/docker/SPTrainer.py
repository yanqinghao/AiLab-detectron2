# coding=utf-8
from __future__ import absolute_import, print_function

import os
from suanpan.app.arguments import Folder, Int, Float, Model
from suanpan.app import app
from suanpan.log import logger
from suanpan.utils import json
from arguments import CFGModel


@app.input(Model(key="inputModel", type=CFGModel))
@app.input(Folder(key="inputTrainImage"))
@app.input(Folder(key="inputTrainJson"))
@app.input(Folder(key="inputTestImage"))
@app.input(Folder(key="inputTestJson"))
@app.param(Int(key="__gpu", default=0))
@app.param(Int(key="ImgsPerBatch", default=2))
@app.param(Float(key="BaseLR", default=0.00025))
@app.param(Int(key="MaxIter", default=300))
@app.param(Int(key="BatchSizePerImg", default=128))
@app.output(Model(key="outputModel", type=CFGModel))
@app.output(Folder(key="outputData"))
def SPTrainer(context):
    args = context.args
    model = args.inputModel
    device = "cuda" if args.__gpu else "cpu"
    logger.info("**********Use {} Device for Training**********")

    trainImg = args.inputTrainImage
    trainJson = os.path.join(args.inputTrainJson, "project.json")

    if args.inputTestImage is None or args.inputTestJson is None:
        testImg = trainImg
    else:
        testImg = args.inputTestImage

    if args.inputTestImage is None or args.inputTestJson is None:
        testJson = trainJson
    else:
        testJson = os.path.join(args.inputTestJson, "project.json")
    jsonData = json.load(trainJson)
    classes = list(jsonData["attribute"]["1"]["options"].values())
    model.dataset_register(trainImg, testImg, trainJson, testJson, classes)
    model.device(device)
    params = {
        "IMS_PER_BATCH": args.ImgsPerBatch,
        "BASE_LR": args.BaseLR,
        "MAX_ITER": args.MaxIter,
        "BATCH_SIZE_PER_IMAGE": args.BatchSizePerImg,
        "NUM_CLASSES": len(classes),
    }
    model.set_params(**params)
    trainer = model.train()
    model.evaluate(trainer, args.outputData)

    return model, args.outputData


if __name__ == "__main__":
    SPTrainer()

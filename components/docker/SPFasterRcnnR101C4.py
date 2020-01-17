# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app import app
from suanpan.app.arguments import Model, Bool
from arguments import CFGModel


@app.param(Bool(key="usePredWeight", default=True))
@app.output(Model(key="model", type=CFGModel))
def SPFasterRcnnR101C4(context):
    args = context.args
    if args.usePredWeight:
        args.model.load_model(
            "./detectron2/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
            "common/model/detectron2/ImageNetPretrained/MSRA/model_final_298dad.pkl",
        )
    else:
        args.model.load_model(
            "./detectron2/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
            "",
        )
    return args.model


if __name__ == "__main__":
    SPFasterRcnnR101C4()

# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app import app
from suanpan.app.arguments import Model, Bool
from arguments import CFGModel


@app.param(Bool(key="usePredWeight", default=True))
@app.output(Model(key="model", type=CFGModel))
def SPMaskRcnnR101DC5(context):
    args = context.args
    if args.usePredWeight:
        args.model.load_model(
            "./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
            "common/model/detectron2/ImageNetPretrained/MSRA/model_final_0464b7.pkl",
        )
    else:
        args.model.load_model(
            "./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
            "",
        )
    return args.model


if __name__ == "__main__":
    SPMaskRcnnR101DC5()

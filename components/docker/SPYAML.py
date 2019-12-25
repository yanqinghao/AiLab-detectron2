# coding=utf-8
from __future__ import absolute_import, print_function

from detectron2.config import get_cfg
from suanpan.app import app
from arguments import CFG


# @app.input(CFG(key="inputModel"))
@app.output(CFG(key="outputModel"))
def SPYAML(context):
    args = context.args
    cfg = get_cfg()
    cfg.merge_from_file(
        "./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    return args.inputModel


if __name__ == "__main__":
    SPYAML()

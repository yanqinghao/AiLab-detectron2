import os
import itertools
import numpy as np
from suanpan.utils import image, json
from detectron2.structures import BoxMode


def get_balloon_dicts(img_dir, json_file):
    """
    Parsing via json
    """
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

# coding=utf-8
from __future__ import absolute_import, print_function

from detectron2.config import get_cfg
from suanpan.components import Result
from suanpan.storage.arguments import Model


class CFG(Model):
    FILETYPE = "cfg"

    def format(self, context):
        super(CFG, self).format(context)
        if self.filePath:
            cfg = get_cfg()
            cfg.merge_from_file(self.filePath)
            self.value = cfg
        return self.value

    def save(self, context, result):
        with open(self.filePath, "wb") as f:
            f.write(result.value.dump().encode("ascii"))
        return super(CFG, self).save(context, Result.froms(value=self.filePath))


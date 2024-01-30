#from lightglue import LightGlue as LightGlue_
from .lightglue import LightGlue as LightGlue_
from omegaconf import OmegaConf

from ..base_model import BaseModel


class LightGluePreTrainedMINE(BaseModel):
    default_conf = {"features": "superpoint", **LightGlue_.default_conf}
    required_data_keys = [
        "view0",
        "keypoints0",
        "descriptors0",
        "view1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        dconf = OmegaConf.to_container(conf)
        self.net = LightGlue_(dconf) #dconf.pop("features"), **dconf)
        print(self.net.conf)
        self.set_initialized()

    def _forward(self, data):
        required_keys = ["keypoints", "descriptors", "scales", "oris"]
        # view0 = {
        #     **data["view0"],
        #     **{k: data[k + "0"] for k in required_keys if (k + "0") in data},
        # }
        # view1 = {
        #     **data["view1"],
        #     **{k: data[k + "1"] for k in required_keys if (k + "1") in data},
        # }
        #merged = {**view0,**view1}
        return self.net(data) #self.net({"image0": view0, "image1": view1})

    def loss(pred, data):
        raise NotImplementedError

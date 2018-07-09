from models.vgg_model import Vgg
from models.vgg_model import Vgg_HParams

from models.isling_model import Isling
from models.isling_model import Isling_HParams

from models.isling_STN_model import IslingST
from models.isling_STN_model import IslingST_HParams

from models.isling_v2_model import Islingv2
from models.isling_v2_model import Islingv2_HParams

__all__ = [
    "Vgg",
    "Isling",
    "IslingST",
    "Islingv2"
]


def get_model(model_name, hps, images, labels, mode):
    if model_name in __all__:
        return globals()[model_name](hps, images, labels, mode)
    else:
        raise Exception('The model name %s does not exist' % model_name)


def get_model_class(model_name):
    if model_name in __all__:
        return globals()[model_name]
    else:
        raise Exception('The model name %s does not exist' % model_name)


def get_model_HParams(model_name):
    if model_name in __all__:
        return globals()[model_name + '_HParams']
    else:
        raise Exception('The model name %s does not exist' % model_name)

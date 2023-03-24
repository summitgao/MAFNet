
from .model_mafnet import MAFNet

__factory = {
    'MAFNet': MAFNet
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown layer:", name)
    return __factory[name](*args, **kwargs)
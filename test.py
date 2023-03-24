# coding=utf-8
# Version:python 3.7
from setup import *
from utils import *
import h5py

def test():
    arg = option()
    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    model = getmodel(arg.input, middle=[64,128,256], model_name = arg.model)
    model = model.to(device)

    model = load_model(model, arg.pretrain_modle, mode='test')

    # ICVL-gauss
    testfile('./data/test/gavyam_0823-0933.mat', model, device, 'icvl', 14, (69, 79), h5py.File, 'rad',
        gettransform(arg.blind, arg.noise), channel_first=True, showres=True, cropcenter=True)


if __name__ == '__main__':
    test()

# coding=utf-8
# Version:python 3.7


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.io as io
from torchvision.transforms import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from add_noise import *
import cv2
import matplotlib.pyplot as plt


def gettransform(blind, noise):
    transform = []
    if blind == 'no':
        transform.append(AddNoise(noise))
    if blind == 'gauss':
        transform.append(AddNoiseNoniid([30, 50, 70]))
    if blind == 'complex':
        transform.extend(
            [
                SequentialSelect(
                    transforms=[
                        lambda x: x,
                        AddNoiseImpulse(),
                        AddNoiseStripe(),
                        AddNoiseDeadline(),
                        AddNoiseComplex(),
                    ]
                )])
    if blind == 'case1':
        transform.extend(
            [
                AddNoiseNoniid([30, 50, 70]),
            ])
    if blind == 'case2':
        transform.extend(
            [
                AddNoiseNoniid([30, 50, 70]),
                AddNoiseStripe()])
    if blind == 'case3':
        transform.extend(
            [
                AddNoiseNoniid([30, 50, 70]),
                AddNoiseDeadline()])
    if blind == 'case4':
        transform.extend(
            [
                AddNoiseNoniid([30, 50, 70]),
                AddNoiseImpulse()])
    if blind == 'case5':
        transform.extend(
            [
                AddNoiseNoniid([30, 50, 70]),
                AddNoiseComplex()])

    transform.append(ToTensor())
    transform = transforms.Compose(transform)
    return transform



def testfile(path, model, device, lock, band, pos, loader, key, transformer, bands=31, crop_square=True,
             normalize=True,
             channel_first=False, showres=True, cropcenter=False):

    device = torch.device('cpu')
    model = model.to(device)
    model = model.eval()

    img = np.array(loader(path)[key])

    # urban & pavia
    if lock == 'urban' or lock == 'pavia':
        img = img[:, :, ::-1]
        img = np.rot90(img, k=2, axes=(-2, -1))
        img = np.fliplr(img)

    if lock == 'urban' or lock == 'cave' or lock == 'icvl':
        img = np.transpose(img, [1, 2, 0]) # chw -> hwc   

    if normalize:
        img = minmax_normalize(img)

    if cropcenter:
        if lock == 'icvl':
            img = crop_center(img, 340, 340, 100, -230) # icvl-case
            # img = crop_center(img, 340, 340) # icvl-gauss

            img = np.flipud(img)

    if lock == 'pavia':
        img = img[:340,:,:]

    noise_copy = None


    if lock == 'urban':
        # noise_image = transformer(np.transpose(img, [2, 0, 1])).float()  # hwc -> chw, then trans
        # noise_copy = noise_image.numpy().copy()
        # noise_copy = np.transpose(noise_copy, [1, 2, 0])  # hwc -> chw

        noise_path = './result_urban/case5-11.mat'
        noise_image = torch.from_numpy(np.array(io.loadmat(noise_path)['data']))    # hwc
        noise_copy = noise_image.numpy().copy() # hwc
        noise_image = np.transpose(noise_image, [ 2, 0, 1]) # hwc -> chw
    elif lock == 'india':
        # real nosie, no need add noise

        noise_copy = img.copy() # hwc
        noise_image = transformer(img if channel_first else np.transpose(img, [2, 0, 1])).float()   # get chw
    elif lock == 'pavia':  
        # noise_image = transformer(np.transpose(img, [2, 0, 1])).float()  # hwc -> chw, then trans
        # noise_copy = noise_image.numpy().copy()
        # noise_copy = np.transpose(noise_copy, [1, 2, 0])  # hwc -> chw

        noise_path = './result_pavia/case5_11.mat'
        noise_image = torch.from_numpy(np.array(io.loadmat(noise_path)['data']))    # hwc
        noise_copy = noise_image.numpy().copy() # hwc
        noise_image = np.transpose(noise_image, [ 2, 0, 1]) # hwc -> chw
    elif lock == 'cave':
        # noise_image = transformer(np.transpose(img, [2, 0, 1])).float()  # hwc -> chw, then trans
        # noise_copy = noise_image.numpy().copy()
        # noise_copy = np.transpose(noise_copy, [1, 2, 0])  # hwc -> chw

        noise_path = './result_cave/gauss-50-14.mat'
        noise_image = torch.from_numpy(np.array(io.loadmat(noise_path)['data']))    # hwc
        noise_copy = noise_image.numpy().copy() # hwc
        noise_image = np.transpose(noise_image, [ 2, 0, 1]) # hwc -> chw
    elif lock == 'icvl':
        # noise_image = transformer(np.transpose(img, [2, 0, 1])).float()  # hwc -> chw, then trans
        # noise_copy = noise_image.numpy().copy()
        # noise_copy = np.transpose(noise_copy, [1, 2, 0])  # hwc -> chw

        noise_path = './result_icvl/case5_12.mat'
        noise_image = torch.from_numpy(np.array(io.loadmat(noise_path)['data']))    # hwc
        noise_copy = noise_image.numpy().copy() # hwc
        noise_image = np.transpose(noise_image, [ 2, 0, 1]) # hwc -> chw       
    else:
        # add noise
        if transformer:
            noise_image = transformer(img if channel_first else np.transpose(img, [2, 0, 1])).float()   # get chw
            img = np.transpose(img, [1, 2, 0]) if channel_first else img    # get hwc
            noise_copy = np.transpose(noise_image, [1, 2, 0]).numpy().copy()
        else:
            noise_image = torch.from_numpy(img).float()

    # calculate mpsnr, mssim, sam
    noisepsnr = mpsnr(img, noise_copy)
    noisessims = mssim(img, noise_copy)
    noisesam = cal_sam(img, noise_copy)
    print('psnr: {:.5f} ssim: {:.5f} sam: {:.5f}'.format(noisepsnr, noisessims, noisesam))


    (h, w, B) = img.shape
    if crop_square:
        length = min(h, w)
        h = length
        w = length

    # get denoise chw
    if B > bands:
        image = [model(noise_image[None, bands * i:bands * (i + 1), :, :])[0] for i in
                 range(noise_image.size(0) // bands)]
        rest = noise_image.size(0) % bands

        if rest > 0:
            depre = model(noise_image[None, -bands:])
        image.append(depre[:, - rest:][0])
        denoise = torch.cat(image, dim=0)
    else:
        denoise = model(noise_image[None])

    denoise = denoise.squeeze().permute(1, 2, 0).detach().numpy()   # chw -> hwc, same as img

    psnrs = mpsnr(img, denoise)
    ssims = mssim(img, denoise)
    sam = cal_sam(img, denoise)
    print('psnr: {:.5f} ssim: {:.5f} sam: {:.5f}'.format(psnrs, ssims, sam))

    if showres:
        # save mat file
        io.savemat(f'./result_{lock}/gt.mat', {'data': img}) # hwc
        if noise_copy is not None:
            io.savemat(f'./result_{lock}/noise.mat', {'data': noise_copy})    # hwc
        io.savemat(f'./result_{lock}/denoise.mat', {'data': denoise}) # hwc


        if lock == 'pavia_grey':
            show_bands = 10
            grey = img[:, :, show_bands]
            cv2.imwrite(f'./result_{lock}/show_gt.jpg',  grey * 255)
            grey = noise_copy[:, :, show_bands]
            cv2.imwrite(f'./result_{lock}/show_noise.jpg',  grey * 255)
            grey = denoise[:, :, show_bands]
            cv2.imwrite(f'./result_{lock}/show_denoise.jpg',  grey * 255)

        elif lock == 'urban' or lock == 'india' or lock == 'cave' or lock == 'icvl' or lock == 'pavia':
            show_bands = [3,13,23] # icvl

            # image RGB
            img = np.stack([img[:, :, show_bands[0]],img[:, :, show_bands[1]],img[:, :, show_bands[2]]],2)
            if lock == 'india':
                img = channel_minmax_normalize(img)
            # ShowEnlargedRectangle(img, show_size, ratio, pos,'gt')
            cv2.imwrite(f'./result_{lock}/show_gt.jpg', img * 255)

            if noise_copy is not None:
                # noise RGB
                noise_copy = np.stack([noise_copy[:, :, show_bands[0]],noise_copy[:, :, show_bands[1]],noise_copy[:, :, show_bands[2]]],2)
                if lock == 'india':
                    noise_copy = channel_minmax_normalize(noise_copy)
                # ShowEnlargedRectangle(noise_copy, show_size, ratio, pos,'noise')
                cv2.imwrite(f'./result_{lock}/show_noise.jpg',noise_copy * 255)

            # denoise RGB
            if lock == 'india':
                denoise = channel_minmax_normalize(denoise)
            denoise = np.stack([denoise[:, :, show_bands[0]],denoise[:, :, show_bands[1]],denoise[:, :, show_bands[2]]],2)
            # ShowEnlargedRectangle(denoise, show_size, ratio, pos,'denoise')
            cv2.imwrite(f'./result_{lock}/show_denoise.jpg',denoise * 255)

        


"""
The img is processed and the patchsize size block is selected at the upleft position (given the top left coordinate point) for enlargement (magnification is ratio) and displayed in the bottom right corner.
"""

def ShowEnlargedRectangle(img, patchsize, ratio, upleft, name):

    if len(img.shape) == 3:
        x, y, _ = img.shape
    else:
        x, y = img.shape
    # img = img * 255
    enlagesize = patchsize * ratio

    # get patch block
    patch = img[upleft[1]:upleft[1] + patchsize, upleft[0]:upleft[0] + patchsize,:]

    # Zoom in on the patch block and place it in the lower right corner
    cv2.rectangle(img, upleft, (upleft[0] + patchsize, upleft[1] + patchsize), (1, 0, 0), 1)
    patch = cv2.resize(patch, dsize=(enlagesize, enlagesize))
    img[x - enlagesize:, y - enlagesize:,:] = patch

    cv2.rectangle(img, (y - enlagesize - 1, x - enlagesize - 1), (y - 1, x - 1), (1, 0, 0), 1)

    plt.axis(False)
    # img = np.clip(img, a_min=0.0, a_max=1.0)
    plt.imshow(img)
    plt.savefig(f'./result/show_{name}.jpg',bbox_inches='tight',pad_inches=0.0)
    # plt.show()


def mpsnr(X, Y, channel_first=True):
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    if channel_first:
        X = np.transpose(X, (1, 2, 0))
        Y = np.transpose(Y, (1, 2, 0))
    return np.mean([psnr(X[:, :, i], Y[:, :, i], data_range=1) for i in range(X.shape[-1])])


def mssim(X, Y, channel_first=True):
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    if channel_first:
        X = np.transpose(X, (1, 2, 0))
        Y = np.transpose(Y, (1, 2, 0))
    return np.mean([ssim(X[:, :, i], Y[:, :, i], data_range=1) for i in range(X.shape[-1])])


def show(img, deimg, noimg, save_fig, figsize=(12, 4), title=None):
    fig = plt.figure(figsize=figsize)
    ax1, ax2, ax3 = fig.subplots(1, 3)
    ax1.set_title('ground truth', fontsize=30)
    ax1.set_axis_off()
    ax1.imshow(img, cmap='gray')
    ax2.set_title('denoise result', fontsize=30)
    ax2.set_axis_off()
    ax2.imshow(deimg, cmap='gray')
    ax3.set_title('hsi_noise', fontsize=30)
    ax3.set_axis_off()
    ax3.imshow(noimg, cmap='gray')
    if save_fig:
        plt.savefig(f'./fig/{title}.png')
    plt.show()



def plot_test(model, testloader, device, n_img=4):
    it = iter(testloader)
    for i in range(1):
        it.next()
    im = it.next()
    denoise_gt, noise_image = im
    denoise_gt, noise_image = denoise_gt.float().to(device), noise_image.float().to(device)
    model = model.eval()
    for i in range(n_img):
        img = denoise_gt[i:i + 1]
        noise_img = noise_image[i:i + 1]
        denoise_pre, noise_pre = model(noise_img)
        fig = plt.figure(figsize=(6, 2))
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.set_title('ground truth')
        ax1.set_axis_off()
        ax1.imshow(img.cpu().numpy()[0, 9], cmap='gray')
        ax2.set_title('denoise result')
        ax2.set_axis_off()
        ax2.imshow(denoise_pre.detach().cpu().numpy()[0, 9], cmap='gray')
        ax3.set_title('hsi_noise')
        ax3.set_axis_off()
        ax3.imshow(noise_img.detach().cpu().numpy()[0, 9], cmap='gray')
        plt.savefig(f'./fig/{i}.png')



def minmax_normalize(array):
    # H W C

    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)



def channel_minmax_normalize(array, channel_first=False):
    # H W C
    if channel_first:
        # array = array.numpy()
        c, h, w = array.shape
        result = np.zeros((c, h, w))

        for i in range(c):
            data = array[i,:,:]
            _range = (np.max(data) - np.min(data)).astype(np.float64)
            result[i,:,:] =  (array[i,:,:] - np.min(data)) / _range 
        result = torch.from_numpy(result)

    else:
        h, w, c = array.shape
        result = np.zeros((h, w, c))

        for i in range(c):
            data = array[:,:,i]
            _range = (np.max(data) - np.min(data)).astype(np.float64)
            result[:,:,i] =  (array[:,:,i] - np.min(data)) / _range 
    return result


class ToTensor(object):
    def __call__(self, data):
        return torch.from_numpy(data)


def crop_center(img, cropx, cropy, set_x = 0, set_y = 0):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2) + set_x
    starty = y // 2 - (cropy // 2) + set_y
    return img[starty:starty + cropy, startx:startx + cropx, :]


def cal_loss(gt_image, pre_de, device):
    deloss = torch.mean(torch.abs(gt_image - pre_de))
    grad_loss = cal_grad_loss(gt_image, pre_de, device)
    return deloss, grad_loss


def cal_sam(X, Y, eps=1e-8):
    # print(X,Y)
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    tmp = (np.sum(X * Y, axis=0) + eps) / (np.sqrt(np.sum(X ** 2, axis=0)) + eps) / (
            np.sqrt(np.sum(Y ** 2, axis=0)) + eps)

    return np.mean(np.real(np.arccos(tmp)))


def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :]

    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)
    if random.random() < 0.5:
        image = image[::-1, :, :]

    return np.ascontiguousarray(image)



def cal_grad_loss(gt_image, pre_de, device):

    _, c_gt, _, _ = gt_image.size()
    _, c_pre, _, _ = pre_de.size()
    gt_data = [gt_image.permute(0,1,2,3), gt_image.permute(0,1,3,2), gt_image.permute(0,2,1,3)]
    pre_data = [pre_de.permute(0,1,2,3), pre_de.permute(0,1,3,2), pre_de.permute(0,2,1,3)]

    # Sobel operator
    kernel = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel, requires_grad=False).to(device)

    grad_sum = 0
    for i in range(3):
        gt = gt_data[i].sum(dim=1, keepdim=True) / c_gt
        pre = pre_data[i].sum(dim=1, keepdim=True) / c_pre

        grad_gt = F.conv2d(gt, weight)
        grad_pre = F.conv2d(pre, weight)       

        grad_sum += torch.mean(torch.abs(torch.norm(grad_gt - grad_pre, p=2)))

    return grad_sum



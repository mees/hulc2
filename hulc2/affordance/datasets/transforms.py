import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from hulc2.utils.img_utils import pixel_after_pad, resize_pixel


class ScaleImageTensor(object):
    """Scale tensor of shape (batch, C, H, W) containing images to [0, 255] range
    Args:
        tensor (torch.tensor): Tensor to be scaled.
    Returns:
        Tensor: Scaled tensor.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor.float().div(255)

class ToIntRange(object):
    """Scale tensor of shape (batch, C, H, W) containing images to [0, 1] range
    Args:
        tensor (torch.tensor): Tensor to be scaled.
    Returns:
        Tensor: Scaled tensor.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor.mul(255)

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, clip=None):
        self.std = torch.Tensor(std)
        self.mean = torch.Tensor(mean)
        self.clip = clip

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        t = tensor + torch.randn(tensor.size()) * self.std + self.mean
        if self.clip:
            t.clamp(self.clip[0], self.clip[1])
        return t

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class ThresholdMasks(object):
    def __init__(self, threshold):
        # Mask is between 0 and 255
        self.threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Convert to 0-1
        assert isinstance(tensor, torch.Tensor)
        return (tensor > self.threshold).long()


class NormalizeVector(object):
    """Normalize a tensor vector with mean and standard deviation."""

    def __init__(self, mean=[0.0], std=[1.0]):
        self.std = torch.Tensor(std)
        # self.std[self.std == 0.0] = 1.0
        self.mean = torch.Tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)

class NormalizeVectorInverse(NormalizeVector):
    """Undo normalization Normalize a tensor vector with mean and standard deviation."""

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-10)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)

class ColorTransform(object):
    '''
        Applies color jitter transform to image
    '''
    def __init__(self, contrast=0.3, brightness=0.3, hue=0.3, prob=0.3):
        super().__init__()
        self.prob = prob
        self.jitter = T.ColorJitter(contrast=contrast, brightness=brightness, hue=hue)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
            Assumes input tensor is between 0-1
        '''
        assert isinstance(tensor, torch.Tensor)
        apply = np.random.rand() < self.prob
        if apply:
            tensor = self.jitter(tensor)
        return tensor


class RandomCrop(object):
    def __init__(self, size: int, rand_crop: float) -> None:
        super().__init__()
        _size = int(size * rand_crop)
        self.orig_size = size
        self.crop = torchvision.transforms.RandomCrop(_size)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        tensor = self.crop(tensor)
        return tensor


class DistanceTransform(object):
    """Apply distance transform to binary mask (0, 1)
    mask.shape = [C, W, H]
    mask.max() = 1
    mask.min() = 0
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        tensor = tensor.permute((1, 2, 0))  # C, W, H -> W, H, C
        mask = tensor.detach().cpu().numpy().astype(np.uint8)
        # cv2.imshow("in", mask*255)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        dist_np = np.array(dist)
        std_g = 2
        gauss_im = 1 / (std_g * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((1 - dist_np) / std_g) ** 2)
        gauss_im = cv2.normalize(gauss_im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask = (gauss_im * 255).astype(np.uint8)  # H, W
        # cv2.imshow("out", mask)
        # cv2.waitKey(1)
        mask = torch.from_numpy(mask).unsqueeze(0)  # 1, H, W
        return mask


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, inp):
        x = inp["img"]
        center = inp["center"]

        shape = x.shape  # C, H, W
        if len(shape) == 3:
            x = x.unsqueeze(0)
            center = np.expand_dims(center, 0)
            _test = torch.zeros_like(x)
            _test[0, :, center[0,0], center[0,1]] = torch.tensor((2,4,6))
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)

        x = F.pad(x, padding, 'replicate')
        padded_center = pixel_after_pad(center, padding)  # 0-shape

        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift

        grid_sample = F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

        # from -1 to 1 in input shape(x)
        _center = ((padded_center[:, ::-1] / x.shape[-1] ) * 2) - 1

        # Grid is of output shape but maps from x... contains
        # coords in x
        np_grid = grid[0].cpu().numpy()
        distances = np.linalg.norm(np_grid.reshape((-1,2)) - _center, axis=1)
        min_dist = np.argmin(distances)
        center = np.unravel_index(min_dist, np_grid.shape[:2])

        center = resize_pixel(center, grid_sample.shape[-2:], shape[-2:])  
        grid_sample = grid_sample.reshape(shape)
        return grid_sample, center

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    from mean and std
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-10)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
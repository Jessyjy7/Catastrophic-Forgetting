import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import _is_tensor_image
from .functional import to_byte_tensor


__all__ = ['ToCPUTensor', 'ToGPUTensor', 'Normalize', 'RandomHorizontalFlip', 'RandomCrop']


class ToCPUTensor(object):
    """Convert a ``PIL Image``, a ``numpy.ndarray`` or a ``torch.ByteTensor`` to tensor.

    Converts a torch.ByteTensor in the range
    [0, 255] to a torch.FloatTensor of shape in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            img: Image to be converted to tensor.

        Returns:
            Tensor: Converted ByteTensor.
        """

        if not isinstance(img, torch.ByteTensor):
            img = to_byte_tensor(img)

        return img.float().div(255)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToGPUTensor(object):
    """Convert a ``PIL Image``, a ``numpy.ndarray`` or a ``torch.ByteTensor`` to GPU tensor.

    Converts a torch.ByteTensor in the range
    [0, 255] to a GPU torch.FloatTensor of shape in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            img: Image to be converted to tensor.

        Returns:
            Tensor: Converted ByteTensor.
        """

        if not isinstance(img, torch.ByteTensor):
            img = to_byte_tensor(img)

        img = img.cuda()

        return img.float().div(255)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)

        return img


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if not _is_tensor_image(img) :
            raise TypeError('pic should be Tensor. Got {}.'.format(type(img)))
        k = np.random.uniform(0, 1, 1)
        if k > self.p:
            img = torch.flip(img, [2])

        return img

    def __repr__(self):
        print('RandomHorizontalFlip(' + self.p + +')')


class RandomCrop(object):
    def __init__(self, output_size, padding=0):
        self.output_size = output_size
        self.padding = 4*[padding]

    def __call__(self, img):
        if not _is_tensor_image(img) :
            raise TypeError('pic should be Tensor. Got {}.'.format(type(img)))

        img = F.pad(img, self.padding)
        h, w = img.shape[1:3]
        th = tw = self.output_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return img[..., i:i+th, j:j+tw].contiguous()

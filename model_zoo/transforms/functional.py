import torch
import numpy as np
try:
    import accimage
except ImportError:
    accimage = None
from torchvision.transforms import functional as F


def to_byte_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to byte tensor.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to byte tensor.

    Returns:
        ByteTensor: Converted image.
    """
    if not(F._is_pil_image(pic) or F._is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.uint8)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    if pic.mode != 'RGB':
        raise ValueError('pic mode should be RGB. Got', pic.mode)
    # handle RGB PIL Image
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()

    return img

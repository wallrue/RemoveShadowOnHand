import math
import random
from PIL import Image, ImageOps
import numbers
import types
import torchvision.transforms as transforms

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, newSize, padding=0):
        
        self.newSize = (int(newSize), int(newSize)) if isinstance(newSize, numbers.Number) else newSize
        self.padding = padding

    def __call__(self, img):
        
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
       
        w, h = img.size
        th, tw = self.newSize
        if w == tw and h == th:
            output = img
        else: 
            x1 = random.randint(0,max(0,w-tw-1))
            y1 =  random.randint(0,max(0,h-th-1))
            output = img.crop((x1, y1, x1 + tw, y1 + th))

        return output
    
class Resize(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.Resampling.NEAREST):
        self.size = (int(size), int(size)) if isinstance(size, numbers.Number) else size
        self.interpolation = interpolation

    def __call__(self, img):
        output = img.resize(self.size, self.interpolation)
            
        return output
    
class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.Resampling.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            output = img
        elif w < h:
            ow = self.size
            oh = int(self.size * h / w)    
            output = img.resize((ow, oh), self.interpolation)   
        else:
            oh = self.size
            ow = int(self.size * w / h)
            output = img.resize((ow, oh), self.interpolation)
            
        return output

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, img):
        flag = random.random() < 0.5
        if flag:
            output = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            output = img
        return output
    
class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = (tensor - self.mean)/self.std
        return tensor
    
def get_transform_list(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop': # resize to loadSize and crop to fineSize
        transform_list.append(Resize(opt.loadSize))
        transform_list.append(RandomCrop(newSize = opt.fineSize))
    elif opt.resize_or_crop == 'resize': # Resize to fineSize
        transform_list.append(Resize(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width': # scale to fineSize for width only
        transform_list.append(Scale(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width_and_crop': # scale to fineSize for width only and crop to fineSize
        transform_list.append(Scale(opt.loadSize))
        transform_list.append(RandomCrop(newSize = opt.fineSize))
    elif opt.resize_or_crop == 'none': # just modify the width and height to be multiple of 4
        pass
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    # training time configuration
    if opt.isTrain:
        if not opt.no_flip:
            transform_list.append(RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(0.5, 0.5)]
    return transform_list
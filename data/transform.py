class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            output = img
        else: 
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
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
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        output = img.resize((w, h), self.interpolation)
            
        return output
    
class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.Resampling.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            output = img
            continue
        
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            output = img.resize((ow, oh), self.interpolation)
            continue
        else:
            oh = self.size
            ow = int(self.size * w / h)
            output = img.resize((ow, oh), self.interpolation)
            
        return output
    
    
def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop': # resize to loadSize and crop to fineSize
        transform_list.append(Resize(opt.loadSize))
        transform_list.append(RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop': # crop to fineSize
        transform_list.append(RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width': # scale to fineSize for width only
        transform_list.append(Scale(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width_and_crop': # scale to fineSize for width only and crop to fineSize
        transform_list.append(Scale(opt.fineSize))
        transform_list.append(RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'none': # just modify the width and height to be multiple of 4
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    if opt.isTrain and not opt.no_flip: # flip if running in training time only
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transform_list #transforms.Compose(transform_list)
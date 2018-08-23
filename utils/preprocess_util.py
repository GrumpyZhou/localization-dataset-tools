import math
import os
import numpy as np
from PIL import Image

__all__ = ['resize_', 'crop_and_pad', 'resize_datasets']

def resize_datasets(base_dir, datasets, resize, save_base_dir):
    for dataset in datasets:
        data_dir = os.path.join(base_dir, dataset)
        print('Resize dataset {} to size {}'.format(data_dir, resize))        
        save_dir = os.path.join(save_base_dir, dataset)
                
        dt_txt = os.path.join(data_dir, 'dataset_train.txt')
        num = 0 
        with open(dt_txt, 'r') as f:
            # im x y z w p q r
            for i,line in enumerate(f):
                if i < 3:
                    continue
                im_name = line.split()[0]
                im = Image.open(os.path.join(data_dir, im_name))
                pdir = os.path.dirname(os.path.join(save_dir, im_name))
                if not os.path.exists(pdir):
                    os.makedirs(pdir)
                    print('mkdir {}'.format(pdir))
                im = resize_(im, resize, interpolation=Image.BICUBIC)
                im.save(os.path.join(save_dir, im_name))
                num += 1
            print('Processed train images {}'.format(num))

        dt_txt = os.path.join(data_dir, 'dataset_test.txt')
        num = 0 
        with open(dt_txt, 'r') as f:
            # im x y z w p q r
            for i,line in enumerate(f):
                if i < 3:
                    continue
                im_name = line.split()[0]
                im = Image.open(os.path.join(data_dir, im_name))
                pdir = os.path.dirname(os.path.join(save_dir, im_name))
                if not os.path.exists(pdir):
                    os.makedirs(pdir)
                    print('mkdir {}'.format(pdir))
                im = resize_(im, resize, interpolation=Image.BICUBIC)
                im.save(os.path.join(save_dir, im_name))
                num += 1
            print('Processed test images {}'.format(num))

def resize_(im, size, interpolation=Image.BICUBIC):
    # size is either an int or a tuple (w, h)
    if isinstance(size, int):
        w, h = im.size
        if (w <= h and w == size) or (h <= w and h == size):
            return im
        if w < h:
            ow = size
            oh = int(size * h / w)
            return im.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return im.resize((ow, oh), interpolation)
    else:
        return im.resize(size, interpolation)
    
def crop_and_pad(im, out_width, out_height):
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    left = int(math.floor((im.width - out_width) / 2))
    right = left + out_width
    up = int(math.floor((im.height - out_height) / 2))
    down = up + out_height
    im = im.crop((left, up, right, down))
    return im

"""
def center_crop_numpy(img, out_width, out_height):
    '''Perform center cropping on a numpy image array'''
    left = int(math.floor((img.shape[1] - out_width) / 2))
    right = left + out_width
    up = int(math.floor((img.shape[0] - out_height) / 2))
    down = up + out_height
    return img[ up:down, left:right, :]
"""
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread


class KITTIrawDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        return left_images, right_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')
    
    def load_mask(self, filename):
        data = Image.open(filename).convert('L')
        return np.ascontiguousarray(data, dtype=np.float32)

    def load_disp(self, filename):
        # data, scale = pfm_imread(filename)
        data = np.load(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.training:
            w, h = left_img.size
            # crop_w, crop_h = 512, 256
            crop_w, crop_h = 576, 288

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img}
        else:
            w, h = left_img.size

            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            top_pad = (32-h%32)%32
            right_pad = (32-w%32)%32

            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            
            # crop_w, crop_h = 960, 512
            # left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            # right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            # disparity = disparity[h - crop_h:h, w - crop_w: w]

            return {"left": left_img,
                    "right": right_img,
                    "top_pad": top_pad,
                    "right_pad": right_pad,
                    "left_filename": self.left_filenames[index]}

import PIL
from PIL import Image
import math
import numpy as np
from skimage.util import random_noise
from glob import glob
import os
from pathlib import Path
from tqdm import tqdm


data_exts = ["*.jpg", "*.jpeg", "*.png"]
img_dir = "D:/Dataset/CelebA/img_align_celeba/img_align_celeba"
lr_img_dir = "D:/Dataset/CelebA/img_align_celeba/img_align_celeba_LR"


def bicubic_downsample(im, factor=4):
    width, height = im.size
    new_width = int(math.floor(width / factor))
    new_height = int(math.floor(height / factor))
    im = im.resize((new_width, new_height), resample=PIL.Image.BICUBIC)
    return im


def generate_noise(im):
    im = np.array(im) / 255
    im = random_noise(im, var=2e-3)
    im = random_noise(im, mode="s&p", amount=2e-3)
    im = Image.fromarray(np.multiply(im, 255).astype("uint8"), "RGB")
    return im


def generate_lr(img_path, img_save_dir):
    im = Image.open(img_path)
    im = bicubic_downsample(im, 4)
    im = generate_noise(im)
    img_path_name = img_path.split(".")[0]
    img_name = Path(img_path).stem
    im.save(os.path.join(img_save_dir, img_name + ".jpg"))


def main():
    if not os.path.exists(lr_img_dir):
        os.makedirs(lr_img_dir)
    imgs_path = []
    for ext in data_exts:
        imgs_path.extend(glob(os.path.join(img_dir, ext)))

    print("Number of pics to be downsampled {0}".format(len(imgs_path)))
    for img_path in tqdm(imgs_path):
        generate_lr(img_path, lr_img_dir)


if __name__ == "__main__":
    main()
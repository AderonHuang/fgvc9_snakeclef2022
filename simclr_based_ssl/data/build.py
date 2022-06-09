# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import pickle
import glob
import copy
import random
import torch
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
# from timm.data.transforms import _pil_interp

from augly.image import (EncodingQuality, OneOf,
                         RandomBlur, RandomEmojiOverlay, RandomPixelization,
                         RandomRotation, ShufflePixels)

from augly.image.functional import overlay_emoji, overlay_image, overlay_text
from augly.image.transforms import BaseTransform
from augly.utils import pathmgr
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils.constants import FONT_LIST_PATH, FONTS_DIR, SMILEY_EMOJI_DIR


from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from .dataset_fg import DatasetMeta

global input_size

# input_size = 384
input_size = 512

train_paths = glob.glob('/mnt/workspace/booyoungxu.xfr/data/copyright/vision_china/train/*.jpg')


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
#             root = os.path.join(config.DATA.DATA_PATH, prefix)
            root = './datasets/imagenet'
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'inaturelist2021':
        root = './datasets/inaturelist2021'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET,
                             class_ratio=config.DATA.CLASS_RATIO,per_sample=config.DATA.PER_SAMPLE)
        nb_classes = 10000
    elif config.DATA.DATASET == 'inaturelist2021_mini':
        root = './datasets/inaturelist2021_mini'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 10000
    elif config.DATA.DATASET == 'inaturelist2017':
        root = './datasets/inaturelist2017'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 5089
    elif config.DATA.DATASET == 'inaturelist2018':
        root = './datasets/inaturelist2018'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 8142
    elif config.DATA.DATASET == 'cub-200':
        root = './datasets/cub-200'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 200
    elif config.DATA.DATASET == 'stanfordcars':
        root = './datasets/stanfordcars'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 196
    elif config.DATA.DATASET == 'oxfordflower':
        root = './datasets/oxfordflower'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 102
    elif config.DATA.DATASET == 'stanforddogs':
        root = './datasets/stanforddogs'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 120
    elif config.DATA.DATASET == 'nabirds':
        root = './datasets/nabirds'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 555
    elif config.DATA.DATASET == 'aircraft':
        root = './datasets/aircraft'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 100
    elif config.DATA.DATASET == 'snakeclef2022':
        root = './datasets/snakeclef2022'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET,over_sampling=config.OVER_SAMPLING)
        nb_classes = 1572
    elif config.DATA.DATASET == 'snakeclef2022trainvalall':
        root = './datasets/snakeclef2022'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET,over_sampling=config.OVER_SAMPLING)
        nb_classes = 1572
    elif config.DATA.DATASET == 'snakeclef2022test':
        root = './datasets/snakeclef2022'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 1572
    elif config.DATA.DATASET == 'snakeclef2022valid':
        root = './datasets/snakeclef2022'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 1572
    elif config.DATA.DATASET == 'snakeclef2022all':
        root = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 1572
    else:
        raise NotImplementedError("We only support ImageNet and inaturelist.")

    return dataset, nb_classes


class NCropsTransform:
    """Take n random crops of one image as the query and key."""

    def __init__(self, aug_moderate, aug_hard, ncrops=2):
        self.aug_moderate = aug_moderate
        self.aug_hard = aug_hard
        self.ncrops = ncrops

    def __call__(self, x):
        # a = [self.aug_moderate(x)]
        # b = [self.aug_hard(x) for _ in range(self.ncrops - 1)]
        # print('self.aug_moderate(x)', self.aug_moderate(x).shape, self.aug_hard(x).shape)

        return torch.cat((self.aug_moderate(x), self.aug_hard(x)), dim=0)
        # return [self.aug_moderate(x)] + [self.aug_hard(x) for _ in range(self.ncrops - 1)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomOverlayText(BaseTransform):
    def __init__(
        self,
        opacity: float = 1.0,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.opacity = opacity
        # print('FONTS_DIR',  FONT_LIST_PATH)

        # with open(Path(FONTS_DIR) / FONT_LIST_PATH) as f:
        with open(FONT_LIST_PATH) as f:
            font_list = [s.strip() for s in f.readlines()]
            blacklist = [
                'TypeMyMusic',
                'PainttheSky-Regular',
            ]
            self.font_list = [
                f for f in font_list
                if all(_ not in f for _ in blacklist)
            ]

        self.font_lens = []
        for ff in self.font_list:
            # font_file = Path(MODULE_BASE_DIR) / ff.replace('.ttf', '.pkl')
            font_file = MODULE_BASE_DIR +'/' +ff.replace('.ttf', '.pkl')
            # print('font_file', font_file)
            with open(font_file, 'rb') as f:
                self.font_lens.append(len(pickle.load(f)))

    def apply_transform(
        self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        i = random.randrange(0, len(self.font_list))
        kwargs = dict(
            font_file=Path(MODULE_BASE_DIR) / self.font_list[i],
            font_size=random.uniform(0.1, 0.3),
            color=[random.randrange(0, 256) for _ in range(3)],
            x_pos=random.uniform(0.0, 0.5),
            metadata=metadata,
            opacity=self.opacity,
        )
        try:
            for j in range(random.randrange(1, 3)):
                if j == 0:
                    y_pos = random.uniform(0.0, 0.5)
                else:
                    y_pos += kwargs['font_size']
                image = overlay_text(
                    image,
                    text=[random.randrange(0, self.font_lens[i]) for _ in range(random.randrange(5, 10))],
                    y_pos=y_pos,
                    **kwargs,
                )
            return image
        except OSError:
            return image


class RandomOverlayImageAndResizedCrop(BaseTransform):
    def __init__(
        self,
        img_paths: List[Path],
        opacity_lower: float = 0.5,
        size_lower: float = 0.4,
        size_upper: float = 0.6,
        input_size: int = 224,
        moderate_scale_lower: float = 0.7,
        hard_scale_lower: float = 0.15,
        overlay_p: float = 0.05,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.img_paths = img_paths
        self.opacity_lower = opacity_lower
        self.size_lower = size_lower
        self.size_upper = size_upper
        self.input_size = input_size
        self.moderate_scale_lower = moderate_scale_lower
        self.hard_scale_lower = hard_scale_lower
        self.overlay_p = overlay_p

    def apply_transform(
        self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:

        raw_image = copy.deepcopy(image)

        if random.uniform(0.0, 1.0) < self.overlay_p:
            if random.uniform(0.0, 1.0) > 0.5:
                background = Image.open(random.choice(self.img_paths))
                background = background.convert('RGB')
                if random.uniform(0.0, 1.0) > 0.5:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    background = Image.new("RGB", (background.size), color)
                overlay = image
            else:
                background = image
                overlay = Image.open(random.choice(self.img_paths))
                overlay = overlay.convert('RGB')

            overlay_size = random.uniform(self.size_lower, self.size_upper)
            image = overlay_image(
                background,
                overlay=overlay,
                opacity=random.uniform(self.opacity_lower, 1.0),
                overlay_size=overlay_size,
                x_pos=random.uniform(0.0, 1.0 - overlay_size),
                y_pos=random.uniform(0.0, 1.0 - overlay_size),
                metadata=metadata,
            )
            if image.size[0] == 0 or image.size[1] == 0:
                image = raw_image

            return transforms.RandomResizedCrop(self.input_size, scale=(self.moderate_scale_lower, 1.))(image)
        else:
            return transforms.RandomResizedCrop(self.input_size, scale=(self.hard_scale_lower, 1.))(image)


class RandomEmojiOverlay(BaseTransform):
    def __init__(
        self,
        emoji_directory: str = SMILEY_EMOJI_DIR,
        opacity: float = 1.0,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.emoji_directory = emoji_directory
        self.emoji_paths = pathmgr.ls(emoji_directory)
        self.opacity = opacity

    def apply_transform(
        self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        emoji_path = random.choice(self.emoji_paths)
        return overlay_emoji(
            image,
            emoji_path=os.path.join(self.emoji_directory, emoji_path),
            opacity=self.opacity,
            emoji_size=random.uniform(0.1, 0.3),
            x_pos=random.uniform(0.0, 1.0),
            y_pos=random.uniform(0.0, 1.0),
            metadata=metadata,
        )


class RandomEdgeEnhance(BaseTransform):
    def __init__(
        self,
        mode=ImageFilter.EDGE_ENHANCE,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.mode = mode

    def apply_transform(self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        return image.filter(self.mode)


class RandomHue(BaseTransform):
    def __init__(self,
        min_val: float = 0.1, 
        max_val: float = 0.4,
        p: float = 1.0):
        """
        @param factor: a saturation factor of below 1.0 lowers the saturation, a
            factor of 1.0 gives the original image, and a factor greater than 1.0
            adds saturation

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.min_val = min_val
        self.max_val = max_val
    

    def apply_transform(
        self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        self.factor = random.random() * (self.max_val - self.min_val) + self.min_val
        im = image.copy()
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        layer = Image.new('RGB', im.size, color) # "hue" selection is done by choosing a color...
        output = Image.blend(im, layer, self.factor)
        return  output


class RandomFilers(BaseTransform):
    def __init__(self,
        p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
    

    def apply_transform(
        self,
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        image = np.asarray(image.convert('RGB'))
        image = image[:,:,::-1]
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flag = random.randint(1, 12)
        if flag == 1:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_AUTUMN)
        elif flag == 2:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_COOL)
        elif flag == 3:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_SUMMER)
        elif flag == 4:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_PINK)
        elif flag == 5:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_SPRING)
        elif flag == 6:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_OCEAN)
        elif flag == 7:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_BONE)
        elif flag == 8:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        elif flag == 9:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_WINTER)
        elif flag == 10:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_RAINBOW)
        elif flag == 11:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_HSV)
        elif flag == 12:
            image = cv2.applyColorMap(im_gray, cv2.COLORMAP_HOT)
        image =Image.fromarray(image)
        return image


class RandomDouyin(BaseTransform):
    def __init__(self,
        p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
    

    def apply_transform(
        self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        
        img = image.convert('RGBA')
        img_arr = np.array(img)
        # 提取R
        img_arr_r = copy.deepcopy(img_arr)
        img_arr_r[:, :, 1:3] = 0
        # 提取GB
        img_arr_gb = copy.deepcopy(img_arr)
        img_arr_gb[:, :, 0] = 0
        # 创建画布把图片错开放
        img_r = Image.fromarray(img_arr_r).convert('RGBA')
        img_gb = Image.fromarray(img_arr_gb).convert('RGBA')
        canvas_r = Image.new('RGB', img.size, color=(0, 0, 0))
        canvas_gb = Image.new('RGB', img.size, color=(0, 0, 0))
        canvas_r.paste(img_r, (6, 6), img_r)
        ###canvas_r.paste(img_r, (36, 36), img_r)
        canvas_gb.paste(img_gb, (0, 0), img_gb)
        img_douyin = Image.fromarray(np.array(canvas_gb) + np.array(canvas_r))
        return  img_douyin


class RandomDouyin1(BaseTransform):
    def __init__(self,
        min_val: float = 0.1, 
        max_val: float = 0.4,
        p: float = 1.0):
        """
        @param factor: a saturation factor of below 1.0 lowers the saturation, a
            factor of 1.0 gives the original image, and a factor greater than 1.0
            adds saturation

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__( p)
    

    def apply_transform(
        self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        
        """
        Alters the saturation of an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: Augmented PIL Image
        """
        img = image.convert('RGBA')
        img_arr = np.array(img)
        # 提取R
        img_arr_r = copy.deepcopy(img_arr)
        img_arr_r[:, :, 1] = 0
        img_arr_r[:, :, 2] = 0
        ### img_arr_r = 255 - img_arr_r
        # 提取G
        img_arr_g = copy.deepcopy(img_arr)
        img_arr_g[:, :, 0] = 0
        img_arr_g[:, :, 2] = 0
        ### img_arr_g = 255 - img_arr_g
        # 提取B
        img_arr_b = copy.deepcopy(img_arr)
        img_arr_b[:, :, 0] = 0
        img_arr_b[:, :, 1] = 0
        ### img_arr_b = 255 - img_arr_b
        # 创建画布把图片错开放
        img_r = Image.fromarray(img_arr_r).convert('RGBA')
        img_g = Image.fromarray(img_arr_g).convert('RGBA')
        img_b = Image.fromarray(img_arr_b).convert('RGBA')
        canvas_r = Image.new('RGB', img.size, color=(0, 0, 0))
        canvas_g = Image.new('RGB', img.size, color=(0, 0, 0))
        canvas_b = Image.new('RGB', img.size, color=(0, 0, 0))
        canvas_r.paste(img_r, (0, 0))
        canvas_g.paste(img_g, (10,10))
        canvas_b.paste(img_b, (20, 20))
        img_douyin = Image.fromarray(np.array(canvas_r) + np.array(canvas_g) + np.array(canvas_b))
        return  img_douyin

class ConvertColor(BaseTransform):
    def __init__(self, p: float = 1.0):
        """
        @param degrees: the amount of degrees that the original image will be rotated
            counter clockwise

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)

    def apply_transform(
        self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        """
        Rotates the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: Augmented PIL Image
        """
        image = 255 - np.asarray(image)

        return Image.fromarray(np.array(image))


class FadeIn(BaseTransform):
    def __init__(self, p: float = 1.0):
        """
        @param degrees: the amount of degrees that the original image will be rotated
            counter clockwise

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)

    def apply_transform(
        self, 
        image: Image.Image, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None
    ) -> Image.Image:
        """
        Rotates the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: Augmented PIL Image
        """
        fading = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        p = random.choice(fading)
        image = np.uint8(p * np.asarray(image))

        return Image.fromarray(np.array(image))

class ShuffledAug:

    def __init__(self, aug_list):
        self.aug_list = aug_list

    def __call__(self, x):
        # without replacement
        shuffled_aug_list = random.sample(self.aug_list, len(self.aug_list))
        for op in shuffled_aug_list:
            x = op(x)
        return x


def convert2rgb(x):
    return x.convert('RGB')



aug_moderate = [
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        convert2rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]

aug_list = [
        transforms.ColorJitter(0.7, 0.7, 0.7, 0.2),
        RandomPixelization(p=0.25),
        ShufflePixels(factor=0.1, p=0.25),
        OneOf([EncodingQuality(quality=q) for q in [10, 20, 30, 50]], p=0.25),
        transforms.RandomGrayscale(p=0.25),
        RandomBlur(p=0.25),
        transforms.RandomPerspective(p=0.25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        # RandomOverlayText(p=0.25),
        # RandomEmojiOverlay(p=0.25),
        # RandomHue(p=0.25),
        # RandomFilers(p=0.25),
        # RandomDouyin1(p=0.25),
        # RandomDouyin(p=0.25),
        # ConvertColor(p=0.25),
        # FadeIn(p=0.25),
        OneOf([RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE), RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE_MORE)], p=0.25),
    ]

   
aug_hard = [
        RandomRotation(p=0.25),
        # RandomOverlayImageAndResizedCrop(
        #     train_paths, opacity_lower=0.6, size_lower=0.4, size_upper=0.6,
        #     input_size=input_size, moderate_scale_lower=0.7, hard_scale_lower=0.15, overlay_p=0.05, p=1.0,
        # ),
        ShuffledAug(aug_list),
        # RandomOverlayText(p=0.25),
        convert2rgb,
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.RandomErasing(value='random', p=0.25),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
s = 1.0

color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
data_transforms = transforms.Compose([
    transforms.RandomApply([transforms.RandomResizedCrop(input_size, scale=(0.2, 1)),],p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * input_size)),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.RandomErasing(value='random', p=0.25),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

# data_transforms = transforms.Compose([
#     transforms.RandomApply([transforms.Pad(10)],p=0.5),
#     transforms.RandomApply([transforms.RandomResizedCrop(input_size, scale=(0.6, 0.9))],p=0.5),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomApply([color_jitter], p=0.8),
#     transforms.RandomGrayscale(p=0.3),
#     GaussianBlur(kernel_size=int(0.1 * input_size)),
#     transforms.Resize((input_size, input_size)),
#     transforms.ToTensor(),
#     transforms.RandomErasing(value='random', p=0.25),
#     transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
#     ])

ssl_transform = NCropsTransform(
        data_transforms, #transforms.Compose(aug_moderate),
        data_transforms, # transforms.Compose(aug_hard),
        2,
)
      
def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.TRAIN_INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)

        ssl_transforms = NCropsTransform(
            transform, #transforms.Compose(aug_moderate),
            data_transforms, # transforms.Compose(aug_hard),
            2,
        )
        return ssl_transforms

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)




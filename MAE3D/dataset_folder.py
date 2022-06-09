import torch
from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import os.path
import random
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_GEBD_dataset(
        pkl_data: {},
        is_train: bool,
        only_reverser: bool = False
) -> List[Tuple[dict, any, any]]:
    instances = []
    image_dir = pkl_data['img_dir']
    use_random = True  # baseline 还是保守点
    print('\n   loadding pkl len: %d  data... \n and use random: %s, is train: %s' % (
        len(pkl_data), use_random, is_train))
    key_list = list(pkl_data.keys())
    lack_count = 0
    less_f03_count = 0
    for key in tqdm(key_list):
        if key in ["classes", "img_dir"]:
            continue
        single_d = pkl_data[key]
        if single_d['f1_consis_avg'] < 0.3 and is_train:
            less_f03_count += 1
            continue

        case_label = 3  # 1 取随机方案 2 去 大于avg的方案 3 取官方训练数据0.3间隔的训练
        if case_label == 1 or case_label == 3:
            timestamps_list = single_d['substages_timestamps']
        elif case_label == 2:
            timestamps_list = single_d['single_lable']
        else:
            raise "case label error"
        # video_duration = single_d['video_duration']
        # img_count = int(video_duration / 0.25)
        if len(timestamps_list) > 0:
            label_i = random.randint(0, len(timestamps_list) - 1)
            timestamps = timestamps_list[label_i]
        else:
            timestamps = []
        pos_flag = [0.0 for _ in range(40)]
        path_video = single_d['path_video']
        image_template = os.path.join(image_dir, path_video.split('/')[-1].replace('.mp4', '') + '=%03d.jpg')
        for single_ts in timestamps:
            case = 1  # 1  方案1， 2 方案2
            # 方案1 取中间值
            if case == 1:
                if case_label == 1:
                    st = (single_ts['start_time'] + single_ts['end_time']) / 2
                elif case_label == 2 or case_label == 3:
                    st = single_ts
                else:
                    raise "case label error"
                pos_st = int(st / 0.25)
                # 对训练数据做一些处理，如边界扰动等 测试数据不扰动
                # r_d = random.randint(0,1)
                r_d = False
                if is_train and use_random and r_d:
                    delta = 1
                    if (st - pos_st * 0.25) > 0.125:
                        delta = 0
                    for get_pos_s in range(pos_st - delta, pos_st + 2 - delta):
                        if get_pos_s >= 0 and get_pos_s < 40:
                            pos_flag[get_pos_s] = 1.0
                else:
                    if pos_st >= 0 and pos_st < 40:
                        pos_flag[pos_st] = 1.0

        # get data
        img_paths = {}
        for img_i in range(1, 41):
            img_p = image_template % img_i
            if not os.path.exists(img_p) and img_i <= 1:
                lack_count += 1
                break
            if not os.path.exists(img_p):
                img_paths[img_i] = img_paths[img_i - 1]
            else:
                img_paths[img_i] = img_p
        if is_train:
            pass
        if len(img_paths) == 40:
            if not only_reverser:
                item = img_paths, np.array(pos_flag), key
                instances.append(item)
            # if is_train:
            else:
                reverse_img_paths = {}
                for img_i in range(1, 41):
                    reverse_img_paths[41 - img_i] = img_paths[img_i]
                pos_flag.reverse()
                item = reverse_img_paths, np.array(pos_flag), key
                instances.append(item)
    print('============== lack videos : %d , f_avg < 0.3: %d======'
          '==========ins : %d' % (lack_count, less_f03_count, len(instances)))
    r_d = random.randint(0, 1)
    # if r_d and is_train:
    random.shuffle(instances)
    return instances


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.JPEG')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetGEBD(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetGEBD, self).__init__(root, transform=transform,
                                          target_transform=target_transform)

        self.pkl_data_map = {}
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_GEBD_dataset(self.pkl_data_map, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, GEBD_file: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            GEBD_file (string): GEBD file path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        with open(GEBD_file, 'rb') as f:
            dict_raw = pickle.load(f, encoding='lartin1')
        self.pkl_data_map = dict_raw
        classes = dict_raw['classes']
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class ImageGEBD(DatasetGEBD):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageGEBD, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                        transform=transform,
                                        target_transform=target_transform,
                                        is_valid_file=is_valid_file)
        self.imgs = self.samples


class DatasetGEBD3D(VisionDataset):
    """
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            is_train: bool,
            loader: Callable[[str], Any],
            only_reverse: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(DatasetGEBD3D, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        self.pkl_data_map = self._find_pkl_map(self.root)
        samples = make_GEBD_dataset(self.pkl_data_map, is_train, only_reverse)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = loader

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_pkl_map(self, GEBD_file: str) -> dict:
        """
        Finds the class folders in a dataset.

        Args:
            GEBD_file (string): GEBD file path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        with open(GEBD_file, 'rb') as f:
            dict_raw = pickle.load(f, encoding='lartin1')
        return dict_raw

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                images_get = []
                paths, target, video_name = self.samples[index]
                for i in range(1, 41):
                    path = paths[i]
                    sample = self.loader(path)
                    images_get.append(sample)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            images_get = self.transform(images_get)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images_get, torch.from_numpy(target), video_name

    def __len__(self) -> int:
        return len(self.samples)


class ImageGEBD3D(DatasetGEBD3D):
    """
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            is_train: bool,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            only_reverse: bool = False,
            loader: Callable[[str], Any] = default_loader,
    ):
        super(ImageGEBD3D, self).__init__(root, is_train=is_train, loader=loader,
                                          transform=transform,
                                          target_transform=target_transform, only_reverse=only_reverse)
        self.imgs = self.samples

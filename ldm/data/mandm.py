import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import torchvision.transforms.functional as F

class MandMBase(Dataset):
    def __init__(self,
                 data_path,
                 mode=None,
                 size=256,
                 interpolation="nearest",
                 num_classes=4
                 ):
        self.h5_file_path = data_path
        self.h5_file = h5py.File(self.h5_file_path, 'r')
        self.slice_mapping = []
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice

        assert mode in ["Training", "Validation", "Test"]
        if mode == 'Training':
            self.dataset = self.h5_file['Training_Labeled']
        elif mode == 'Validate':
            self.dataset = self.h5_file['Validation']
        else:
            self.dataset = self.h5_file['Test']
        self.image_keys = [key for key in self.dataset.keys() if 'image' in key]

        # Create a mapping of (image_key, slice_index)
        for key in self.image_keys:
            image_shape = self.dataset[key].shape
            num_slices = image_shape[2]  # Number of slices
            for slice_idx in range(num_slices):
                self.slice_mapping.append((key, slice_idx))

        self._length = len(self.slice_mapping)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        data = dict()
        img_key, slice_idx = self.slice_mapping[i]
        lbl_key = img_key.replace('image', 'label')  # Corresponding label key
        image = self.dataset[img_key][:, :, slice_idx]  # Get the slice
        label = self.dataset[lbl_key][:, :, slice_idx]  # Get the slice

        img = Image.fromarray(self.normalize(image)).convert('RGB')
        mask = Image.fromarray(np.uint8(label)).convert('RGB')
        if self.size is not None:
            mask = mask.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            img = img.resize((self.size, self.size), resample=PIL.Image.BICUBIC)

        mask = np.array(mask).astype(np.float32)

        exist_class = sorted(list(set(mask.flatten())))
        class_id = np.random.choice(np.array(exist_class), size=1,
                                    p=None).astype(np.int64)

        # # choose class from id (3 channel, 1 class)
        if class_id != 0:
            mask = (mask == class_id)   # for multi, get a random class (existed) except 0
        else:
            mask = (mask != class_id)   # (empty slice) or (not empty slice & class_id==0)

        # turn segmentation map [0, 1] -> [-1, 1]
        data["class_id"] = class_id
        data["segmentation"] = ((mask.astype(np.float32) * 2) - 1)
        # if self.mode == 'Test':
        #     data['segmentation'] = mask
        # else:
        #     data['segmentation'] = (segmentation / 3.0) * 2.0 - 1.0

        img = np.array(img).astype(np.float32) / 255.
        img = (img * 2.) - 1.
        data['image'] = img
        return data

    @staticmethod
    def normalize(image):
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val <= 1:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = ((image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return image_array



class MandMTrain(MandMBase):
    def __init__(self, **kwargs):
        super().__init__(data_path="/kaggle/input/mandm-challenge/mandm.h5", mode="Training", **kwargs)


class MandMValidation(MandMBase):
    def __init__(self, **kwargs):
        super().__init__(data_path="/kaggle/input/mandm-challenge/mandm.h5", mode="Validation", **kwargs)

class MandMTest(MandMBase):
    def __init__(self, **kwargs):
        super().__init__(data_path="/kaggle/input/mandm-challenge/mandm.h5", mode="Test", **kwargs)




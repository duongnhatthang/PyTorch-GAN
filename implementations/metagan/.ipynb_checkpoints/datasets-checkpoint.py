import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class PACS_Dataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):#mode = {train, test, crossval}
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        def _prepare_data(path):
            files = []
            labels = []
            tmp = open(path, "r").readlines()
            for item in tmp:
                f, l = item.strip().split()
                files.append(root+"/pacs_data/pacs_data/"+f)
                labels.append(float(l))
            return files, labels

        self.files_A ,self.labels_A = _prepare_data(root+"/pacs_label/art_painting_"+mode+"_kfold.txt")
        self.files_C ,self.labels_C = _prepare_data(root+"/pacs_label/cartoon_"+mode+"_kfold.txt")
        self.files_B ,self.labels_B = _prepare_data(root+"/pacs_label/photo_"+mode+"_kfold.txt")
        self.files_D ,self.labels_D = _prepare_data(root+"/pacs_label/sketch_"+mode+"_kfold.txt")

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        label_A = self.labels_A[index % len(self.files_A)]

        if self.unaligned:
            idx_B = random.randint(0, len(self.files_B) - 1)
            label_B = self.labels_B[idx_B]
            image_B = Image.open(self.files_B[idx_B])
            idx_C = random.randint(0, len(self.files_C) - 1)
            label_C = self.labels_C[idx_C]
            image_C = Image.open(self.files_C[idx_C])
            idx_D = random.randint(0, len(self.files_D) - 1)
            label_D = self.labels_D[idx_D]
            image_D = Image.open(self.files_D[idx_D])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
            image_C = Image.open(self.files_C[index % len(self.files_C)])
            image_D = Image.open(self.files_D[index % len(self.files_D)])
            label_B = self.labels_B[index % len(self.files_B)]
            label_C = self.labels_C[index % len(self.files_C)]
            label_D = self.labels_D[index % len(self.files_D)]

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
            image_C = to_rgb(image_C)
            image_D = to_rgb(image_D)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_C = self.transform(image_C)
        item_D = self.transform(image_D)
        return {"A": item_A, "B": item_B, "C": item_C, "D": item_D, "label_A": label_A, "label_B": label_B, "label_C": label_C, "label_D": label_D}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B), len(self.files_C), len(self.files_D))
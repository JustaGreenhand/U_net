## 1.0.1
from torch.utils.data import Dataset
import torch
import os
from PIL import Image



def read_split(root):
    if os.path.exists(root) == False:
        print("--the dataset does not exict.--")
        exit()
    Myclass=[cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    #print(Myclass)
    #['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']
    Myclass.sort()
    train_name = [cla for cla in os.listdir(os.path.join(root, Myclass[0]))]
    test_name = [cla for cla in os.listdir(os.path.join(root, Myclass[1]))]

    train_img_path = [os.path.join(root, Myclass[0], name) for name in train_name]
    train_label_path = [os.path.join(root, Myclass[2], name) for name in train_name]
    test_img_path = [os.path.join(root, Myclass[1], name) for name in test_name]
    test_label_path = [os.path.join(root, Myclass[3], name) for name in test_name]

    #print(train_label_path,train_img_path,test_img_path,test_label_path)
    return train_img_path,train_label_path,test_img_path,test_label_path

class My_Dataset(Dataset):
    def __init__(self, img_path: list, label_path: list, transforms= None, dataset_type = "train"):
        self.img_path = img_path
        self.label_path = label_path
        self.transforms = transforms
        self.dataset_type = dataset_type
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, item):
        if self.dataset_type == "train":#read file from the path
            img = Image.open(self.label_path[item]).convert("L")
            label = Image.open(self.label_path[item]).convert("L")
            if self.transforms is not None:#transforms
                img = self.transforms(img)
                label = self.transforms(label)
            return img, label
        
        elif self.dataset_type == "test":
            img = Image.open(self.label_path[item]).convert("L")
            # print(type(img))
            size = img.size
            # print(size)
            # print(type(size))
            if self.transforms is not None:
                img = self.transforms(img)
            return img, self.img_path[item], size
                # label = self.transforms(label)
        # print(img.shape,label.shape)
        # exit()
        #print(img.shape,label.shape)
        # img = img / 256
        # label = label /256
        
    
    @staticmethod
    def collate_fn(batch):
        tmp = tuple(zip(*batch))#解包
        if len(tmp) == 3:
            images, path, size = tmp
            images = torch.stack(images, dim=0)
            return images, path, size
        elif len(tmp) == 2:
            images, labels = tmp
            images = torch.stack(images, dim=0)
            labels = torch.stack(labels, dim=0)
            return images, labels


# path = "./archive"
# read_split(path)
# def Save_Image(data, save_path):
#     array = data.cpu().detach().numpy()
#     print(array.shape)
#     img = Image.fromarray(array, mode="L")
#     img.save(save_path)
#     print("finish")
def unsample_add(size:tuple):
    if size[0][0] >= size[0][1]:
        if (size[0][0] - size[0][1]) %2 == 0:
            left_add = right_add = int((size[0][0] - size[0][1]) /2)
        elif (size[0][0] - size[0][1]) %2 !=0:
            right_add = int((size[0][0] - size[0][1]) /2)
            left_add = right_add + 1
        return left_add, right_add, True
    else:
        if (size[0][1] - size[0][0]) %2 == 0:
            up_add = down_add = int((size[0][1] - size[0][0]) /2)
        elif (size[0][1] - size[0][0]) %2 !=0:
            up_add = int((size[1][0] - size[0][0]) /2)
            down_add = up_add + 1
        return up_add, down_add, False
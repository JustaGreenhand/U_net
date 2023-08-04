## 1.0.2
import torch
from Network import Unet
import My_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

path = "./archive"
train_img_path, train_label_path, test_img_path, test_label_path = My_dataset.read_split(path)
Available = torch.cuda.is_available()
USE_GPU = True
mask_path = "./archive/save_mask"

data_transforms = {
        "test":transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512,antialias=True),
            transforms.CenterCrop((512,512))            
            ])
    }
# test set
test_set = My_dataset.My_Dataset(img_path=test_img_path,
                          label_path=test_label_path,
                          transforms=data_transforms["test"],
                          dataset_type="test")
# dataloader
test_loader = DataLoader(dataset=test_set,
                             shuffle=True,
                             collate_fn=test_set.collate_fn)

def test():
    unet = Unet(in_channel=1, out_channel=1)
    unet.load_state_dict(torch.load("unet.pt"))
    loss_function = torch.nn.BCEWithLogitsLoss()
    if Available and USE_GPU:
        unet = unet.cuda()
        loss_function = loss_function.cuda()
    

    for data in test_loader:
        images, path,size = data
        print(path, size) # width, height
        name = os.path.split(path[0])[-1]#最后一个
        if Available and USE_GPU:
            images = images.cuda()
            labels = images.cuda()
        output = unet(images)
        resize = transforms.Resize(min(size[0][0],size[0][1]), antialias=True)
        output = resize(output)
        #blnum:判断长边是竖直还是水平
        add_one, add_two, blnum= My_dataset.unsample_add(size) # True:width > height
        if blnum:
            pad = torch.nn.ZeroPad2d(padding=(add_one, add_two, 0, 0))
        else:
            pad = torch.nn.ZeroPad2d(padding=(0, 0, add_one, add_two))
        output = pad(output)
        batch, channel, h, w = output.shape
        img = output.reshape((h,w))
        save_path = os.path.join(mask_path, name)
        
        img = img > 0.5 #高清化
        img = img.float()#节省内存
        # img = img * 255
        # print(img)
        # My_dataset.Save_Image(img, save_path)
        save_image(img, save_path)
        # print("finish")
        # exit()



if __name__ == '__main__':
    test()
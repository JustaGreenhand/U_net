## 1.0.1
import torch
import os
from torchvision import transforms
import My_dataset
from torch.utils.data import DataLoader
from Network import Unet
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

path = "./archive"
#path
train_img_path, train_label_path = My_dataset.read_split(path)
#print(train_img_path,train_label_path,test_img_path,test_label_path)
#transforms

#super_para
LR = 0.001
Epoch = 2
Batch_size = 8
Num_worker = min([os.cpu_count(), Batch_size if Batch_size>1 else 0,8])
USE_GPU = True
Available = torch.cuda.is_available()
# writer = SummaryWriter(log_dir='runs/MNIST_experiment')

#device
data_transforms = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512,antialias=True),
            transforms.CenterCrop((512,512))
            ])
    }

#num_set
train_set = My_dataset.My_Dataset(img_path=train_img_path,
                           label_path=train_label_path,
                           transforms=data_transforms["train"])

train_loader = DataLoader(dataset=train_set,
                              batch_size=Batch_size,
                              shuffle=True,
                              num_workers=Num_worker,
                              collate_fn=train_set.collate_fn)

def main():
    # import netwoek
    unet = Unet(in_channel=1, out_channel=1)
    # define loss function
    loss_function = nn.BCELoss()
    # 优化器
    optimizer=torch.optim.RMSprop(unet.parameters(),lr=LR,weight_decay=1e-8,momentum=0.9)
    #GPU
    if Available and USE_GPU:
        unet = unet.cuda()
        loss_function = loss_function.cuda()


    # e = 1
    for epoch in range(Epoch):
        for data in train_loader:
            images, labels = data
            # print(images.shape)
            # exit()
            if Available and USE_GPU:
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            output = unet(images)
            # exit()
            loss = loss_function(output, labels)
            print(loss)
            loss.backward()
            # writer.add_scalar('训练损失值', loss, e)
            optimizer.step()
            # e+=1

    # save para
    torch.save(unet.state_dict(), "unet.pt") 
    print("finish")
if __name__ == '__main__':
    main()

# unet = Unet()
# img = torch.randn(2,1,256,256)
# output = unet(img)
# print(output.shape)
# cv2.imwrite()
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import cv2


class MyData(Dataset):

    def __init__(self, data_dir):
        self.path = data_dir
        self.img = os.listdir(data_dir)

    def __getitem__(self, index):
        img_name = self.img[index]
        img_path = os.path.join(self.path, img_name)
        _img = cv2.imread(img_path)
        _label = img_name[0]
        _img = cv2.resize(_img, dsize=(400, 300))
        return _img, _label

    def __len__(self):
        return len(self.img)


myData = MyData("D://Project//project_python//trainset_support")
train_loader = DataLoader(myData, 4)

writer = SummaryWriter("../writerBoard")
step = 0
for data in train_loader:
    img, label = data
    print(label, img.shape)
    writer.add_images("img_340", img, global_step=step, dataformats="NHWC")

    step += 1

writer.close()

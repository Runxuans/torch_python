import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()  # 先将数据转换为tensor类型，可以在后面添加各种操作
])

train_set = torchvision.datasets.MNIST(root="../dataset", transform=dataset_transform, train=True, download=False)
test_set = torchvision.datasets.MNIST(root="../dataset", transform=dataset_transform, train=False, download=False)

writer = SummaryWriter("numberBoard")

train_loader = DataLoader(train_set, 64)
step = 0
for data in train_loader:
    img, label = data
    writer.add_images("figure_2", img, step)
    step += 1

writer.close()

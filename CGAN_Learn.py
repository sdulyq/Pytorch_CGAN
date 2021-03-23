# 一次练习

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image


# 用到的参数及设置
batch_size = 128
lr = 0.0002
train_epoch = 50
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])




# 准备数据
data = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256) #　100是噪声的维度
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)


    # 循环时字典遍历的是key
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    def forward(self, input, label):
        batch_size = input.size(0)
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1) # 这样就变为512维了
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        # 可以reshape为图片的维度
        gen_img = x.view(batch_size, 1, 28, 28)
        return gen_img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(784, 1024)
        self.fc1_2 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        # Flatten操作，展平为784个像素后再操作
        batch_size = input.shape[0]
        x = input.view(batch_size, -1)

        x = F.leaky_relu(self.fc1_1(x), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1) # 按列进行连接，变为2048维
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x))

        return x


# network，结构可以参照图片
G = Generator().cuda()
D = Discriminator().cuda()
G.weight_init(mean=0, std=0.02)
D.weight_init(mean=0, std=0.02)
criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


def train(epoch):

    # 随着训练进行， 让lr变小

    if (epoch+1) == 30:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")


    if (epoch+1) == 40:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")


    for ind, (images, labels) in enumerate(data_loader):

        # 随机噪声z，100维
        z = Variable(torch.Tensor(numpy.random.normal(0, 1, (images.size(0), 100))))

        # train discriminator D

        mini_batch = images.size(0)
        D.zero_grad()

        #这是对应的二分类真与假的标签值
        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        # 这两句是生成label的one-hot向量,因为bceloss用的是one-hot
        y_label_ = torch.zeros(mini_batch, 10)
        y_label_.scatter_(1, labels.view(mini_batch, 1), 1)
        # (128,10)

        images, y_label_, y_real_, y_fake_ = Variable(images.cuda()), Variable(y_label_.cuda()), Variable(\
            y_real_.cuda()), Variable(y_fake_.cuda())


        D_result = D(images, y_label_).squeeze()  # 这是告诉判别器标签和图片的配对情况
        # 这个地方suqeeze就是128， 不是就是(128,1)， 不能BCEloss

        D_real_loss = criterion(D_result, y_real_)  # Discriminator的一个loss

        y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor)
        y_label_ = torch.zeros(mini_batch, 10)
        y_label_.scatter_(1, y_.view(mini_batch, 1), 1)
        z, y_label_ = Variable(z.cuda()), Variable(y_label_.cuda())

        G_result = G(z, y_label_)
        gen_imgs = G(z, y_label_)

        D_result = D(G_result, y_label_).squeeze()
        D_fake_loss = criterion(D_result, y_fake_)  # Discriminator的假图片loss

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # train generator G
        G.zero_grad()

        # 杂乱的生成的标签
        z_ = torch.rand((mini_batch, 100))
        y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor)
        y_label_ = torch.zeros(mini_batch, 10)
        y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

        z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_label_).squeeze()
        G_train_loss = criterion(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()


        if ind % 50 == 0:
            print('G_loss:', G_train_loss.data.item(), 'D_loss:', D_train_loss.data.item())
    save_image(gen_imgs.data[:25], 'images{}.png'.format(epoch), nrow=5, normalize=True)


if __name__ == '__main__':
    for epoch in range(train_epoch):
        train(epoch)


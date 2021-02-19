import numpy as np
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class DeformConv2d(nn.Module):
    """
    此为可变卷积核，作为可变神经网络的核与池化基础
    """

    def __init__(self, inc, out_c, kernel_size=3, padding=1, bias=None):
        """
        :param inc:  input channel \n
        :param out_c:  output channel \n
        :param kernel_size: 核大小 \n
        :param padding: 是否进行零填充 \n
        :param bias: 是否偏置 \n
        """
        super(DeformConv2d, self).__init__()  # 继承
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)  # 零填充
        self.conv_kernel = nn.Conv2d(inc, out_c, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        data_type = offset.data.type()
        kernel_size = self.kernel_size
        n = offset.size(1) // 2

        offset_index = Variable(torch.cat([torch.arange(0, 2 * n, 2),
                                           torch.arange(1, 2 * n + 1, 2)]),
                                requires_grad=False).type_as(x).long()
        # 增加和变换维度
        offset_index = offset_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # 偏移量，即delta p
        offset = torch.gather(offset, dim=1, index=offset_index)
        # 检查是否需要进行零填充
        if self.padding:
            x = self.zero_padding(x)

        p = self.get_p(offset, data_type)
        # 得到p，其结构为(b, 2n, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)
        # 转换p，使其变为(b, h, w, 2n)

        # 下面使用双线性插值法进行学习，得到最后的偏移量
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        # 三点表示多维度切片
        q_lt = torch.cat([torch.clamp(q_lt[..., :n], 0, x.size(2) - 1), torch.clamp(q_lt[..., n:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :n], 0, x.size(2) - 1), torch.clamp(q_rb[..., n:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :n], q_rb[..., n:]], -1)
        q_rt = torch.cat([q_rb[..., :n], q_lt[..., n:]], -1)

        mask = torch.cat([p[..., :n].lt(self.padding) + p[..., :n].gt(x.size(2) - 1 - self.padding),
                          p[..., n:].lt(self.padding) + p[..., n:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        # 得到(b, h, w, n)的结构
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :n], 0, x.size(2) - 1), torch.clamp(p[..., n:], 0, x.size(3) - 1)], dim=-1)

        # 线性插值(b, h, w, n)
        g_lt = (1 + (q_lt[..., :n].type_as(p) - p[..., :n])) * (1 + (q_lt[..., n:].type_as(p) - p[..., n:]))
        g_rb = (1 - (q_rb[..., :n].type_as(p) - p[..., :n])) * (1 - (q_rb[..., n:].type_as(p) - p[..., n:]))
        g_lb = (1 + (q_lb[..., :n].type_as(p) - p[..., :n])) * (1 - (q_lb[..., n:].type_as(p) - p[..., n:]))
        g_rt = (1 - (q_rt[..., :n].type_as(p) - p[..., :n])) * (1 + (q_rt[..., n:].type_as(p) - p[..., n:]))

        # 得到(b, c, h, w, n)
        x_q_lt = self.get_xq(x, q_lt, n)
        x_q_rb = self.get_xq(x, q_rb, n)
        x_q_lb = self.get_xq(x, q_lb, n)
        x_q_rt = self.get_xq(x, q_rt, n)

        # 形成输出数据的结构(b, c, h, w, n)
        x_offset = (g_lt.unsqueeze(dim=1) * x_q_lt) + \
                   (g_rb.unsqueeze(dim=1) * x_q_rb) + \
                   (g_lb.unsqueeze(dim=1) * x_q_lb) + \
                   (g_rt.unsqueeze(dim=1) * x_q_rt)

        x_offset = self.reshape_the_offset(x_offset, kernel_size)
        out_x = self.conv_kernel(x_offset)

        return out_x

    def get_pn(self, n, data_type):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   indexing='ij')
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))  # (2n, 1)
        p_n = np.reshape(p_n, (1, 2 * n, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(data_type), requires_grad=False)

        return p_n

    @staticmethod
    def get_p0(h, w, n, data_type):
        p0_x, p0_y = np.meshgrid(range(1, h + 1), range(1, w + 1), indexing='ij')
        p0_x = p0_x.flatten().reshape(1, 1, h, w).repeat(n, axis=1)
        p0_y = p0_y.flatten().reshape(1, 1, h, w).repeat(n, axis=1)
        p_0 = np.concatenate((p0_x, p0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(data_type), requires_grad=False)

        return p_0

    def get_p(self, offset, data_type):
        n, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self.get_pn(n, data_type)  # (1, 2n, 1, 1)
        p_0 = self.get_p0(h, w, n, data_type)  # (1, 2n, h, w)
        p = p_0 + p_n + offset
        return p

    @staticmethod
    def get_xq(x, q, n):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)  # (b, c, h*w)

        # (b, h, w, n)
        index = q[..., :n] * padded_w + q[..., n:]  # offset_x*w + offset_y
        # (b, c, h*w*n)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, n)

        return x_offset

    @staticmethod
    def reshape_the_offset(x_offset, kernel_size):
        b, c, h, w, n = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + kernel_size].contiguous().view(b, c, h, w * kernel_size) for s in
                              range(0, n, kernel_size)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * kernel_size, w * kernel_size)

        return x_offset


class DCN(nn.Module):
    def __init__(self):
        super(DCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv4 = DeformConv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        offsets = self.offsets(x)
        x = F.relu(self.conv4(x, offsets))
        x = self.bn4(x)

        x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


# 此为训练集
train_loader = torch.utils.data.DataLoader(
    # 第一个为数据集的路径，如果download为True，则表示需要从网络下载mnist数据集
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=6, shuffle=True)
# 此为测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=6, shuffle=True)

# 检查当前设备是GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_device(x):  # 200-207行为现实当前设备类型
    if x == 'cpu':
        return 'CPU'
    else:
        return 'GPU'


print('你当前的设备类型是: {}'.format(print_device(device)))  # 测试时可以打印出来看一看

DEVICE = device  # 设备类型
EPOCHS = 3  # 训练轮次
model = DCN().to(DEVICE)  # 模型载入设备中，如果是GPU则载入GPU
# 随机梯度下降的方法
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# optimizer = optim.Adam(model.parameters())

# 训练数据区
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# 测试数据区
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set (epoch{}): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader, epoch)

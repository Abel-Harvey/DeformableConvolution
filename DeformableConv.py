import numpy as np
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class DeformConv2D(nn.Module):
    def __init__(self, inc, out_c, kernel_size=3, padding=1, bias=None):
        """
        :param inc: input channel
        :param out_c: output channel
        :param kernel_size:  这不用说吧
        :param padding:  是否填充 Y=1, N=0
        :param bias: 偏置
        """
        super(DeformConv2D, self).__init__()

        self.kernel_size = kernel_size
        self.conv_kernel = nn.Conv2d(inc, out_c, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)

    def forward(self, x, offset):
        data_type = offset.data.type()  # 得到例如torch.FloatTensor的数据类型
        ks = self.kernel_size
        n = offset.size(1) // 2  # 得到a*b的b,再将b//2赋值给N

        # 得到偶数range和奇数的，再将他们连接起来，再转换为Variable类型
        offset_index = Variable(torch.cat([torch.arange(0, 2 * n, 2), torch.arange(1, 2 * n + 1, 2)]),
                                requires_grad=False).type_as(x).long()
        offset_index = offset_index.unsqueeze(dim=0)  # 增加一个维度，例如原本为[1,2,3]现在变成了[[1,2,3]]
        offset_index = offset_index.unsqueeze(dim=-1)  # 先增加一个维度，然后将其转换为列向量,变成了如[[[1],[2],[3]]]
        offset_index = offset_index.unsqueeze(dim=-1)
        offset_index = offset_index.unsqueeze(dim=-1)  # 变成了[[ [[[[1]]]],[[[[2]]]],[[[[3]]]]]]
        offset_index = offset_index.expand(*offset.size())  # 扩展
        offset = torch.gather(offset, dim=1, index=offset_index)  # 沿给定轴dim，将输入索引张量index指定位置的值进行聚合

        if self.padding:  # 如果需要进行0填充，则对x进行填充
            x = self.zero_padding(x)

        # 下面所进行的操作，均为双线性插值的算法，即计算出偏移量delta P
        p = self.get_p(offset, data_type)
        p = p.contiguous().permute(0, 2, 3, 1)  # 将tensor的维度换位, 类似zip，将其逐一对应形成新的维度
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :n], 0, x.size(2) - 1), torch.clamp(q_lt[..., n:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :n], 0, x.size(2) - 1), torch.clamp(q_rb[..., n:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :n], q_rb[..., n:]], -1)
        q_rt = torch.cat([q_rb[..., :n], q_lt[..., n:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :n].lt(self.padding) + p[..., :n].gt(x.size(2) - 1 - self.padding),
                          p[..., n:].lt(self.padding) + p[..., n:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :n], 0, x.size(2) - 1), torch.clamp(p[..., n:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :n].type_as(p) - p[..., :n])) * (1 + (q_lt[..., n:].type_as(p) - p[..., n:]))
        g_rb = (1 - (q_rb[..., :n].type_as(p) - p[..., :n])) * (1 - (q_rb[..., n:].type_as(p) - p[..., n:]))
        g_lb = (1 + (q_lb[..., :n].type_as(p) - p[..., :n])) * (1 - (q_lb[..., n:].type_as(p) - p[..., n:]))
        g_rt = (1 - (q_rt[..., :n].type_as(p) - p[..., :n])) * (1 + (q_rt[..., n:].type_as(p) - p[..., n:]))

        # (b, c, h, w, N)
        x_q_lt = self.get_xq(x, q_lt, n)
        x_q_rb = self.get_xq(x, q_rb, n)
        x_q_lb = self.get_xq(x, q_lb, n)
        x_q_rt = self.get_xq(x, q_rt, n)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self.reshape_offset_x(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    @staticmethod
    def get_p0(h, w, n, data_type):
        p0_x, p0_y = np.meshgrid(range(1, h + 1), range(1, w + 1), indexing='ij')
        p0_x = p0_x.flatten().reshape(1, 1, h, w).repeat(n, axis=1)
        p0_y = p0_y.flatten().reshape(1, 1, h, w).repeat(n, axis=1)
        p0 = np.concatenate((p0_x, p0_y), axis=1)
        p0 = Variable(torch.from_numpy(p0).type(data_type), requires_grad=False)

        return p0

    def get_pn(self, n, data_type):
        pn_x, pn_y = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                 range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1 // 2 + 1)),
                                 indexing='ij')
        pn_x = pn_x.flatten()
        pn_y = pn_y.flatten()
        pn = np.concatenate((pn_x, pn_y))
        pn = np.reshape(pn, (1, 2 * n, 1, 1))
        pn = Variable(torch.from_numpy(pn).type(data_type), requires_grad=False)

        return pn

    def get_p(self, offset, data_type):
        n = offset.size(1) // 2
        h = offset.size(2)
        w = offset.size(3)

        pn = self.get_pn(n, data_type)
        p0 = self.get_p0(h, w, n, data_type)
        p = p0 + pn + offset

        return p

    @staticmethod
    def get_xq(x, q, n):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # 有些tensor并不是占用一整块内存，而是由不同的数据块组成
        # 而tensor的view()操作依赖于内存是整块的
        # 这时只需要执行contiguous()这个函数,把tensor变成在内存中连续分布的形式。
        x = x.contiguous().view(b, c, -1)
        index = q[..., :n] * padded_w + q[..., n:]  # 多维切片
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        offset_x = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, n)

        return offset_x

    @staticmethod
    def reshape_offset_x(offset_x, kernel_size):
        b, c, h, w, n = offset_x.size()
        offset_x = torch.cat([offset_x[..., s:s + kernel_size].contiguous().view(b, c, h, w * kernel_size)
                              for s in range(0, n, kernel_size)], dim=-1)
        offset_x = offset_x.contiguous().view(b, c, h * kernel_size, w * kernel_size)

        return offset_x


class DCN(nn.Module):
    def __init__(self):
        super(DCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


net = DCN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=6, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=6, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
DEVICE = device
EPOCHS = 2
model = DCN().to(DEVICE)
optimizer = optim.Adam(model.parameters())


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


def test(model, device, test_loader):
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
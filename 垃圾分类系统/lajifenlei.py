import os
import random
import sys
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm  #显示训练进度条

#1、 数据增强
#定义数据增强和预处理的变换
data_transforms = {
    "train": transforms.Compose([
        # 随机旋转图片
        transforms.RandomRotation(45),
        # 调整图片大小
        transforms.Resize(256),
        # 中心裁剪
        transforms.CenterCrop(224),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        # 随机垂直翻转
        transforms.RandomVerticalFlip(p=0.5),
        # 调整亮度、对比度、饱和度和色调
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        # 将图片转换为 Tensor
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "valid": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# ImageFolder 自动从目录结构中加载图像分类数据集
# 训练集使用增强变换，验证集仅做标准化
train_dataset = ImageFolder( "waste_big/train",transform=data_transforms["train"])
test_dataset = ImageFolder( "waste_big/valid", transform=data_transforms["valid"])
# 构建 DataLoader，用于批量读取数据，训练集打乱顺序，验证集保持顺序
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2、定义模型等变量
#判断是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
#增加载预训练的 ResNet50 模型(来自 TorchVision)
model = resnet50(weights=ResNet50_Weights.DEFAULT)
# 冻结模型中所有参数的梯度计算
for param in model.parameters():
    param.requires_grad=False
# 修改全连接层:替换原始全连接层(输出为1000 类)为新的输出层(245 类垃圾类别)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features,245)
# 将模型整体移至指定设备(GPU或CPU)
model = model.to(device)

#3、损失函数与优化器
# 使用交叉熵损失函数进行多分类任务
loss_function =nn.CrossEntropyLoss().to(device)
# 使用 Adam 优化器对模型参数进行更新
optimizer =torch.optim.Adam(model.parameters(),lr=0.0001)

#4、实现完整的训练+验证流程，并保存最佳模型
# 4、训练函数
def train(epoch, epochs):
    # 将模型设置为训练模式
    model.train()
    # 初始化批次损失为0
    batch_loss = 0.0
    # 初始化正确预测的数量为0
    correct = 0
    # 获取训练数据集的总大小
    total = len(train_dataset)
    # 使用tqdm包装训练数据加载器，以便在迭代时显示进度条
    train_bar = tqdm(train_loader, file=sys.stdout)
    # 遍历训练数据集中的每个批次
    for inputs, labels in train_bar:
        # 将输入数据移动到指定的设备
        inputs = inputs.to(device)
        # 将标签数据移动到指定的设备
        labels = labels.to(device)
        # 通过模型对输入数据进行前向传播，获取输出
        outputs = model(inputs)
        # 从输出中获取每个样本的最大值和对应的索引
        max_val, max_index = torch.max(outputs, 1)
        # 计算并累加正确预测的数量
        correct += torch.sum(max_index == labels).item()
        # 计算损失
        loss = loss_function(outputs, labels)
        # 清除之前的梯度
        optimizer.zero_grad()
        # 反向传播损失，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 累加批次损失
        batch_loss += loss.item()
        # 更新进度条的描述，显示当前批次的损失
        train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
    # 计算平均损失
    avg_loss = batch_loss / len(train_loader)
    # 计算训练精度
    acc_train = correct / total
    # 返回平均损失和训练精度
    return avg_loss, acc_train

# 5、验证函数（不进行梯度计算，加快验证速度）
def test(epoch, epochs):
    # 将模型设置为评估模式，以禁用dropout等仅在训练时需要的操作
    model.eval()
    # 初始化正确预测的数量为0
    correct = 0
    # 获取测试数据集的总大小
    total = len(test_dataset)
    # 在评估过程中禁用梯度计算，以减少内存消耗和提高计算速度
    with torch.no_grad():
        # 使用tqdm包装测试数据加载器，以显示评估进度条
        val_bar = tqdm(test_loader, file=sys.stdout)
        # 遍历测试数据集中的每个批次
        for inputs_test, labels_test in val_bar:
            # 将输入数据移动到指定设备（如GPU）
            inputs_test = inputs_test.to(device)
            # 将标签数据移动到指定设备（如GPU）
            labels_test = labels_test.to(device)
            # 使用模型对输入数据进行预测
            outputs_test = model(inputs_test)
            # 获取每个样本预测结果的最大值和对应的索引
            max_val, max_index = torch.max(outputs_test, 1)
            # 计算并累加正确预测的数量
            correct += torch.sum(max_index == labels_test).item()
            # 更新进度条的描述，显示当前是第几个评估周期
            val_bar.desc = f"train epoch[{epoch + 1}/{epochs}]"
    # 计算测试集上的准确率
    acc_test = correct / total
    # 返回测试集上的准确率
    return acc_test
# 6、主训练循环
epochs = 3  # 设置训练轮数
best_acc = 0  # 设置初始最佳准确率
for i in range(epochs):
    # 训练一个epoch
    avg_loss, acc_train = train(i,epochs)
    acc_test = test(i,epochs)
    # 更新最佳准确率
    if acc_test > best_acc:
        # 保存当前权重
        best_acc = acc_test
        # 创建保存目录
        os.makedirs('models', exist_ok=True)
        # 保存模型
        torch.save(model.state_dict(), "models/model_waste_50_best.pth")
    # 打印当前轮次信息
    print(f"epoch{i + 1},平均误差：{avg_loss},训练集准确率：{acc_train:.3%},测试集准确率：{acc_test:.3%}")

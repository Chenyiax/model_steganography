"""
文件名: train.py
作者: 徐辰屹
日期: 2024年2月1日

说明: 用于模型训练的文件
"""

from tqdm import tqdm

from get_data import get_cnn_data
from init_function import init_vit
from model_steganorgraphy import ModelSteganography
from stego_model import *
from test import test_model



def train_model(model, train_loader, criterion, optimizer,num_epochs=5, with_secret=True):
    '''
    训练模型的函数
    Args:
        model: 需要被训练的模型
        train_loader: 数据集
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        with_secret: 这个模型是否含有秘密信息(用于存储损失数据的命名)

    Returns:

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # init_func = init_vit
    # ms = ModelSteganography(init_func, max_nums=600000)
    acc_list = []
    loss_list = []
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度归零

            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播

            optimizer.step()  # 更新权重

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # 提取秘密信息
        # outputs_secrets, outputs_secrets_bch = ms.decode(model)
        # correct = (outputs_secrets == secret_bits).sum().item()
        # accuracy = correct / outputs_secrets.numel()
        # print("Extraction Accuracy of Secret Information:", accuracy)
        # acc_list.append(accuracy)

        epoch_loss = running_loss / len(train_loader)
        loss_list.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Acc:{total_correct/total_samples}")
    # torch.save(acc_list,f"data/train_acc_FashionMNIST_{model.__class__.__name__}_with_secret.pth")
    if with_secret:
        torch.save(loss_list,f"data/train_loss_{model.__class__.__name__}_with_secret.pth")
    else:
        torch.save(loss_list, f"data/train_loss_{model.__class__.__name__}_without_secret.pth")
    print("Training finished")


if __name__ == '__main__':
    train_loader, test_loader = get_cnn_data()
    task_model = Vgg16()

    # train_loader, test_loader, vocab_size, vocab_len = get_rnn_data()
    # task_model = TextRNN(vocab_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-5)
    train_model(task_model, train_loader, criterion, optimizer, num_epochs=300, with_secret=False)
    test_model(task_model, test_loader, criterion)
    torch.save(task_model, f"models/{task_model.__class__.__name__}_without_secret.pth")


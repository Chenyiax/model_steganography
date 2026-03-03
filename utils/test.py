"""
文件名: test.py
作者: 徐辰屹
日期: 2024年2月1日

说明:
用于模型测试的文件
"""

import torch


def test_model(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置模型为评估模式

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

    accuracy = total_correct / total_samples
    average_loss = total_loss / len(test_loader)

    print(f"Test Accuracy: {accuracy}, Average Loss: {average_loss}")
    return accuracy

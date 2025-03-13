import torch.nn as nn
import sys
sys.path.append('.')
sys.path.append('..')
from torchvision import models

# class Classifier(nn.Module):
#     def __init__(self, num_class):
#         super(Classifier, self).__init__()
#         self.num_class = num_class

#         self.network = ResNet(BasicBlock, [3, 4, 6, 3])
#         # del self.network.linear_layer

#         self.linear_layer = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, self.num_class),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         feature = self.network(x)
#         # for name, param in self.named_parameters():
#         #     if param.grad is not None:
#         #         print(f'Gradient for {name}: {param.grad.norm()}')
#         #     else:
#         #         print(f'No gradient for {name}')
#         x = self.linear_layer(feature)
#         # 返回除了预测结果外还有一维的特征向量
#         return x, feature.detach()

# class Classifier(nn.Module):
#     def __init__(self, num_class):
#         super(Classifier, self).__init__()
#         self.num_class = num_class

#         # 使用在ImageNet上预训练的ResNet34
#         # self.network = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

#         # # 删除最后一层全连接层
#         # self.network.fc = nn.Identity()

#         # self.network = models.efficientnet_b0(weights='DEFAULT')  # 选择 EfficientNet B0

#         # self.network.classifier = nn.Sequential(
#         #     nn.Linear(self.network.classifier[1].in_features, 512),  # 修改输出层
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.4),
#         # )

#         self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
#         # 将权重加载到模型中
#         self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
#         self.network.head =  nn.Sequential(
#             nn.Linear(self.network.head.in_features, 512),  # 修改输出层
#             nn.ReLU(),
#         )

#         # 添加自定义的线性层
#         self.linear_layer = nn.Sequential(
#             nn.Linear(512, self.num_class),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         feature = self.network(x)
#         x = self.linear_layer(feature)
#         # 返回预测结果和特征向量
#         return x, feature.detach()

# 输出额外的特征 
class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.num_class = num_class

        # 使用在ImageNet上预训练的ResNet34
        # self.network = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # # 删除最后一层全连接层
        # self.network.fc = nn.Identity()

        # self.network = models.efficientnet_b0(weights='DEFAULT')  # 选择 EfficientNet B0
        # self.hidden_size = self.network.classifier[1].in_features
        # self.network.classifier = nn.Identity()

        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()


        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()

        
        self.linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_class),  # 修改输出层
        )

    def forward(self, x):
        feature = self.network(x)
        # 返回预测结果
        return self.linear_layer(feature), feature
    
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


# 只输出预测结果
class ClassifierTwo(nn.Module):
    def __init__(self, num_class):
        super(ClassifierTwo, self).__init__()
        self.num_class = num_class

        # 使用在ImageNet上预训练的ResNet34
        # self.network = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # # 删除最后一层全连接层
        # self.network.fc = nn.Identity()

        # self.network = models.efficientnet_b0(weights='DEFAULT')  # 选择 EfficientNet B0
        # self.hidden_size = self.network.classifier[1].in_features
        # self.network.classifier = nn.Identity()

        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()


        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()

        
        self.linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_class),  # 修改输出层
            nn.Softmax()
        )

    def forward(self, x):
        feature = self.network(x)
        # 返回预测结果
        return self.linear_layer(feature)
    
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


class ClassifierAblation(nn.Module):
    def __init__(self, num_class):
        super(ClassifierAblation, self).__init__()
        self.num_class = num_class

        # 使用在ImageNet上预训练的ResNet34
        # self.network = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # # 删除最后一层全连接层
        # self.network.fc = nn.Identity()

        # self.network = models.efficientnet_b0(weights='DEFAULT')  # 选择 EfficientNet B0
        # self.hidden_size = self.network.classifier[1].in_features
        # self.network.classifier = nn.Identity()

        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()


        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()

        
        self.linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_class),  # 修改输出层
        )

    def forward(self, x):
        feature = self.network(x)
        # 返回预测结果
        return self.linear_layer(feature)
    
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True

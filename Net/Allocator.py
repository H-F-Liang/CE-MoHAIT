import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert (
                self.head_dim * num_heads == hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        # 定义线性层用于 Q, K, V 的线性变换
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        # 输出线性层
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # 计算 Q, K, V
        Q = self.fc_q(x)  # [batch_size, seq_length, hidden_dim]
        K = self.fc_k(x)  # [batch_size, seq_length, hidden_dim]
        V = self.fc_v(x)  # [batch_size, seq_length, hidden_dim]

        # 重塑为多头形式
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]

        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_length, seq_length]
        attn = F.softmax(energy, dim=-1)  # 注意力权重

        # 加权 V
        out = torch.matmul(attn, V)  # [batch_size, num_heads, seq_length, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)  # [batch_size, seq_length, hidden_dim]

        return self.fc_out(out)  # [batch_size, seq_length, hidden_dim]


class AttentionFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_heads):
        super(AttentionFusion, self).__init__()

        # 线性层将不同长度的输入映射到相同的 hidden_dim
        self.fc_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for input_dim in input_dims])

        # 多头注意力机制
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads)

        # 可选：添加 LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vectors):
        # 使用共享的线性层映射所有输入向量到相同的 hidden_dim 维度
        mapped_vectors = [F.gelu(fc(v)) for fc, v in zip(self.fc_layers, vectors)]

        # 将映射后的向量堆叠成一个三维张量
        stacked_vectors = torch.stack(mapped_vectors, dim=1)  # [batch_size, num_vectors, hidden_dim]

        # 应用多头注意力机制
        fused_vector = self.multihead_attention(stacked_vectors)  # [batch_size, num_vectors, hidden_dim]

        fused_vector = self.norm(fused_vector.mean(dim=1))  # 对序列维度取平均并进行 LayerNorm

        return fused_vector  # [batch_size, hidden_dim]


class Allocator(nn.Module):
    def __init__(self, num_expert, num_class, cls_hidden_size):
        super(Allocator, self).__init__()
        self.num_expert = num_expert
        self.num_class = num_class
        self.hidden_dim = 256
        self.cls_hidden_size = cls_hidden_size

        # 使用 EfficientNet B0 作为特征提取网络
        # self.network = models.efficientnet_b0(weights='DEFAULT')  # 选择 EfficientNet B0
        # # self.network = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        # self.hidden_size = self.network.classifier[1].in_features
        # self.network.classifier = nn.Identity()

        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(pretrained=True)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()

        # 注意力融合模块
        self.w_attention_fusion = AttentionFusion([self.hidden_size, self.cls_hidden_size, self.num_class], hidden_dim=self.hidden_dim, num_heads=4)

        self.p_attention_fusion = AttentionFusion([self.hidden_size, self.cls_hidden_size, self.num_class, self.num_expert], hidden_dim=self.hidden_dim, num_heads=4)

        # 输出专家权重
        self.weight_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, self.num_expert)
        )

        # 输出分配概率
        self.prob_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1)
        )

    def forward(self, x, cls_feature, cls_d):
        # 提取特征
        x = self.network(x)
        # 分别计算 w p

        # w 用于判断当前图片更适合哪个专家 [batch_size, num_expert]
        # 输入的是分配器提取的图片特征
        w_feature = self.w_attention_fusion([x, cls_feature.detach(), cls_d.detach()])
        w = self.weight_linear_layer(w_feature)

        # p 用于判断当前预测的健壮程度 用于决定当前图片是否应该由人类决策 [batch_size, 1]
        # 输入分配器提取的图像特征、分配器的分配概率分布、分类器提取的图像特征、分类器的预测概率分布
        p_feature = self.p_attention_fusion([x, cls_feature.detach(), cls_d.detach(), w.detach()])
        p = self.prob_linear_layer(p_feature)
        return p, w

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


class AllocatorTwo(nn.Module):
    def __init__(self, num_expert):
        super(AllocatorTwo, self).__init__()
        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(pretrained=True)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        self.num_expert = num_expert

        self.weight_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, self.num_expert),
            nn.Softmax()
        )

    def forward(self, x):
        # 提取特征
        x = self.network(x)
        w = self.weight_linear_layer(x)
        return w

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


# 用于消融实验 输入原始数据 输出分配权重
class AllocatorAblationOne(nn.Module):
    def __init__(self, num_expert):
        super(AllocatorAblationOne, self).__init__()
        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(pretrained=True)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        self.num_expert = num_expert

        self.weight_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, self.num_expert)
        )

    def forward(self, x):
        # 提取特征
        x = self.network(x)
        w = self.weight_linear_layer(x)
        return w

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


# 用于消融实验 输入原始数据 输出分配权重和专家弱点置信度
class AllocatorAblationTwo(nn.Module):
    def __init__(self, num_expert):
        super(AllocatorAblationTwo, self).__init__()
        self.num_expert = num_expert
        self.hidden_dim = 256

        # 使用 EfficientNet B0 作为特征提取网络
        # self.network = models.efficientnet_b0(weights='DEFAULT')  # 选择 EfficientNet B0
        # # self.network = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        # self.hidden_size = self.network.classifier[1].in_features
        # self.network.classifier = nn.Identity()

        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(pretrained=True)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()

        self.p_attention_fusion = AttentionFusion([self.hidden_size, self.num_expert], hidden_dim=self.hidden_dim, num_heads=4)

        # 输出专家权重
        self.weight_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, self.num_expert)
        )

        # 输出分配概率
        self.prob_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 提取特征
        x = self.network(x)
        # 分别计算 w p

        # w 用于判断当前图片更适合哪个专家 [batch_size, num_expert]
        # 输入的是分配器提取的图片特征
        w = self.weight_linear_layer(x)

        # p 用于判断当前预测的健壮程度 用于决定当前图片是否应该由人类决策 [batch_size, 1]
        # 输入分配器提取的图像特征、分配器的分配概率分布、分类器提取的图像特征、分类器的预测概率分布
        p_feature = self.p_attention_fusion([x, w.detach()])
        p = self.prob_linear_layer(p_feature)
        return p, w

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


# 用于消融实验 输入原始数据和补充信息 输出分配权重
class AllocatorAblationThree(nn.Module):
    def __init__(self, num_expert, num_class, cls_hidden_size):
        super(AllocatorAblationThree, self).__init__()
        self.num_expert = num_expert
        self.num_class = num_class
        self.hidden_dim = 256
        self.cls_hidden_size = cls_hidden_size

        # 使用 EfficientNet B0 作为特征提取网络
        # self.network = models.efficientnet_b0(weights='DEFAULT')  # 选择 EfficientNet B0
        # # self.network = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        # self.hidden_size = self.network.classifier[1].in_features
        # self.network.classifier = nn.Identity()

        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(pretrained=True)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()

        # 注意力融合模块
        self.w_attention_fusion = AttentionFusion([self.hidden_size, self.cls_hidden_size, self.num_class], hidden_dim=self.hidden_dim, num_heads=4)

        # 输出专家权重
        self.weight_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, self.num_expert)
        )

    def forward(self, x, cls_feature, cls_d):
        # 提取特征
        x = self.network(x)
        # w 用于判断当前图片更适合哪个专家 [batch_size, num_expert]
        # 输入的是分配器提取的图片特征
        w_feature = self.w_attention_fusion([x, cls_feature.detach(), cls_d.detach()])
        w = self.weight_linear_layer(w_feature)
        return w

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


class AllocatorFour(nn.Module):
    def __init__(self, num_expert, num_class, cls_hidden_size):
        super(AllocatorFour, self).__init__()
        self.num_expert = num_expert
        self.num_class = num_class
        self.hidden_dim = 256
        self.cls_hidden_size = cls_hidden_size

        # 使用 EfficientNet B0 作为特征提取网络
        # self.network = models.efficientnet_b0(weights='DEFAULT')  # 选择 EfficientNet B0
        # # self.network = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        # self.hidden_size = self.network.classifier[1].in_features
        # self.network.classifier = nn.Identity()

        # 使用 ResNet-18 作为特征提取网络
        self.network = models.resnet18(pretrained=True)  # 加载预训练的 ResNet-18 模型
        # 获取 ResNet-18 的最后一个全连接层的输入特征数
        self.hidden_size = self.network.fc.in_features
        # 将 ResNet-18 的最后分类层替换为 nn.Identity()，用于特征提取
        self.network.fc = nn.Identity()

        # self.network = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 将权重加载到模型中
        # self.network.load_state_dict(torch.load('/home/userlhf/projects/HAIT/Team/deit_base_patch16_224-b5f2ef4d.pth')['model'])
        # self.hidden_size = self.network.head.in_features
        # self.network.head =  nn.Identity()

        # 注意力融合模块
        self.w_attention_fusion = AttentionFusion([self.hidden_size, self.cls_hidden_size, self.num_class], hidden_dim=self.hidden_dim, num_heads=4)

        self.p_attention_fusion = AttentionFusion([self.hidden_size, self.cls_hidden_size, self.num_class, self.num_expert], hidden_dim=self.hidden_dim, num_heads=4)

        # 输出专家权重
        self.weight_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, self.num_expert)
        )

        # 输出分配概率
        self.prob_linear_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1)
        )

    def forward(self, x, cls_feature, cls_d):
        # 提取特征
        x = self.network(x)
        # 分别计算 w p

        # w 用于判断当前图片更适合哪个专家 [batch_size, num_expert]
        # 输入的是分配器提取的图片特征
        # TODO: 考虑将 x、cls_feature 和 cls_feature对齐
        w_feature = self.w_attention_fusion([x, cls_feature, cls_d.detach()])
        w = self.weight_linear_layer(w_feature)

        # p 用于判断当前预测的健壮程度 用于决定当前图片是否应该由人类决策 [batch_size, 1]
        # 输入分配器提取的图像特征、分配器的分配概率分布、分类器提取的图像特征、分类器的预测概率分布
        p_feature = self.p_attention_fusion([x, cls_feature, cls_d.detach(), w.detach()])
        p = self.prob_linear_layer(p_feature)
        return torch.cat([p, w], dim=1)  # 形状: [batch_size, 1 + num_expert]

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True

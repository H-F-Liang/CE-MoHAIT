import os
import random
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import sys

sys.path.append('.')
sys.path.append('..')
from Net.Allocator import *
from Net.Classifier import *
from utils.TrainingLog import save_log
from colorama import Fore, Style, init
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict
import pandas as pd

# 初始化 colorama
init(autoreset=True)
# 切换工作目录到当前
os.chdir(os.path.dirname(__file__))


def setup_seed(seed):
    random.seed(seed)
    # 固定 NumPy 随机数生成器的种子
    np.random.seed(seed)
    # 固定 PyTorch 的随机数生成器的种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多块 GPU
    # 设置 PyTorch 的 CuDNN 后端为确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 确保每次卷积算法选择一致


def create_file(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 创建文件的父目录（如果需要）
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 创建文件
        with open(file_path, 'w') as file:
            pass  # 使用 pass 创建空文件


class CustomDataset(Dataset):
    def __init__(self, data_info, root_dir, transform=None):
        self.data_info = data_info  # JSON 中的数据
        self.root_dir = root_dir  # 数据存放的根目录
        self.transform = transform  # 图像预处理

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 获取真实标签
        label = self.data_info[idx]["true_labels"]  # 金标准标签
        expert_labels = torch.tensor(self.data_info[idx]["expert_labels"])  # 专家标签

        # 获取图像文件路径
        file_name = self.data_info[idx]["image_name"]
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name).convert("RGB")  # 打开图像并转换为 RGB 格式
        if self.transform:
            image = self.transform(image)  # 图像预处理

        return image, label, expert_labels


def load_data_info(individual_csv, test_csv, val_csv, experts_id=[4323195249, 4295232296], col="Airspace opacity"):
    # 读取专家标签文件
    df_individual = pd.read_csv(individual_csv)
    df_test = pd.read_csv(test_csv)
    df_val = pd.read_csv(val_csv)

    # 只保留关注的列
    df_individual = df_individual[["Image ID", "Patient ID", "Reader ID", "Fracture", "Pneumothorax", "Airspace opacity", "Nodule/mass"]]
    df_test = df_test[["Image Index", "Fracture", "Pneumothorax", "Airspace opacity", "Nodule or mass"]]
    df_val = df_val[["Image Index", "Fracture", "Pneumothorax", "Airspace opacity", "Nodule or mass"]]

    # 将测试集和验证集中的真实标签组合
    df_true = pd.concat([df_test, df_val], ignore_index=True)
    df_true["Image Index"] = df_true["Image Index"].astype(str)

    # 对每个 Image ID 分组，检查是否包含指定专家的标注
    expert_labels_dict = {}
    patient_ids = {}  # 用于存储每张图片对应的患者 ID
    for img_id, group in df_individual.groupby("Image ID"):
        # 检查该图片是否包含所有指定专家的标注
        readers = set(group["Reader ID"])
        if set(experts_id).issubset(readers):
            # 只保留指定专家的标注
            expert_labels = []
            for expert_id in experts_id:
                label = group[group["Reader ID"] == expert_id][col].iloc[0] == "YES"
                expert_labels.append([label])

            expert_labels_dict[img_id] = expert_labels
            # 获取患者 ID
            patient_ids[img_id] = group["Patient ID"].iloc[0]  # 假设同一张图片的患者 ID 一致

    # 创建数据列表
    data_info = []
    for _, row in df_true.iterrows():
        img_id = str(row["Image Index"])
        true_labels = [row[col] == "YES"]

        # 只保留符合条件的图片
        if img_id in expert_labels_dict and len(expert_labels_dict[img_id]) == len(experts_id):
            data_info.append({
                "image_name": img_id,  # 假设图片为 png 格式
                "patient_id": patient_ids[img_id],  # 添加患者 ID
                "true_labels": torch.tensor(true_labels, dtype=torch.float32),
                "expert_labels": expert_labels_dict[img_id]
            })
    return data_info


def create_patient_based_folds(data_info, n_splits=5, seed=42):
    # 按患者 ID 分组
    patient_to_images = defaultdict(list)
    for item in data_info:
        patient_id = item["patient_id"]
        patient_to_images[patient_id].append(item)

    # 获取所有患者 ID
    patient_ids = list(patient_to_images.keys())

    # 创建 KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    # 根据患者 ID 划分 folds
    for train_idx, val_idx in kf.split(patient_ids):
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]

        # 根据患者 ID 获取对应的数据
        train_data = [item for pid in train_patients for item in patient_to_images[pid]]
        val_data = [item for pid in val_patients for item in patient_to_images[pid]]

        folds.append((train_data, val_data))

    return folds


def create_patient_folds(data_info, n_splits=10, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    将数据按照患者分组划分为训练集、验证集和测试集，并返回数据本身。

    Args:
        data_info (list): 每条数据包含 "image_name", "patient_id", "true_labels", "expert_labels" 等信息。
        n_splits (int): 数据划分的组数，用于多折验证。
        train_ratio (float): 训练集比例。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。
        seed (int): 随机种子。

    Returns:
        list[dict]: 每个折包含训练集、验证集和测试集的数据。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "train_ratio, val_ratio, and test_ratio must sum to 1."

    # 设置随机种子
    random.seed(seed)

    # 将数据按 patient_id 分组
    patient_to_images = defaultdict(list)
    for entry in data_info:
        patient_to_images[entry['patient_id']].append(entry)

    # 获取所有患者 ID
    all_patients = list(patient_to_images.keys())

    # 多折划分结果
    folds = []

    for split in range(n_splits):
        # 打乱患者顺序
        random.shuffle(all_patients)

        # 按比例计算每个数据集的患者数量
        num_patients = len(all_patients)
        train_count = int(num_patients * train_ratio)
        val_count = int(num_patients * val_ratio)
        test_count = num_patients - train_count - val_count

        # 划分患者到训练集、验证集和测试集
        train_patients = all_patients[:train_count]
        val_patients = all_patients[train_count:train_count + val_count]
        test_patients = all_patients[train_count + val_count:]

        # 根据患者分配数据
        train_data = [item for patient in train_patients for item in patient_to_images[patient]]
        val_data = [item for patient in val_patients for item in patient_to_images[patient]]
        test_data = [item for patient in test_patients for item in patient_to_images[patient]]

        # 保存当前折
        folds.append((train_data, val_data, test_data))

    return folds


def create_stratified_patient_folds(data_info, n_splits=10, seed=42):
    """
    按照患者分组，并基于 stratified k-fold 方法进行数据划分。
    数据按照 radiologist 的性能分层，确保分布一致。

    Args:
        data_info (list): 每条数据包含 "image_name", "patient_id", "true_labels", "expert_labels" 等信息。
        n_splits (int): 数据划分的组数，用于多折验证。
        seed (int): 随机种子。

    Returns:
        list[tuple]: 每个折包含 (train_data, val_data, test_data) 的列表。
    """

    # 将数据转换为 DataFrame
    data_df = pd.DataFrame(data_info)

    # 确保输入数据包含所需字段
    assert "patient_id" in data_df.columns, "数据中缺少 'patient_id' 字段。"
    assert "true_labels" in data_df.columns, "数据中缺少 'true_labels' 字段。"
    assert "expert_labels" in data_df.columns, "数据中缺少 'expert_labels' 字段。"

    # 按患者计算 radiologist 的性能
    data_df["expert_correct"] = data_df["expert_labels"] == data_df["true_labels"]
    patient_perf = data_df.groupby("patient_id").agg(
        num_images=("image_name", "count"),
        expert_correct=("expert_correct", "sum")
    ).reset_index()
    patient_perf["expert_perf"] = patient_perf["expert_correct"] / patient_perf["num_images"]

    # 创建分层目标变量
    patient_perf["stratify_target"] = patient_perf["expert_perf"].round(2).astype(str)

    # 初始化 StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    # 按患者划分训练、验证和测试集
    patient_ids = patient_perf["patient_id"].values
    stratify_target = patient_perf["stratify_target"].values

    for train_val_idx, test_idx in skf.split(patient_ids, stratify_target):
        # 测试集
        test_patients = patient_ids[test_idx]

        # 使用下一个 2 折作为验证集
        val_idx = train_val_idx[:len(train_val_idx) // (n_splits - 1) * 2]
        val_patients = patient_ids[val_idx]

        # 剩下的用于训练集
        train_idx = train_val_idx[len(train_val_idx) // (n_splits - 1) * 2:]
        train_patients = patient_ids[train_idx]

        # 根据患者分配数据
        train_data = data_df[data_df["patient_id"].isin(train_patients)].to_dict("records")
        val_data = data_df[data_df["patient_id"].isin(val_patients)].to_dict("records")
        test_data = data_df[data_df["patient_id"].isin(test_patients)].to_dict("records")

        folds.append((train_data, val_data, test_data))

    return folds


def create_2_stratified_patient_folds(data_info, n_splits=10, seed=42):
    """
    按照患者分组，并基于 stratified k-fold 方法进行数据划分。
    数据按照两个放射科医生的性能组合分层，确保分布一致。

    Args:
        data_info (list): 每条数据包含 "image_name", "patient_id", "true_labels", "expert_labels" 等信息。
        n_splits (int): 数据划分的组数，用于多折验证。
        seed (int): 随机种子。

    Returns:
        list[tuple]: 每个折包含 (train_data, val_data, test_data) 的列表。
    """

    # 将数据转换为 DataFrame
    data_df = pd.DataFrame(data_info)

    # 确保输入数据包含所需字段
    assert "patient_id" in data_df.columns, "数据中缺少 'patient_id' 字段。"
    assert "true_labels" in data_df.columns, "数据中缺少 'true_labels' 字段。"
    assert "expert_labels" in data_df.columns, "数据中缺少 'expert_labels' 字段。"

    # 计算每个图像的专家平均性能
    def calculate_expert_performance(row):
        # 计算每个专家的正确性
        expert_correct = [expert == row["true_labels"] for expert in row["expert_labels"]]
        # 返回平均性能
        return sum(expert_correct) / len(expert_correct)

    data_df["expert_perf"] = data_df.apply(calculate_expert_performance, axis=1)

    # 按患者计算平均性能
    patient_perf = data_df.groupby("patient_id").agg(
        num_images=("image_name", "count"),
        expert_perf=("expert_perf", "mean")
    ).reset_index()

    # 假设有两个放射科医生，分别计算他们的性能
    # 这里假设 expert_labels 是一个列表，包含两个放射科医生的标注
    def calculate_radiologist_performance(row, radiologist_index):
        # 获取指定放射科医生的标注
        radiologist_label = row["expert_labels"][radiologist_index]
        # 返回是否正确
        return radiologist_label == row["true_labels"]

    # 计算第一个放射科医生的性能
    data_df["radiologist_1_perf"] = data_df.apply(lambda row: calculate_radiologist_performance(row, 0), axis=1)
    # 计算第二个放射科医生的性能
    data_df["radiologist_2_perf"] = data_df.apply(lambda row: calculate_radiologist_performance(row, 1), axis=1)

    # 按患者计算两个放射科医生的平均性能
    patient_perf = data_df.groupby("patient_id").agg(
        num_images=("image_name", "count"),
        radiologist_1_perf=("radiologist_1_perf", "mean"),
        radiologist_2_perf=("radiologist_2_perf", "mean")
    ).reset_index()

    # 创建分层目标变量：将两个放射科医生的性能组合成一个字符串
    patient_perf["stratify_target"] = (
            patient_perf["radiologist_1_perf"].round(2).astype(str) + "_" +
            patient_perf["radiologist_2_perf"].round(2).astype(str)
    )

    # 初始化 StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    # 按患者划分训练、验证和测试集
    patient_ids = patient_perf["patient_id"].values
    stratify_target = patient_perf["stratify_target"].values

    for train_idx, test_idx in skf.split(patient_ids, stratify_target):
        # 测试集
        test_patients = patient_ids[test_idx]

        train_patients = patient_ids[train_idx]

        # 根据患者分配数据
        train_data = data_df[data_df["patient_id"].isin(train_patients)].to_dict("records")
        test_data = data_df[data_df["patient_id"].isin(test_patients)].to_dict("records")

        folds.append((train_data, test_data))

    return folds


def create_3_stratified_patient_folds(data_info, n_splits=10, seed=42):
    """
    按照患者分组，并基于 stratified k-fold 方法进行数据划分。
    数据按照两个放射科医生的性能组合分层，确保分布一致。

    Args:
        data_info (list): 每条数据包含 "image_name", "patient_id", "true_labels", "expert_labels" 等信息。
        n_splits (int): 数据划分的组数，用于多折验证。
        seed (int): 随机种子。

    Returns:
        list[tuple]: 每个折包含 (train_data, val_data, test_data) 的列表。
    """

    # 将数据转换为 DataFrame
    data_df = pd.DataFrame(data_info)

    # 确保输入数据包含所需字段
    assert "patient_id" in data_df.columns, "数据中缺少 'patient_id' 字段。"
    assert "true_labels" in data_df.columns, "数据中缺少 'true_labels' 字段。"
    assert "expert_labels" in data_df.columns, "数据中缺少 'expert_labels' 字段。"

    # 计算每个图像的专家平均性能
    def calculate_expert_performance(row):
        # 计算每个专家的正确性
        expert_correct = [expert == row["true_labels"] for expert in row["expert_labels"]]
        # 返回平均性能
        return sum(expert_correct) / len(expert_correct)

    data_df["expert_perf"] = data_df.apply(calculate_expert_performance, axis=1)

    # 按患者计算平均性能
    patient_perf = data_df.groupby("patient_id").agg(
        num_images=("image_name", "count"),
        expert_perf=("expert_perf", "mean")
    ).reset_index()

    # 假设有两个放射科医生，分别计算他们的性能
    # 这里假设 expert_labels 是一个列表，包含两个放射科医生的标注
    def calculate_radiologist_performance(row, radiologist_index):
        # 获取指定放射科医生的标注
        radiologist_label = row["expert_labels"][radiologist_index]
        # 返回是否正确
        return radiologist_label == row["true_labels"]

    # 计算第一个放射科医生的性能
    data_df["radiologist_1_perf"] = data_df.apply(lambda row: calculate_radiologist_performance(row, 0), axis=1)
    # 计算第二个放射科医生的性能
    data_df["radiologist_2_perf"] = data_df.apply(lambda row: calculate_radiologist_performance(row, 1), axis=1)

    # 按患者计算两个放射科医生的平均性能
    patient_perf = data_df.groupby("patient_id").agg(
        num_images=("image_name", "count"),
        radiologist_1_perf=("radiologist_1_perf", "mean"),
        radiologist_2_perf=("radiologist_2_perf", "mean")
    ).reset_index()

    # 创建分层目标变量：将两个放射科医生的性能组合成一个字符串
    patient_perf["stratify_target"] = (
            patient_perf["radiologist_1_perf"].round(2).astype(str) + "_" +
            patient_perf["radiologist_2_perf"].round(2).astype(str)
    )

    # 初始化 StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    # 按患者划分训练、验证和测试集
    patient_ids = patient_perf["patient_id"].values
    stratify_target = patient_perf["stratify_target"].values

    for train_val_idx, test_idx in skf.split(patient_ids, stratify_target):
        # 测试集
        test_patients = patient_ids[test_idx]

        # 使用下一个 2 折作为验证集
        val_idx = train_val_idx[:len(train_val_idx) // (n_splits - 1) * 2]
        val_patients = patient_ids[val_idx]

        # 剩下的用于训练集
        train_idx = train_val_idx[len(train_val_idx) // (n_splits - 1) * 2:]
        train_patients = patient_ids[train_idx]

        # 根据患者分配数据
        train_data = data_df[data_df["patient_id"].isin(train_patients)].to_dict("records")
        val_data = data_df[data_df["patient_id"].isin(val_patients)].to_dict("records")
        test_data = data_df[data_df["patient_id"].isin(test_patients)].to_dict("records")

        folds.append((train_data, val_data, test_data))

    return folds


def eval(dataLoader, allocator, classifier, tlog_file, test=False):
    allocator.eval()
    classifier.eval()
    total_correct = 0
    total_testset = 0

    with torch.no_grad():
        for data in tqdm(dataLoader):
            images, labels, experts_labels = data
            images = images.to(device)
            labels = labels.to(device).to(torch.int64).squeeze(-1)
            experts_labels = experts_labels.to(device).to(torch.int64).squeeze(-1)

            predicts, features = classifier(images)
            weights = allocator(images, features, predicts)

            # 从权重中挑选最大的作为互补专家
            maxW, max_indices = torch.max(weights, dim=1)
            max_indices = max_indices.to(device)

            onehot_label = torch.zeros((labels.shape[0], num_class)).to(device)
            onehot_label[torch.arange(labels.shape[0]), labels] = 1

            expert_onehot_label = torch.zeros((labels.shape[0], num_expert, num_class), dtype=torch.long).to(device)
            # 使用 scatter_ 将对应位置的值设置为 1
            expert_onehot_label.scatter_(2, experts_labels.unsqueeze(-1), 1)
            expert_onehot_label = expert_onehot_label * 10.0

            combined_labels = torch.cat([predicts.unsqueeze(1), expert_onehot_label], dim=1)  # 形状: [batch_size, 1 + num_expert, num_class]
            # 使用 max_indices 挑选结果
            res_label = combined_labels[torch.arange(labels.shape[0]), max_indices]  # 形状: [batch_size, num_class]

            _, predicted = torch.max(res_label.data, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct

            total_testset += labels.shape[0]

    accuracy = 100 * total_correct / total_testset
    print(
        f'Eval round: accuracy={round(accuracy, 4)}%')
    if test:
        save_log(tlog_file, accuracy)
    return accuracy


# transform_train = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),  # 调整为 224x224
#         transforms.ToTensor(),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.5),
#         transforms.RandomApply(
#             [
#              transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1)),
#             ],
#             p=0.4,
#         ),
#         transforms.RandomApply(
#             [
#              transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.01),
#             ],
#             p=0.4,
#         ),
#         transforms.RandomApply(
#             [
#             transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.5)),
#             ],
#             p=0.4,
#         ),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# 自定义窗宽窗位调整
# def apply_window(image, window_center=-600, window_width=1500):
#     min_val = window_center - window_width / 2
#     max_val = window_center + window_width / 2
#     image = torch.clamp(image, min_val, max_val)
#     image = (image - min_val) / (max_val - min_val)  # 归一化到 [0, 1]
#     return image

# transform_train = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),  # 调整为 224x224
#         transforms.ToTensor(),  # 转换为张量
#         transforms.Lambda(lambda img: apply_window(img)),  # 窗宽窗位调整
#         transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 随机缩放和裁剪
#         transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
#         transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
#         transforms.RandomRotation(degrees=15),  # 随机旋转
#         transforms.RandomApply(
#             [transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.02)],  # 添加高斯噪声
#             p=0.5,
#         ),
#         transforms.RandomApply(
#             [transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.5))],  # 高斯模糊
#             p=0.5,
#         ),

#         transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
#     ]
# )

transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整为 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整为 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# transform_test = transforms.Compose(
#     [
#         # transforms.Lambda(lambda img: apply_window(img)),  # 窗宽窗位调整（与训练集一致）
#         transforms.Resize((224, 224)),  # 调整为 224x224
#         transforms.ToTensor(),  # 转换为张量
#         transforms.Lambda(lambda img: apply_window(img)),  # 窗宽窗位调整（与训练集一致）
#         transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]（与训练集一致）
#     ]
# )

for exp_experts_num in [2]:
    for TARGET in ["Airspace opacity", "Pneumothorax", "Fracture"]:
        # for TARGET in ["Pneumothorax", "Fracture"]:
        for ids in [[4295342357, 4295349121], [4323195249, 4295194124], [4295342357, 4295354117], [4323195249, 4295232296]]:
            # for ids in [[4323195249, 4295194124], [4295342357, 4295354117], [4323195249, 4295232296]]:
            our_approach_accuracies = []
            for exp_times in [1, 2, 3, 4, 5]:
                LABELER_IDS = ids
                ############################################## PARAMETER SETTING ##############################################
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                seed = [42, 233, 666, 2025, 3407][exp_times - 1]
                num_expert = exp_experts_num
                num_class = 2
                batch_size = 32
                allocator_learning_rate = 1e-4
                classifier_learning_rate = 1e-3
                epochs = 10
                T_max = epochs // 1
                load = False
                with_scheduler = True
                log_file = os.path.join('./log/NIH-exp/', f'{TARGET}_expert_{LABELER_IDS[0]}_and_{LABELER_IDS[1]}_seed_{seed}_exp_{exp_times}_log.txt')
                print(
                    f"{Fore.CYAN}Experiment Parameters:\n"
                    f"{Fore.CYAN}--------------------------------------------\n"
                    f"{Fore.GREEN}Device: {Fore.YELLOW}{device}\n"
                    f"{Fore.GREEN}Random Seed: {Fore.YELLOW}{seed}\n"
                    f"{Fore.GREEN}Number of Experts: {Fore.YELLOW}{num_expert}\n"
                    f"{Fore.GREEN}Number of Classes: {Fore.YELLOW}{num_class}\n"
                    f"{Fore.GREEN}Batch Size: {Fore.YELLOW}{batch_size}\n"
                    f"{Fore.GREEN}Allocator Learning Rate: {Fore.YELLOW}{allocator_learning_rate} --> {1e-5} (LR Scheduler)\n"
                    f"{Fore.GREEN}Classifier Learning Rate: {Fore.YELLOW}{classifier_learning_rate} --> {1e-5} (LR Scheduler)\n"
                    f"{Fore.GREEN}Epochs: {Fore.YELLOW}{epochs}\n"
                    f"{Fore.GREEN}T_max (Scheduler max steps): {Fore.YELLOW}{T_max}\n"
                    f"{Fore.GREEN}Load Previous Model: {Fore.YELLOW}{load}\n"
                    f"{Fore.GREEN}Log File Path: {Fore.YELLOW}{log_file}\n"
                    f"{Fore.CYAN}--------------------------------------------\n"
                )
                setup_seed(seed=seed)
                root_dir = "../data/NIH"  # 图片根目录
                individual_csv = os.path.join(root_dir, "four_findings_expert_labels_individual_readers.csv")
                test_csv = os.path.join(root_dir, "four_findings_expert_labels_test_labels.csv")
                val_csv = os.path.join(root_dir, "four_findings_expert_labels_validation_labels.csv")

                data_info = load_data_info(individual_csv, test_csv, val_csv, LABELER_IDS, TARGET)
                avg_acc = 0
                # 设置 K 折
                k_folds = 10
                stratified_folds = create_3_stratified_patient_folds(data_info, n_splits=k_folds, seed=seed)
                for fold, (train_data, val_data, test_data) in enumerate(stratified_folds):
                    print(len(train_data), len(val_data), len(test_data))
                    ############################################## LOAD TRAIN DATA ##############################################
                    trainset = CustomDataset(train_data, root_dir=root_dir, transform=transform_train)
                    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
                    ############################################### LOAD VAL DATA ###############################################
                    valset = CustomDataset(val_data, root_dir=root_dir, transform=transform_test)
                    valLoader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
                    ############################################### LOAD TEST DATA ###############################################
                    testset = CustomDataset(test_data, root_dir=root_dir, transform=transform_test)
                    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
                    ########################################## ALLOCATOR AND CLASSIFIER ##########################################
                    classifier = Classifier(num_class).to(device)
                    allocator = AllocatorFour(num_expert, num_class, classifier.hidden_size).to(device)
                    if load:
                        classifier.load_state_dict(torch.load(r'./classifier_nih.pth'))

                    allocator_optimizer = optim.Adam(allocator.parameters(), lr=allocator_learning_rate, weight_decay=1e-5, betas=(0.9, 0.999))
                    classifier_optimizer = optim.Adam(classifier.parameters(), lr=classifier_learning_rate, weight_decay=1e-5, betas=(0.9, 0.999))
                    allocator_scheduler = None
                    classifier_scheduler = None
                    if with_scheduler:
                        allocator_scheduler = optim.lr_scheduler.CosineAnnealingLR(allocator_optimizer, T_max=T_max, eta_min=1e-5, last_epoch=-1)
                        classifier_scheduler = optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, T_max=T_max, eta_min=1e-5, last_epoch=-1)
                    val_best_acc = 0.0
                    best_acc = 0.0
                    best_loss = torch.inf
                    ############################################### START TRAINING ###############################################
                    for epoch in range(epochs):
                        allocator.train()
                        classifier.train()

                        running_loss = 0.0
                        loss_1 = 0.0
                        loss_2 = 0.0
                        loss_3 = 0.0
                        loss_4 = 0.0

                        total_correct = 0
                        machine_total_correct = 0
                        experts_total_correct = 0
                        total_trainset = 0
                        correct_by_machine_wrong_by_expert = 0
                        wrong_by_expert = 0

                        machine_correct_per_class = [0] * num_class
                        machine_total_per_class = [0] * num_class

                        experts_correct_per_class = [0] * num_class
                        experts_total_per_class = [0] * num_class

                        for data in tqdm(trainLoader, desc=f'Fold {fold + 1}/{k_folds}, epoch {epoch + 1}/{epochs}'):
                            images, labels, experts_labels = data
                            images = images.to(device)
                            labels = labels.to(device).to(torch.int64).squeeze(-1)
                            experts_labels = experts_labels.to(device).to(torch.int64).squeeze(-1)

                            predicts, features = classifier(images)
                            weights = allocator(images, features, predicts)

                            # 从权重中挑选最大的作为互补专家
                            maxW, max_indices = torch.max(weights, dim=1)
                            max_indices = max_indices.to(device)

                            onehot_label = torch.zeros((labels.shape[0], num_class)).to(device)
                            onehot_label[torch.arange(labels.shape[0]), labels] = 1

                            expert_onehot_label = torch.zeros((labels.shape[0], num_expert, num_class), dtype=torch.long).to(device)
                            # 使用 scatter_ 将对应位置的值设置为 1
                            expert_onehot_label.scatter_(2, experts_labels.unsqueeze(-1), 1)
                            # 保证专家预测与分类器预测结果分布一致
                            expert_onehot_label = expert_onehot_label * 10.0

                            combined_labels = torch.cat([predicts.unsqueeze(1), expert_onehot_label], dim=1)  # 形状: [batch_size, 1 + num_expert, num_class]
                            # 使用 max_indices 挑选结果
                            res_label = combined_labels[torch.arange(labels.shape[0]), max_indices]  # 形状: [batch_size, num_class]
                            main_loss = F.cross_entropy(res_label, labels)

                            predicts_labels_bool = (predicts.argmax(dim=1) == labels).unsqueeze(1).float()
                            experts_labels_bool = (experts_labels == labels.unsqueeze(1)).float()
                            experts_weak = (experts_labels_bool.sum(dim=1) == 0).unsqueeze(1)
                            combined_labels_bool = torch.cat([experts_weak, experts_labels_bool], dim=1)
                            # 分配器的损失
                            weights_loss = F.binary_cross_entropy_with_logits(weights, combined_labels_bool, reduction='none')

                            gamma = torch.ones_like(weights_loss)  # 先创建一个全为 1 的张量
                            gamma[:, 0] = 1  # 第一列（即第一个损失）使用 weight_first
                            gamma[:, 1:] = 1  # 后续列（即后续损失）使用 weight_rest

                            weights_loss = (weights_loss * gamma).mean()

                            loss = main_loss + weights_loss

                            classifier_optimizer.zero_grad()
                            allocator_optimizer.zero_grad()
                            loss.backward()
                            classifier_optimizer.step()
                            allocator_optimizer.step()

                            running_loss += loss.item()
                            loss_1 += main_loss.item()
                            loss_2 += weights_loss.item()

                            _, predicted = torch.max(res_label.data, 1)
                            correct = (predicted == labels).sum().item()
                            total_correct += correct

                            total_trainset += labels.shape[0]

                        accuracy = 100 * total_correct / total_trainset
                        print(
                            f"Train round: running_loss={[round(running_loss, 4), round(loss_1, 4), round(loss_2, 4)]}, train_accuracy={round(accuracy, 4)}%")

                        acc = eval(valLoader, allocator, classifier, log_file, test=False)
                        if val_best_acc < acc:
                            val_best_acc = acc
                            acc = eval(testLoader, allocator, classifier, log_file, test=True)
                            best_acc = acc

                        if with_scheduler:
                            print(f"Current Learning Rate: {[allocator_scheduler.get_last_lr(), classifier_scheduler.get_last_lr()]}")
                            allocator_scheduler.step()
                            classifier_scheduler.step()
                    avg_acc += best_acc
                    with open(log_file, 'a') as file:
                        file.write(f"--------------------------------------- FOLD {fold + 1} OVER ---------------------------------------\n")
                print(f"{Fore.CYAN} avgerage accuracy: {avg_acc / k_folds}")
                # 记录每一次实验地平均准确率
                our_approach_accuracies.append(avg_acc / k_folds)
            # 计算五次重复实验结果    
            mean_our_approach_accuracy = np.mean(our_approach_accuracies)
            mean_our_approach_accuracy_std = np.std(our_approach_accuracies)
            file_path = f"./log/NIH-exp/result-exp-NIH.txt"
            with open(file_path, 'a') as file:
                file.write(f"TARGET: {TARGET}\n")
                file.write(f"IDS: {LABELER_IDS[0]} AND {LABELER_IDS[1]}\n")
                file.write(f"Ours: {mean_our_approach_accuracy:.8f} {mean_our_approach_accuracy_std:.8f}\n")
                file.write(f"-------------------------------------------------------------------------------\n")

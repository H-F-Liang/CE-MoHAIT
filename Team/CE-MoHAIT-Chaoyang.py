import os
import random
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append('.')
sys.path.append('..')
from Net.Allocator import *
from Net.Classifier import *
from utils.TrainingLog import save_log
from colorama import Fore, Style, init
from PIL import Image
import json
from sklearn.model_selection import KFold
# 初始化 colorama
init(autoreset=True)
# 切换工作目录到当前
os.chdir(os.path.dirname(__file__))


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


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
        # 获取标签
        label = self.data_info[idx]["label"]  # 金标准标签
        expert_labels = torch.tensor([self.data_info[idx]["label_A"], self.data_info[idx]["label_B"]])  # 专家标签

        # 获取图像文件路径
        file_name = os.path.basename(self.data_info[idx]["name"])
        img_name = os.path.join(self.root_dir, str(label), file_name)
        image = Image.open(img_name).convert("RGB")  # 打开图像并转换为 RGB 格式
        if self.transform:
            image = self.transform(image)  # 图像预处理

        return image, label, expert_labels


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

    print(f'Eval round: accuracy={round(accuracy, 4)}%')
    save_log(tlog_file, accuracy)
    return accuracy

        
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整为 224x224
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomApply(
            [
            #  transforms.RandomAffine(degrees=None, scale=(0.85, 1.15), translate=None, shear=(-20, 20)),
             transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.85, 1.15)),
            ],
            p=0.4,
        ),
        transforms.RandomApply(
            [
             transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.01),
            ],
            p=0.4,
        ),
        transforms.RandomApply(
            [
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0)),
            ],
            p=0.4,
        ),
        transforms.Normalize((0.6470, 0.5523, 0.6694), (0.1751, 0.2041, 0.1336))
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整为 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.6470, 0.5523, 0.6694), (0.1751, 0.2041, 0.1336))
    ]
)

for exp_experts_num in [2]:
    ours_accuracies = []
    for exp_times in [1, 2, 3, 4]:
        ############################################## PARAMETER SETTING ##############################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed = [42, 233, 666, 3407][exp_times - 1]
        num_expert = exp_experts_num  # 2 4 6 8 10
        num_class = 4
        batch_size = 64
        allocator_learning_rate = 1e-4
        classifier_learning_rate = 1e-3
        epochs = 20
        T_max = epochs // 4
        load = False
        with_scheduler = True
        log_file = os.path.join('./log/Chaoyang-exp', f'num_expert_{num_expert}_seed_{seed}_exp_{exp_times}-2.txt')
        print(
            f"{Fore.CYAN}Experiment Parameters:\n"
            f"{Fore.CYAN}--------------------------------------------\n"
            f"{Fore.GREEN}Device: {Fore.YELLOW}{device}\n"
            f"{Fore.GREEN}Random Seed: {Fore.YELLOW}{seed}\n"
            f"{Fore.GREEN}Number of Experts: {Fore.YELLOW}{num_expert}\n"
            f"{Fore.GREEN}Number of Classes: {Fore.YELLOW}{num_class}\n"
            f"{Fore.GREEN}Batch Size: {Fore.YELLOW}{batch_size}\n"
            f"{Fore.GREEN}Allocator Learning Rate: {Fore.YELLOW}{allocator_learning_rate} --> {0} (LR Scheduler)\n"
            f"{Fore.GREEN}Classifier Learning Rate: {Fore.YELLOW}{classifier_learning_rate} --> {0} (LR Scheduler)\n"
            f"{Fore.GREEN}Epochs: {Fore.YELLOW}{epochs}\n"
            f"{Fore.GREEN}T_max (Scheduler max steps): {Fore.YELLOW}{T_max}\n"
            f"{Fore.GREEN}Load Previous Model: {Fore.YELLOW}{load}\n"
            f"{Fore.GREEN}Log File Path: {Fore.YELLOW}{log_file}\n"
            f"{Fore.CYAN}--------------------------------------------\n"
        )
        setup_seed(seed=seed)
        # 读取 JSON 文件
        with open('../data/chaoyang/train_label.json', 'r') as f:
            data_info = json.load(f)

        root_dir = "../data/chaoyang"
        avg_acc = 0
        # 设置 K 折
        k_folds = 10
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(kf.split(data_info)):
            ############################################## LOAD TRAIN DATA ##############################################
            trainset = Subset(CustomDataset(data_info, root_dir=root_dir, transform=transform_train), train_idx)
            trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

            ############################################### LOAD TEST DATA ###############################################
            testset = Subset(CustomDataset(data_info, root_dir=root_dir, transform=transform_test), test_idx)
            testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

            ########################################## ALLOCATOR AND CLASSIFIER ##########################################
            classifier = Classifier(num_class).to(device)
            allocator = AllocatorFour(num_expert, num_class, classifier.hidden_size).to(device)
            if load:
                classifier.load_state_dict(torch.load(r'./classifier_chaoyang.pth'))

            allocator_optimizer = optim.Adam(allocator.parameters(), lr=allocator_learning_rate, weight_decay=1e-6)
            classifier_optimizer = optim.Adam(classifier.parameters(), lr=classifier_learning_rate, weight_decay=1e-6)
            allocator_scheduler = None
            classifier_scheduler = None
            if with_scheduler:
                allocator_scheduler = optim.lr_scheduler.CosineAnnealingLR(allocator_optimizer, T_max=T_max, eta_min=0, last_epoch=-1)
                classifier_scheduler = optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, T_max=T_max, eta_min=0, last_epoch=-1)
            best_acc = 0.0
            best_loss = torch.inf
            ############################################### START TRAINING ###############################################
            for epoch in range(epochs):
                allocator.train()
                classifier.train()
                running_loss = 0.0
                loss_1 = 0.0
                loss_2 = 0.0

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

                for data in tqdm(trainLoader, desc=f'Seed {seed}, Fold {fold + 1}/{k_folds}, epoch {epoch + 1}/{epochs}'):
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
                print(f"Train round: running_loss={round(running_loss, 4), round(loss_1, 4), round(loss_2, 4)}, train_accuracy={round(accuracy, 4)}%")

                acc = eval(testLoader, allocator, classifier, log_file, test=False)
                if epoch == epochs - 1:
                    best_acc = acc
                    acc = eval(testLoader, allocator, classifier, log_file, test=True)
    
                if with_scheduler:
                    print(f"Current Learning Rate: {[allocator_scheduler.get_last_lr(), classifier_scheduler.get_last_lr()]}")
                    allocator_scheduler.step()
                    classifier_scheduler.step()
            avg_acc += best_acc
        print(f"{Fore.CYAN} avgerage accuracy: {avg_acc / k_folds}")
        ours_accuracies.append(avg_acc / k_folds)
    mean_ours_accuracy, std_ours_accuracy = np.mean(ours_accuracies), np.std(ours_accuracies)
    file_path = f"./log/Chaoyang-exp/ours-exp-chaoyang.txt"
    # 将数据追加写入文件
    with open(file_path, 'a') as file:
        file.write(f"Ours: {mean_ours_accuracy} {std_ours_accuracy}\n")


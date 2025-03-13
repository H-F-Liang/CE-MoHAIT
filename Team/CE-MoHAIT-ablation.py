import os
import random
import numpy as np
import torchvision
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

# 初始化 colorama
init(autoreset=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomCIFAR100(Dataset):
    def __init__(self, original_dataset, extra_labels):
        self.original_dataset = original_dataset  # 原始数据集
        self.extra_labels = extra_labels  # 额外的标签

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始数据集的图片和标签
        image, label = self.original_dataset[idx]
        # 获取额外的标签
        extra_label = self.extra_labels[idx]

        # 返回图片、原始标签和额外标签
        return image, label, extra_label


def generate_ability_IJCAI(num_class, mean=70, std_dev=5):
    """
    生成每个专家对每个类别的能力。

    参数:
        num_class (int): 类别的总数量（通常为 100）。
        mean (float): 专家完美预测类别的平均数量，默认值为 70。
        std_dev (float): 专家完美预测类别数量的标准差，默认值为 5。

    返回:
        np.ndarray: 每个类别的能力分布，0 或 1，表示专家是否能够完美预测该类别。
    """
    # 从正态分布 N(70, 5) 中抽取一个值作为该专家能够完美预测的类别数量
    num_perfect_predictions = int(np.clip(np.random.normal(mean, std_dev), 0, num_class))

    # 创建一个全 0 的能力数组
    abilities = np.zeros(num_class)

    # 随机选择 num_perfect_predictions 个类别，设置其能力为 1
    perfect_indices = np.random.choice(num_class, num_perfect_predictions, replace=False)
    abilities[perfect_indices] = 1

    return abilities


# 模拟每个专家为数据生成固定的标签
def labeling_IJCAI(dataset, expert_ability, num_class):
    Y = []
    ability = expert_ability
    for data in tqdm(dataset):
        images, label = data
        if random.random() < ability[label]:
            Y.append(label)
        else:
            s = list(set(range(num_class)) - {label})
            idx = random.choice(s)
            Y.append(idx)
    return Y


def max_min_alignment(target, onehot_label):
    """
    全局-最大最小对齐
    参数:
        predicts (torch.Tensor): 预测的类别概率，形状为 [batch_size, num_class]。
        expert_onehot_label (torch.Tensor): 专家标签的 one-hot 编码，形状为 [batch_size, num_expert, num_class]。
    返回:
        torch.Tensor: 最大最小对齐后的张量，形状为 [batch_size, num_expert, num_class]。
    """
    min_val = target.min()
    max_val = target.max()
    return min_val + (max_val - min_val) * onehot_label


def test(dataLoader, allocator, classifier, tlog_file, test=False):
    allocator.eval()
    classifier.eval()
    total_correct = 0
    machine_total_correct = 0
    experts_total_correct = 0
    total_testset = 0
    wrong_by_expert = 0
    correct_by_machine_wrong_by_expert = 0

    machine_correct_per_class = [0] * num_class
    machine_total_per_class = [0] * num_class

    experts_correct_per_class = [0] * num_class
    experts_total_per_class = [0] * num_class

    with torch.no_grad():
        for data in tqdm(dataLoader):
            images, labels, experts_labels = data
            images = images.to(device)
            labels = labels.to(device)
            experts_labels = experts_labels.to(device)

            predicts, features = classifier(images)
            weights = allocator(images, features, predicts)

            # 从权重中挑选最大的作为互补专家
            maxW, max_indices = torch.max(torch.softmax(weights, dim=1), dim=1)
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
    print(f'Val round: accuracy={round(accuracy, 4)}')
    if test:
        save_log(tlog_file, accuracy, 0, 0, 0)
    return accuracy


transform_train = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),  # 调整为 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整为 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)
for exp_experts_num in [4]:
    for exp_experts_ability in [(75, 5)]:
        our_approach_accuracies = []
        our_approach_machine_accuracies = []
        our_approach_expert_accuracies = []
        for exp_times in [1, 2, 3, 4, 5]:
            ############################################## PARAMETER SETTING ##############################################
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            seed = [42, 233, 666, 2025, 3407][exp_times - 1]
            num_expert = exp_experts_num  # 2 4 6 8 10
            mean, std_dev = exp_experts_ability
            num_class = 100
            batch_size = 128
            allocator_learning_rate = 2e-4
            classifier_learning_rate = 2e-4
            epochs = 20
            T_max = epochs // 1
            load = False
            root_dir = os.path.join('', f'./log/CIFAR-100-exp/')
            log_file = os.path.join(root_dir, f'1-num_expert_{num_expert}_ability_{mean}-{exp_times}.txt')
            print(
                f"{Fore.CYAN}Experiment Parameters:\n"
                f"{Fore.CYAN}--------------------------------------------\n"
                f"{Fore.GREEN}Device: {Fore.YELLOW}{device}\n"
                f"{Fore.GREEN}Random Seed: {Fore.YELLOW}{seed}\n"
                f"{Fore.GREEN}Number of Experts: {Fore.YELLOW}{num_expert}\n"
                f"{Fore.GREEN}Experts Ability: {Fore.YELLOW}N(mean={mean}, std_dev={std_dev})\n"
                f"{Fore.GREEN}Number of Classes: {Fore.YELLOW}{num_class}\n"
                f"{Fore.GREEN}Batch Size: {Fore.YELLOW}{batch_size}\n"
                f"{Fore.GREEN}Allocator Learning Rate: {Fore.YELLOW}{allocator_learning_rate} --> {1e-6} (LR Scheduler)\n"
                f"{Fore.GREEN}Classifier Learning Rate: {Fore.YELLOW}{classifier_learning_rate} --> {1e-6} (LR Scheduler)\n"
                f"{Fore.GREEN}Epochs: {Fore.YELLOW}{epochs}\n"
                f"{Fore.GREEN}T_max (Scheduler max steps): {Fore.YELLOW}{T_max}\n"
                f"{Fore.GREEN}Load Previous Model: {Fore.YELLOW}{load}\n"
                f"{Fore.GREEN}Log File Path: {Fore.YELLOW}{log_file}\n"
                f"{Fore.CYAN}--------------------------------------------\n"
            )
            setup_seed(seed=seed)
            ############################################## GENERATE EXPERTS ##############################################
            # 对比时采用IJCAI2022的专家设置 专家数量 4, 能力采样 N(70, 5)
            # 其余实验专家数量[2 4 6 8], 能力采样 N(25, 5)
            experts = [generate_ability_IJCAI(num_class, mean=mean, std_dev=std_dev) for i in range(num_expert)]
            experts_team_upperbound = np.clip(np.array(experts).sum(axis=0), 0, 1).mean()
            with open(os.path.join(root_dir, f'exp_num_expert_{num_expert}_upperbound-{exp_times}.txt'), 'w') as file:
                file.write(str(experts_team_upperbound))
            # print(np.clip(np.array(experts).sum(axis=0), 0, 1).mean())
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            open(log_file, 'a').close()

            ############################################## LOAD TRAIN / VAL DATA ##############################################
            train_val_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)

            # 从训练集中划分出验证集和新的训练集
            train_size = 40000  # 新的训练集大小
            val_size = 10000  # 验证集大小
            test_size = 10000  # 测试集大小

            # 随机选择训练集中的样本索引
            total_train_indices = list(range(50000))  # 原始训练集大小是50000
            random.shuffle(total_train_indices)

            # 划分训练集和验证集
            train_indices = total_train_indices[:train_size]
            val_indices = total_train_indices[train_size:train_size + val_size]

            # 创建新的训练集和验证集
            trainset = Subset(train_val_set, train_indices)
            valset = Subset(train_val_set, val_indices)

            # 为每个专家生成标签
            experts_labels_train = []
            experts_labels_val = []
            print("Generates training set expert labels ...")
            for i, e in enumerate(experts):
                print(f"Expert {i} ...")
                experts_labels_train.append(labeling_IJCAI(dataset=trainset, expert_ability=e, num_class=num_class))
            print("Generates validation set expert labels ...")
            for i, e in enumerate(experts):
                print(f"Expert {i} ...")
                experts_labels_val.append(labeling_IJCAI(dataset=valset, expert_ability=e, num_class=num_class))

            # 转换成tensor
            experts_labels_train = torch.tensor(experts_labels_train).T
            experts_labels_val = torch.tensor(experts_labels_val).T

            # 将专家标签加入到训练集和验证集中
            trainset = CustomCIFAR100(trainset, extra_labels=experts_labels_train)
            valset = CustomCIFAR100(valset, extra_labels=experts_labels_val)

            # 创建数据加载器
            trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            valLoader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

            ############################################### LOAD TEST DATA ###############################################
            testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)

            # 为每个专家生成测试集标签
            experts_labels_test = []
            print("Generates test set expert labels ...")
            for i, e in enumerate(experts):
                print(f"Expert {i} ...")
                experts_labels_test.append(labeling_IJCAI(dataset=testset, expert_ability=e, num_class=num_class))

            # 转换成tensor
            experts_labels_test = torch.tensor(experts_labels_test).T

            # 将专家标签加入到测试集中
            testset = CustomCIFAR100(testset, extra_labels=experts_labels_test)

            # 创建测试集数据加载器
            testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

            ########################################## ALLOCATOR AND CLASSIFIER ##########################################
            classifier = nn.DataParallel(Classifier(num_class).to(device))
            allocator = nn.DataParallel(AllocatorFour(num_expert, num_class, classifier.module.hidden_size).to(device))
            if load:
                classifier.load_state_dict(torch.load(r'./classifier_cifar-100.pth'))

            allocator_optimizer = optim.Adam(allocator.parameters(), lr=allocator_learning_rate, weight_decay=1e-6)
            classifier_optimizer = optim.Adam(classifier.parameters(), lr=classifier_learning_rate, weight_decay=1e-6)
            allocator_scheduler = optim.lr_scheduler.CosineAnnealingLR(allocator_optimizer, T_max=T_max, eta_min=1e-6, last_epoch=-1)
            classifier_scheduler = optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, T_max=T_max, eta_min=1e-6, last_epoch=-1)

            ############################################### START TRAINING ###############################################
            val_acc_best = 0.0
            test_acc = 0.0
            test_machine_acc = 0.0
            test_expert_acc = 0.0
            for epoch in range(epochs):
                allocator.train()
                classifier.train()

                if epoch >= 40:
                    for param in allocator.network.parameters():
                        param.requires_grad = False
                    for param in classifier.network.parameters():
                        param.requires_grad = False

                running_loss = 0.0
                loss_1 = 0.0
                loss_2 = 0.0

                total_correct = 0
                total_trainset = 0

                for data in tqdm(trainLoader):
                    images, labels, experts_labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    experts_labels = experts_labels.to(device)

                    predicts, features = classifier(images)
                    weights = allocator(images, features, predicts)

                    # 从权重中挑选最大的作为互补专家
                    maxW, max_indices = torch.max(torch.softmax(weights, dim=1), dim=1)
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
                    res_label = combined_labels[torch.arange(labels.shape[0]), max_indices] * maxW.unsqueeze(-1)  # 形状: [batch_size, num_class]
                    main_loss = F.cross_entropy(res_label, labels)

                    # predicts_labels_bool = (predicts.argmax(dim=1) == labels).unsqueeze(1).float()
                    # experts_labels_bool = (experts_labels == labels.unsqueeze(1)).float()
                    # experts_weak = (experts_labels_bool.sum(dim=1) == 0).unsqueeze(1)
                    # combined_labels_bool = torch.cat([experts_weak, experts_labels_bool], dim=1)
                    # # 分配器的损失
                    # weights_loss = F.binary_cross_entropy_with_logits(weights, combined_labels_bool)

                    loss = main_loss  # + weights_loss

                    classifier_optimizer.zero_grad()
                    allocator_optimizer.zero_grad()
                    loss.backward()
                    classifier_optimizer.step()
                    allocator_optimizer.step()

                    running_loss += loss.item()
                    loss_1 += main_loss.item()
                    loss_2 += 0

                    _, predicted = torch.max(res_label.data, 1)
                    correct = (predicted == labels).sum().item()
                    total_correct += correct

                    total_trainset += labels.shape[0]

                accuracy = 100 * total_correct / total_trainset
                print(f"Train round, epoch {epoch}: running_loss={[round(running_loss, 4), round(loss_1, 4), round(loss_2, 4)]}, train_accuracy={round(accuracy, 4)}%")

                acc = test(valLoader, allocator, classifier, log_file, False)
                if val_acc_best < acc:
                    val_acc_best = acc
                    acc = test(testLoader, allocator, classifier, log_file, True)
                    test_acc = acc

                print(f"Current Learning Rate: {[allocator_scheduler.get_last_lr(), classifier_scheduler.get_last_lr()]}")
                allocator_scheduler.step()
                classifier_scheduler.step()

            our_approach_accuracies.append(test_acc)
            our_approach_machine_accuracies.append(test_machine_acc)
            our_approach_expert_accuracies.append(test_expert_acc)

        mean_our_approach_accuracy = np.mean(our_approach_accuracies)
        mean_our_approach_accuracy_std = np.std(our_approach_accuracies)
        file_path = f"./log/CIFAR-100-exp/result-exp-CIFAR-100-1.txt"
        with open(file_path, 'a') as file:
            file.write(f"EXPERT NUMBER: {exp_experts_num}\n")
            file.write(f"EXPERT ABILITY: ({exp_experts_ability[0]}, {exp_experts_ability[1]})\n")
            file.write(f"Ours-1 acc: {mean_our_approach_accuracy:.8f} {mean_our_approach_accuracy_std:.8f}\n")
            file.write(f"-------------------------------------------------------------------------------\n")

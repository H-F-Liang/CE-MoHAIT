import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 4
DROPOUT = 0.2
NUM_HIDDEN_UNITS = 128
LR = 1e-3
USE_LR_SCHEDULER = True
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 20
K_FOLD = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 自定义chaoyang数据集读取 返回金标准和专家标签
class Chaoyang_Dataset(Dataset):
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


class Chaoyang_K_Fold_Dataloader:
    def __init__(self, train_idx, test_idx, train_batch_size=128, test_batch_size=128, seed=42):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
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
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.5)),
                    ],
                    p=0.4,
                ),
                transforms.Normalize((0.6470, 0.5523, 0.6694), (0.1751, 0.2041, 0.1336))
            ])

        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # 调整为 224x224
                transforms.ToTensor(),
                transforms.Normalize((0.6470, 0.5523, 0.6694), (0.1751, 0.2041, 0.1336))
            ]
        )

        ############################################## LOAD TRAIN DATA ##############################################
        trainset = torch.utils.data.Subset(Chaoyang_Dataset(data_info, root_dir="../data/chaoyang", transform=transform_train), train_idx)
        self.trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True, drop_last=True)

        ############################################### LOAD TEST DATA ###############################################
        testset = torch.utils.data.Subset(Chaoyang_Dataset(data_info, root_dir="../data/chaoyang", transform=transform_test), test_idx)
        self.testLoader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, drop_last=False)

    def get_data_loader(self):
        return self.trainLoader, self.testLoader


def joint_sparse_framework_loss(epoch, classifier_output, allocation_system_output, expert_preds, targets):
    # Input:
    #   epoch: int = current epoch (used for epoch-dependent weighting of allocation system loss)
    #   classifier_output: softmax probabilities as class probabilities,  nxm matrix with n=batch size, m=number of classes
    #   allocation_system_output: sigmoid outputs as expert weights,  nx(m+1) matrix with n=batch size, m=number of experts + 1 for machine
    #   expert_preds: nxm matrix with expert predictions with n=number of experts, m=number of classes
    #   targets: targets as 1-dim vector with n length with n=batch_size

    # loss for allocation system 

    # set up zero-initialized tensor to store weighted team predictions
    batch_size = len(targets)
    weighted_team_preds = torch.zeros((batch_size, NUM_CLASSES)).to(classifier_output.device)

    # for each team member add the weighted prediction to the team prediction
    # start with machine
    weighted_team_preds = weighted_team_preds + allocation_system_output[:, 0].reshape(-1, 1) * classifier_output
    # continue with human experts
    for idx in range(NUM_EXPERTS):
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[expert_preds[idx].astype(int)]).to(classifier_output.device)
        weighted_team_preds = weighted_team_preds + allocation_system_output[:, idx + 1].reshape(-1, 1) * one_hot_expert_preds

    # calculate team probabilities using softmax
    team_probs = nn.Softmax(dim=1)(weighted_team_preds)

    # alpha2 is 1-epoch^0.5 (0.5 taken from code of preprint paper) <--- used for experiments
    alpha2 = 1 - (epoch ** -0.5)
    alpha2 = torch.tensor(alpha2).to(classifier_output.device)

    # weight the negative log likelihood loss with alpha2 to get team loss
    log_team_probs = torch.log(team_probs + 1e-7)
    allocation_system_loss = nn.NLLLoss(reduction="none")(log_team_probs, targets.long())
    allocation_system_loss = torch.mean(alpha2 * allocation_system_loss)

    # loss for classifier

    alpha1 = 1
    log_classifier_output = torch.log(classifier_output + 1e-7)
    classifier_loss = nn.NLLLoss(reduction="none")(log_classifier_output, targets.long())
    classifier_loss = alpha1 * torch.mean(classifier_loss)

    # combine both losses
    system_loss = classifier_loss + allocation_system_loss

    return system_loss


def our_loss(epoch, classifier_output, allocation_system_output, expert_preds, targets):
    # Input:
    #   epoch: int = current epoch (not used)
    #   classifier_output: softmax probabilities as class probabilities,  nxm matrix with n=batch size, m=number of classes
    #   allocation_system_output: softmax outputs as weights,  nx(m+1) matrix with n=batch size, m=number of experts + 1 for machine
    #   expert_preds: nxm matrix with expert predictions with n=number of experts, m=number of classes
    #   targets: targets as 1-dim vector with n length with n=batch_size

    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, NUM_CLASSES)).to(classifier_output.device)  # set up zero-initialized tensor to store team predictions
    team_probs = team_probs + allocation_system_output[:, 0].reshape(-1, 1) * classifier_output  # add the weighted classifier prediction to the team prediction
    for idx in range(NUM_EXPERTS):  # continue with human experts
        if isinstance(expert_preds[idx], np.ndarray):
            index = torch.tensor(expert_preds[idx]).clone().detach()
        else:
            index = expert_preds[idx]
        index = index.int()
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[index.cpu()]).to(classifier_output.device)
        # print(allocation_system_output[:, idx + 1].reshape(-1, 1).shape, one_hot_expert_preds.shape, expert_preds.shape, index.shape)
        team_probs = team_probs + allocation_system_output[:, idx + 1].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    system_loss = nn.NLLLoss()(log_output, targets)

    return system_loss


def mixture_of_ai_experts_loss(allocation_system_output, classifiers_outputs, targets):
    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, NUM_CLASSES)).to(allocation_system_output.device)
    classifiers_outputs = classifiers_outputs.to(allocation_system_output.device)

    for idx in range(NUM_EXPERTS + 1):
        team_probs = team_probs + allocation_system_output[:, idx].reshape(-1, 1) * classifiers_outputs[idx]

    log_output = torch.log(team_probs + 1e-7)
    moae_loss = nn.NLLLoss()(log_output, targets)

    return moae_loss


def mixture_of_human_experts_loss(allocation_system_output, human_expert_preds, targets):
    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, NUM_CLASSES)).to(allocation_system_output.device)

    # human experts
    for idx in range(NUM_EXPERTS):
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[human_expert_preds[idx]]).to(allocation_system_output.device)
        team_probs = team_probs + allocation_system_output[:, idx].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    mohe_loss = nn.NLLLoss()(log_output, targets)

    return mohe_loss


class Resnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        del self.resnet.fc

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.training = False

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        features = torch.flatten(x, 1)
        return features


class Network(nn.Module):
    def __init__(self, output_size, softmax_sigmoid="softmax"):
        super().__init__()
        self.softmax_sigmoid = softmax_sigmoid

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_UNITS, output_size)
        )

    def forward(self, features):
        output = self.classifier(features)
        if self.softmax_sigmoid == "softmax":
            output = nn.Softmax(dim=1)(output)
        elif self.softmax_sigmoid == "sigmoid":
            output = nn.Sigmoid()(output)
        return output


def get_accuracy(preds, targets):
    if len(targets) > 0:
        acc = accuracy_score(targets, preds)
    else:
        acc = 0

    return acc


def get_coverage(task_subset_targets, targets):
    num_images = len(targets)
    num_images_in_task_subset = len(task_subset_targets)
    coverage = num_images_in_task_subset / num_images

    return coverage


def get_classifier_metrics(classifier_preds, allocation_system_decisions, targets):
    # classifier performance on all tasks
    classifier_accuracy = get_accuracy(classifier_preds, targets)

    # filter for subset of tasks that are allocated to the classifier
    task_subset = (allocation_system_decisions == 0)

    # classifier performance on those tasks
    task_subset_classifier_preds = classifier_preds[task_subset]
    task_subset_targets = targets[task_subset]
    classifier_task_subset_accuracy = get_accuracy(task_subset_classifier_preds, task_subset_targets)

    # coverage
    classifier_coverage = get_coverage(task_subset_targets, targets)

    return classifier_accuracy, classifier_task_subset_accuracy, classifier_coverage


def get_experts_metrics(expert_preds, allocation_system_decisions, targets):
    expert_accuracies = []
    expert_task_subset_accuracies = []
    expert_coverages = []

    # calculate metrics for each expert
    for expert_idx in range(NUM_EXPERTS):
        # expert performance on all tasks
        preds = expert_preds[expert_idx]
        expert_accuracy = get_accuracy(preds, targets)

        # filter for subset of tasks that are allocated to the expert with number "idx"
        task_subset = (allocation_system_decisions == expert_idx + 1)

        # expert performance on tasks assigned by allocation system
        task_subset_expert_preds = preds[task_subset]
        task_subset_targets = targets[task_subset]
        expert_task_subset_accuracy = get_accuracy(task_subset_expert_preds, task_subset_targets)

        # coverage
        expert_coverage = get_coverage(task_subset_targets, targets)

        expert_accuracies.append(expert_accuracy)
        expert_task_subset_accuracies.append(expert_task_subset_accuracy)
        expert_coverages.append(expert_coverage)

    return expert_accuracies, expert_task_subset_accuracies, expert_coverages


def get_metrics(epoch, allocation_system_outputs, classifier_outputs, expert_preds, targets, loss_fn):
    metrics = {}

    # Metrics for system
    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifier_preds = np.argmax(classifier_outputs, 1)
    preds = np.vstack((classifier_preds, expert_preds)).T
    # print(preds.shape, allocation_system_decisions.astype(int).shape)
    # print(np.max(preds), np.max(allocation_system_decisions.astype(int)))
    system_preds = preds[range(len(preds)), allocation_system_decisions.astype(int)]
    system_accuracy = get_accuracy(system_preds, targets)

    system_loss = loss_fn(epoch=epoch,
                          classifier_output=torch.tensor(classifier_outputs).float(),
                          allocation_system_output=torch.tensor(allocation_system_outputs).float(),
                          expert_preds=expert_preds,
                          targets=torch.tensor(targets).long())

    metrics["System Accuracy"] = system_accuracy
    metrics["System Loss"] = system_loss

    # Metrics for classifier
    classifier_accuracy, classifier_task_subset_accuracy, classifier_coverage = get_classifier_metrics(classifier_preds, allocation_system_decisions, targets)
    metrics["Classifier Accuracy"] = classifier_accuracy
    metrics["Classifier Task Subset Accuracy"] = classifier_task_subset_accuracy
    metrics["Classifier Coverage"] = classifier_coverage

    # Metrics for experts 
    """expert_accuracies, experts_task_subset_accuracies, experts_coverages = get_experts_metrics(expert_preds, allocation_system_decisions, targets)

    for expert_idx, (expert_accuracy, expert_task_subset_accuracy, expert_coverage) in enumerate(zip(expert_accuracies, experts_task_subset_accuracies, experts_coverages)):
        metrics[f'Expert {expert_idx+1} Accuracy'] = expert_accuracy
        metrics[f'Expert {expert_idx+1} Task Subset Accuracy'] = expert_task_subset_accuracy
        metrics[f'Expert {expert_idx+1} Coverage'] = expert_coverage"""

    return system_accuracy, system_loss, metrics


def train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, loss_fn):
    feature_extractor.train()
    classifier.train()
    allocation_system.train()

    for i, (batch_input, batch_labels, batch_expert_labels) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_labels = batch_labels.to(device)
        batch_expert_labels = batch_expert_labels.to(device)

        batch_features = feature_extractor(batch_input)
        batch_outputs_classifier = classifier(batch_features)
        batch_outputs_allocation_system = allocation_system(batch_features)

        batch_loss = loss_fn(epoch=epoch, classifier_output=batch_outputs_classifier, allocation_system_output=batch_outputs_allocation_system,
                             expert_preds=batch_expert_labels.permute(1, 0).cpu().numpy(), targets=batch_labels)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if USE_LR_SCHEDULER:
            scheduler.step()


def evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, data_loader, loss_fn):
    feature_extractor.eval()
    classifier.eval()
    allocation_system.eval()

    classifier_outputs = torch.tensor([]).to(device)
    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    expert_preds = torch.tensor([]).to(device)

    with torch.no_grad():
        for i, (batch_input, batch_labels, batch_expert_labels) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_labels = batch_labels.to(device)
            batch_expert_labels = batch_expert_labels.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)
            batch_allocation_system_outputs = allocation_system(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))

            targets = torch.cat((targets, batch_labels))
            expert_preds = torch.cat((expert_preds, batch_expert_labels))

    classifier_outputs = classifier_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    expert_preds = expert_preds.permute(1, 0).cpu().numpy()

    system_accuracy, system_loss, metrics = get_metrics(epoch, allocation_system_outputs, classifier_outputs, expert_preds, targets, loss_fn)

    return system_accuracy, system_loss, metrics


def run_team_performance_optimization(method, data_info, seed):
    print(f'Team Performance Optimization with {method}')

    if method == "Joint Sparse Framework":
        loss_fn = joint_sparse_framework_loss
        allocation_system_activation_function = "sigmoid"
    elif method == "Our Approach":
        loss_fn = our_loss
        allocation_system_activation_function = "softmax"
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    avg_acc = 0
    avg_cov = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_info)):
        print(f"Fold {fold}:")

        feature_extractor = Resnet().to(device)

        classifier = Network(output_size=NUM_CLASSES,
                             softmax_sigmoid="softmax").to(device)

        allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                    softmax_sigmoid=allocation_system_activation_function).to(device)

        cifar_dl = Chaoyang_K_Fold_Dataloader(train_idx=train_idx, test_idx=test_idx, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed)
        train_loader, test_loader = cifar_dl.get_data_loader()

        parameters = list(classifier.parameters()) + list(allocation_system.parameters())
        optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

        best_metrics = None

        for epoch in tqdm(range(1, EPOCHS + 1)):
            train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, loss_fn)
            test_system_accuracy, test_system_loss, test_metrics = evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, test_loader, loss_fn)
            best_metrics = test_metrics

            system_metrics_keys = [key for key in best_metrics.keys() if "System" in key]
            for k in system_metrics_keys:
                print(f'\t {k}: {best_metrics[k]}')
            print()

        print(f'\n Earlystopping Results for {method}:')
        system_metrics_keys = [key for key in best_metrics.keys() if "System" in key]
        for k in system_metrics_keys:
            print(f'\t {k}: {best_metrics[k]}')
        print()

        classifier_metrics_keys = [key for key in best_metrics.keys() if "Classifier" in key]
        for k in classifier_metrics_keys:
            print(f'\t {k}: {best_metrics[k]}')
        print()

        """for exp_idx in range(NUM_EXPERTS):
        expert_metrics_keys = [key for key in best_metrics.keys() if f'Expert {exp_idx+1} ' in key]
        for k in expert_metrics_keys:
            print(f'\t {k}: {best_metrics[k]}')
        print()"""
        avg_acc += best_metrics["System Accuracy"]
        avg_cov += best_metrics["Classifier Coverage"]
    return avg_acc / K_FOLD, avg_cov / K_FOLD


def get_accuracy_of_best_expert(seed):
    print("Best expert:")
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    avg_acc = 0
    for fold, (train_idx, test_idx) in tqdm(enumerate(kf.split(data_info))):
        cifar_dl = Chaoyang_K_Fold_Dataloader(train_idx=train_idx, test_idx=test_idx, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed)
        train_loader, test_loader = cifar_dl.get_data_loader()

        targets = torch.tensor([]).long()
        expert_preds = torch.tensor([])

        with torch.no_grad():
            for i, (_, batch_labels, batch_expert_labels) in enumerate(test_loader):
                targets = torch.cat((targets, batch_labels))
                expert_preds = torch.cat((expert_preds, batch_expert_labels))
        expert_preds = expert_preds.permute(1, 0)
        expert_accuracies = []
        for idx in range(NUM_EXPERTS):
            preds = expert_preds[idx]
            acc = accuracy_score(targets, preds)
            expert_accuracies.append(acc)

        avg_acc += max(expert_accuracies)

        print(f'Best Expert Accuracy: {max(expert_accuracies)}\n')

    return avg_acc / K_FOLD


def get_accuracy_of_average_expert(seed):
    print("Average expert:")
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    avg_acc = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_info)):
        cifar_dl = Chaoyang_K_Fold_Dataloader(train_idx=train_idx, test_idx=test_idx, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed)
        train_loader, test_loader = cifar_dl.get_data_loader()

        targets = torch.tensor([]).long()
        expert_preds = torch.tensor([])

        with torch.no_grad():
            for i, (_, batch_labels, batch_expert_labels) in enumerate(test_loader):
                targets = torch.cat((targets, batch_labels))
                expert_preds = torch.cat((expert_preds, batch_expert_labels))
        expert_preds = expert_preds.permute(1, 0)
        # 交替专家交替预测即 avg_expert
        avg_expert_preds = [None] * targets.shape[0]
        for idx, expert_pred in enumerate(expert_preds):
            avg_expert_preds[idx::NUM_EXPERTS] = expert_pred[idx::NUM_EXPERTS]

        avg_acc += accuracy_score(targets, avg_expert_preds)
    print(f'Average Expert Accuracy: {avg_acc / K_FOLD}\n')

    return avg_acc / K_FOLD


def train_full_automation_one_epoch(feature_extractor, classifier, train_loader, optimizer, scheduler):
    # switch to train mode
    feature_extractor.train()
    classifier.train()

    for i, (batch_input, batch_targets, _) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        batch_features = feature_extractor(batch_input)
        batch_outputs_classifier = classifier(batch_features)

        log_output = torch.log(batch_outputs_classifier + 1e-7)
        batch_loss = nn.NLLLoss()(log_output, batch_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if USE_LR_SCHEDULER:
            scheduler.step()


def evaluate_full_automation_one_epoch(feature_extractor, classifier, data_loader):
    feature_extractor.eval()
    classifier.eval()

    classifier_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    filenames = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_filenames) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            targets = torch.cat((targets, batch_targets))
            filenames.extend(batch_filenames)

    log_output = torch.log(classifier_outputs + 1e-7)
    full_automation_loss = nn.NLLLoss()(log_output, targets.long())

    classifier_outputs = classifier_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    classifier_preds = np.argmax(classifier_outputs, 1)
    full_automation_accuracy = get_accuracy(classifier_preds, targets)

    return full_automation_accuracy, full_automation_loss


def run_full_automation(seed):
    print(f'Training full automation baseline')
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    avg_acc = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_info)):

        feature_extractor = Resnet().to(device)

        classifier = Network(output_size=NUM_CLASSES,
                             softmax_sigmoid="softmax").to(device)

        cifar_dl = Chaoyang_K_Fold_Dataloader(train_idx=train_idx, test_idx=test_idx, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed)
        train_loader, test_loader = cifar_dl.get_data_loader()

        optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

        best_val_system_loss = 100
        best_test_system_accuracy = None

        for epoch in tqdm(range(1, EPOCHS + 1)):
            train_full_automation_one_epoch(feature_extractor, classifier, train_loader, optimizer, scheduler)

            test_system_accuracy, test_system_loss, = evaluate_full_automation_one_epoch(feature_extractor, classifier, test_loader)

            if epoch == EPOCHS:
                best_test_system_accuracy = test_system_accuracy
        avg_acc += best_test_system_accuracy

        print(f'Full Automation Accuracy: {best_test_system_accuracy}\n')
    return avg_acc / K_FOLD


def train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler):
    # switch to train mode
    feature_extractor.train()
    allocation_system.train()
    for classifier in classifiers:
        classifier.train()

    for i, (batch_input, batch_targets, _) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        batch_features = feature_extractor(batch_input)
        batch_outputs_allocation_system = allocation_system(batch_features)
        batch_outputs_classifiers = torch.empty((NUM_EXPERTS + 1, len(batch_targets), NUM_CLASSES))
        for idx, classifier in enumerate(classifiers):
            batch_outputs_classifiers[idx] = classifier(batch_features)

        # compute and record loss
        batch_loss = mixture_of_ai_experts_loss(allocation_system_output=batch_outputs_allocation_system,
                                                classifiers_outputs=batch_outputs_classifiers, targets=batch_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if USE_LR_SCHEDULER:
            scheduler.step()


def evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, data_loader):
    feature_extractor.eval()
    allocation_system.eval()
    for classifier in classifiers:
        classifier.eval()

    classifiers_outputs = torch.tensor([]).to(device)
    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).long().to(device)

    with torch.no_grad():
        for i, (batch_input, batch_targets, _) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_allocation_system_outputs = allocation_system(batch_features)
            batch_outputs_classifiers = torch.empty((NUM_EXPERTS + 1, len(batch_targets), NUM_CLASSES)).to(device)
            for idx, classifier in enumerate(classifiers):
                batch_outputs_classifiers[idx] = classifier(batch_features)

            classifiers_outputs = torch.cat((classifiers_outputs, batch_outputs_classifiers), dim=1)
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets.float()))

    moae_loss = mixture_of_ai_experts_loss(allocation_system_output=allocation_system_outputs,
                                           classifiers_outputs=classifiers_outputs, targets=targets.long())

    classifiers_outputs = classifiers_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifiers_preds = np.argmax(classifiers_outputs, 2).T
    team_preds = classifiers_preds[range(len(classifiers_preds)), allocation_system_decisions.astype(int)]
    moae_accuracy = get_accuracy(team_preds, targets)

    return moae_accuracy, moae_loss


def run_moae(seed):
    print(f'Training Mixture of artificial experts baseline')
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    avg_acc = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_info)):
        feature_extractor = Resnet().to(device)

        allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                    softmax_sigmoid="softmax").to(device)

        classifiers = []
        for _ in range(NUM_EXPERTS + 1):
            classifier = Network(output_size=NUM_CLASSES,
                                 softmax_sigmoid="softmax").to(device)
            classifiers.append(classifier)

        cifar_dl = Chaoyang_K_Fold_Dataloader(train_idx=train_idx, test_idx=test_idx, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed)
        train_loader, test_loader = cifar_dl.get_data_loader()

        parameters = list(allocation_system.parameters())
        for classifier in classifiers:
            parameters += list(classifier.parameters())

        optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

        best_val_system_loss = 100
        best_test_system_accuracy = None

        for epoch in range(1, EPOCHS + 1):
            print("-" * 20, f'Epoch {epoch}', "-" * 20)

            train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler)

            test_moae_accuracy, test_moae_loss = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, test_loader)

            if epoch == EPOCHS:
                best_test_system_accuracy = test_moae_accuracy

        avg_acc += best_test_system_accuracy
        print(f'Mixture of Artificial Experts Accuracy: {best_test_system_accuracy}\n')
    return avg_acc / K_FOLD


def train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler):
    # switch to train mode
    feature_extractor.train()
    allocation_system.train()

    for i, (batch_input, batch_labels, batch_expert_labels) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_labels = batch_labels.to(device)

        expert_batch_preds = batch_expert_labels.permute(1, 0)  # np.empty((NUM_EXPERTS, len(batch_labels)))

        batch_features = feature_extractor(batch_input)
        batch_outputs_allocation_system = allocation_system(batch_features)

        # compute and record loss
        batch_loss = mixture_of_human_experts_loss(allocation_system_output=batch_outputs_allocation_system,
                                                   human_expert_preds=expert_batch_preds, targets=batch_labels)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if USE_LR_SCHEDULER:
            scheduler.step()


def evaluate_mohe_one_epoch(feature_extractor, allocation_system, data_loader):
    feature_extractor.eval()
    allocation_system.eval()

    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    expert_preds = torch.tensor([]).to(device)
    with torch.no_grad():
        for i, (batch_input, batch_labels, batch_expert_labels) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_labels = batch_labels.to(device)
            batch_expert_labels = batch_expert_labels.to(device)

            batch_features = feature_extractor(batch_input)
            batch_allocation_system_outputs = allocation_system(batch_features)

            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_labels))
            expert_preds = torch.cat((expert_preds, batch_expert_labels))

    expert_preds = expert_preds.permute(1, 0).cpu().long()  # np.empty((NUM_EXPERTS, len(targets)))

    # compute and record loss
    mohe_loss = mixture_of_human_experts_loss(allocation_system_output=allocation_system_outputs,
                                              human_expert_preds=expert_preds, targets=targets.long())

    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    expert_preds = expert_preds.T
    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    team_preds = expert_preds[range(len(expert_preds)), allocation_system_decisions.astype(int)]
    mohe_accuracy = get_accuracy(team_preds, targets)

    return mohe_accuracy, mohe_loss


def run_mohe(seed):
    print(f'Training Mixture of human experts baseline')
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    avg_acc = 0
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_info)):

        feature_extractor = Resnet().to(device)

        allocation_system = Network(output_size=NUM_EXPERTS,
                                    softmax_sigmoid="softmax").to(device)

        cifar_dl = Chaoyang_K_Fold_Dataloader(train_idx=train_idx, test_idx=test_idx, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed)
        train_loader, test_loader = cifar_dl.get_data_loader()

        parameters = allocation_system.parameters()
        optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

        best_val_system_loss = 100
        best_test_system_accuracy = None

        for epoch in range(1, EPOCHS + 1):
            print("-" * 20, f'Epoch {epoch}', "-" * 20)

            train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler)
            test_mohe_accuracy, test_mohe_loss = evaluate_mohe_one_epoch(feature_extractor, allocation_system, test_loader)

            if epoch == EPOCHS:
                best_test_system_accuracy = test_mohe_accuracy
        avg_acc += best_test_system_accuracy
        print(f'Mixture of Human Experts Accuracy: {best_test_system_accuracy}\n')
    return avg_acc / K_FOLD


if __name__ == '__main__':

    NUM_EXPERTS = 2
    best_expert_accuracies = []
    avg_expert_accuracies = []
    our_approach_accuracies = []
    our_approach_coverages = []
    jsf_accuracies = []
    jsf_coverages = []
    mohe_accuracies = []
    full_automation_accuracies = []
    moae_accuracies = []

    # 读取 chaoyang 数据集JSON 文件
    with open('../data/chaoyang/train_label.json', 'r') as f:
        data_info = json.load(f)

    for seed in [42, 233, 666, 3407]:
        print(f'Seed: {seed}')
        setup_seed(seed=seed)

        best_expert_accuracy = get_accuracy_of_best_expert(seed)
        best_expert_accuracies.append(best_expert_accuracy)

        avg_expert_accuracy = get_accuracy_of_average_expert(seed)
        avg_expert_accuracies.append(avg_expert_accuracy)

        our_approach_accuracy, our_approach_coverage = run_team_performance_optimization("Our Approach", data_info, seed)
        our_approach_accuracies.append(our_approach_accuracy)

        jsf_accuracy, jsf_coverage = run_team_performance_optimization("Joint Sparse Framework", data_info, seed)
        jsf_accuracies.append(jsf_accuracy)

        mohe_accuracy = run_mohe(seed)
        mohe_accuracies.append(mohe_accuracy)

        full_automation_accuracy = run_full_automation(seed)
        full_automation_accuracies.append(full_automation_accuracy)

        moae_accuracy = run_moae(seed)
        moae_accuracies.append(moae_accuracy)

    mean_best_expert_accuracy, mean_best_expert_accuracy_std = np.mean(best_expert_accuracies), np.std(best_expert_accuracies)
    mean_avg_expert_accuracy, mean_avg_expert_accuracy_std = np.mean(avg_expert_accuracies), np.std(avg_expert_accuracies)
    mean_jsf_accuracy, mean_jsf_accuracy_std = np.mean(jsf_accuracies), np.std(jsf_accuracies)
    mean_our_approach_accuracy, mean_our_approach_accuracy_std = np.mean(our_approach_accuracies), np.std(our_approach_accuracies)
    mean_full_automation_accuracy, mean_full_automation_accuracy_std = np.mean(full_automation_accuracies), np.std(full_automation_accuracies)
    mean_moae_accuracy, mean_moae_accuracy_std = np.mean(moae_accuracies), np.std(moae_accuracies)
    mean_mohe_accuracy, mean_mohe_accuracy_std = np.mean(mohe_accuracies), np.std(mohe_accuracies)

    print(f"Best expert: {mean_best_expert_accuracy:.8f} {mean_best_expert_accuracy_std:.8f}\n")
    print(f"Avg expert: {mean_avg_expert_accuracy:.8f} {mean_avg_expert_accuracy_std:.8f}\n")
    print(f"JSF: {mean_jsf_accuracy:.8f} {mean_jsf_accuracy_std:.8f}\n")
    print(f"Full automation: {mean_full_automation_accuracy:.8f} {mean_full_automation_accuracy_std:.8f}\n")
    print(f"MoAE: {mean_moae_accuracy:.8f} {mean_moae_accuracy_std:.8f}\n")
    print(f"MoHE: {mean_mohe_accuracy:.8f} {mean_mohe_accuracy_std:.8f}\n")
    print(f"HAIT: {mean_our_approach_accuracy:.8f} {mean_our_approach_accuracy_std:.8f}\n")
    print(f"-------------------------------------------------------------------------------\n")
    file_path = "./log/Chaoyang-exp/exp-chaoyang.txt"

    with open(file_path, 'a') as file:
        file.write(f"Best expert: {mean_best_expert_accuracy:.8f} {mean_best_expert_accuracy_std:.8f}\n")
        file.write(f"Avg expert: {mean_avg_expert_accuracy:.8f} {mean_avg_expert_accuracy_std:.8f}\n")
        file.write(f"JSF: {mean_jsf_accuracy:.8f} {mean_jsf_accuracy_std:.8f}\n")
        file.write(f"Full automation: {mean_full_automation_accuracy:.8f} {mean_full_automation_accuracy_std:.8f}\n")
        file.write(f"MoAE: {mean_moae_accuracy:.8f} {mean_moae_accuracy_std:.8f}\n")
        file.write(f"MoHE: {mean_mohe_accuracy:.8f} {mean_mohe_accuracy_std:.8f}\n")
        file.write(f"HAIT: {mean_our_approach_accuracy:.8f} {mean_our_approach_accuracy_std:.8f}\n")
        file.write(f"-------------------------------------------------------------------------------\n")

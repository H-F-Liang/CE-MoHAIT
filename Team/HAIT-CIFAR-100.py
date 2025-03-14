import random
import math
from sympy import im
from tqdm import tqdm
import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from PIL import Image
from utils.TrainingLog import save_log

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 100
DROPOUT = 0.00
NUM_HIDDEN_UNITS = 100
LR = 5e-3
USE_LR_SCHEDULER = True
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
EPOCHS = 50

LOG = './log/CIFAR-100-exp/'
CLS_LOG = ''
CLS_TEAM_LOG = ''
EXPERT_TEAM_LOG = ''
JSF_LOG = ''
HAIT_LOG = ''


class CIFAR100_Dataset(torchvision.datasets.CIFAR100):
    def __getitem__(self, index: int):
        img, target, fine_target = self.data[index], self.targets[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # 将 target_transform 移除，直接使用 target作为fine_target
        return img, target, fine_target  # 这里返回 target 作为目标


class CIFAR100_3_Split_Dataloader:
    def __init__(self, train_batch_size=128, test_batch_size=128, seed=42, small_version=False):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
        self.small_version = small_version

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761])])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761])])

        np.random.seed(self.seed)
        train_val_set = CIFAR100_Dataset(root='../data', train=True, download=True, transform=transform_train)
        all_indices = np.arange(0, 50000, 1)
        train_indices = np.random.choice(all_indices, 40000, replace=False)
        val_indices = np.setdiff1d(all_indices, train_indices)
        self.trainset = torch.utils.data.Subset(train_val_set, train_indices)
        self.valset = torch.utils.data.Subset(train_val_set, val_indices)

        self.testset = CIFAR100_Dataset(root='../data', train=False, download=True, transform=transform_test)

        if self.small_version:
            np.random.seed(self.seed)
            train_indices = np.random.choice(np.arange(0, 40000, 1), 4000, replace=False)
            val_indices = np.random.choice(np.arange(0, 10000, 1), 1000, replace=False)
            test_indices = np.random.choice(np.arange(0, 10000, 1), 1000, replace=False)

            self.trainset = torch.utils.data.Subset(self.trainset, train_indices)
            self.valset = torch.utils.data.Subset(self.valset, val_indices)
            self.testset = torch.utils.data.Subset(self.testset, test_indices)

    def get_data_loader(self):
        train_loader = self._get_data_loader(dataset=self.trainset, batch_size=self.train_batch_size, drop_last=True)
        val_loader = self._get_data_loader(dataset=self.valset, batch_size=self.test_batch_size, drop_last=False)
        test_loader = self._get_data_loader(dataset=self.testset, batch_size=self.test_batch_size, drop_last=False)
        return train_loader, val_loader, test_loader

    def _get_data_loader(self, dataset, batch_size, drop_last, shuffle=True):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=drop_last, pin_memory=True)


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
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[index]).to(classifier_output.device)
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
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[human_expert_preds[idx].astype(int)]).to(allocation_system_output.device)
        team_probs = team_probs + allocation_system_output[:, idx].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    mohe_loss = nn.NLLLoss()(log_output, targets)

    return mohe_loss


class Resnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
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


class Cifar100Expert:
    """A class used to represent an Expert on CIFAR100 data.

    Parameters
    ----------
    strengths : list[int]
        list of subclass indices defining the experts strengths
    weaknesses : list[int]
        list of subclass indices defining the experts weaknesses

    Attributes
    ----------
    strengths : list[int]
        list of subclass indices defining the experts strengths. If the subclass index of an image is in strength the expert makes a correct prediction, if not expert predicts a random superclass
    weaknesses : list[int]
        list of subclass indices defining the experts weaknesses. If the subclass index of an image is in weaknesses the expert predicts a random superclass, if not the expert makes a correct prediction
    use_strengths : bool
        a boolean indicating whether the expert is defined by its strengths or its weaknesses. True if strengths are not empty, False if strengths are empty
    subclass_idx_to_superclass_idx : dict of {int : int}
        a dictionary that maps the 100 subclass indices of CIFAR100 to their 20 superclass indices

    Methods
    -------
    predict(fine_ids)
        makes a prediction based on the specified strengths or weaknesses and the given subclass indices
    """

    def __init__(self, strengths: list = [], weaknesses: list = []):
        self.strengths = strengths
        self.weaknesses = weaknesses

        assert len(self.strengths) > 0 or len(
            self.weaknesses) > 0, "the competence of a Cifar100Expert needs to be specified using either strengths or weaknesses"

        self.use_strengths = len(self.strengths) > 0

    def predict(self, subclass_idxs: list) -> list:
        """Predicts the superclass indices for the given subclass indices

        Parameters
        ----------
        subclass_idxs : list of int
            list of subclass indices to get a prediction for. Predictions are made based on the specified strengths or weaknesses.

        Returns
        -------
        list of int
            returns a list of superclass indices that represent the experts prediction

        """
        predictions = []
        if self.use_strengths:
            for subclass_idx in subclass_idxs:
                if subclass_idx in self.strengths:
                    predictions.append(subclass_idx.item())
                else:
                    predictions.append(random.randint(0, 99))
        else:
            for subclass_idx in subclass_idxs:
                if subclass_idx in self.weaknesses:
                    predictions.append(random.randint(0, 99))
                else:
                    predictions.append(subclass_idx.item())

        return predictions


class Cifar100AverageExpert:
    """A class used to represent a cohort of Cifar100Experts.

        Parameters
        ----------
        expert_fns : list[Cifar100Expert.predict]
            list of Cifar100Expert.predict functions that return the predictions of a Cifar100Expert for given subclass_idxs

        Attributes
        ----------
        expert_fns : list[Cifar100Expert.predict]
            list of Cifar100Expert.predict functions that return the predictions of a Cifar100Expert for given subclass_idxs
        num_experts : int
            the number of experts in the cohort. Is the length of expert_fns

        Methods
        -------
        predict(subclass_idxs)
            makes a prediction for the given subclass indices
        """

    def __init__(self, expert_fns=[]):
        self.expert_fns = expert_fns
        self.num_experts = len(self.expert_fns)

    def predict(self, subclass_idxs):
        """Returns the predictions of a random Cifar100Expert for each image for the given subclass indices

        The first expert in expert_fns predicts the first image in subclass_idx.
        The second expert in expert_fns predicts the second image in subclass_idx.
        ...
        If all experts in expert_fns made their prediction for one image, the first expert starts again.
        If three experts are defined in expert_fns, the first expert predicts the 1st, 4th, 7th, 10th ... image

        Parameters
        ----------
        subclass_idxs : list of int
            list of subclass indices to get a prediction for

        Returns
        -------
        list of int
            returns a list of superclass indices that represent the experts prediction
        """
        all_experts_predictions = [expert_fn(subclass_idxs) for expert_fn in self.expert_fns]
        predictions = [None] * len(subclass_idxs)

        for idx, expert_predictions in enumerate(all_experts_predictions):
            predictions[idx::self.num_experts] = expert_predictions[idx::self.num_experts]

        return predictions


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
    print(preds.shape, allocation_system_decisions.astype(int).shape)
    print(np.max(preds), np.max(allocation_system_decisions.astype(int)))
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


def train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, expert_fns, loss_fn):
    feature_extractor.eval()
    classifier.train()
    allocation_system.train()

    for i, (batch_input, batch_targets, batch_subclass_idxs) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        expert_batch_preds = np.empty((NUM_EXPERTS, len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_subclass_idxs))

        batch_features = feature_extractor(batch_input)
        batch_outputs_classifier = classifier(batch_features)
        batch_outputs_allocation_system = allocation_system(batch_features)

        batch_loss = loss_fn(epoch=epoch, classifier_output=batch_outputs_classifier, allocation_system_output=batch_outputs_allocation_system,
                             expert_preds=expert_batch_preds, targets=batch_targets)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if USE_LR_SCHEDULER:
            scheduler.step()


def evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, data_loader, expert_fns, loss_fn):
    feature_extractor.eval()
    classifier.eval()
    allocation_system.eval()

    classifier_outputs = torch.tensor([]).to(device)
    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    subclass_idxs = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_subclass_idxs) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)
            batch_allocation_system_outputs = allocation_system(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets))
            subclass_idxs.extend(batch_subclass_idxs)

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(subclass_idxs))

    classifier_outputs = classifier_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    system_accuracy, system_loss, metrics = get_metrics(epoch, allocation_system_outputs, classifier_outputs, expert_preds, targets, loss_fn)

    return system_accuracy, system_loss, metrics


def run_team_performance_optimization(method, seed, expert_fns):
    print(f'Team Performance Optimization with {method}')

    if method == "Joint Sparse Framework":
        loss_fn = joint_sparse_framework_loss
        allocation_system_activation_function = "sigmoid"


    elif method == "Our Approach":
        loss_fn = our_loss
        allocation_system_activation_function = "softmax"

    feature_extractor = Resnet().to(device)

    classifier = Network(output_size=NUM_CLASSES,
                         softmax_sigmoid="softmax").to(device)

    allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                softmax_sigmoid=allocation_system_activation_function).to(device)

    cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed, small_version=False)
    train_loader, val_loader, test_loader = cifar_dl.get_data_loader()

    parameters = list(classifier.parameters()) + list(allocation_system.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

    best_val_system_accuracy = 0
    best_val_system_loss = 100
    best_metrics = None

    for epoch in tqdm(range(1, EPOCHS + 1)):
        train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, expert_fns, loss_fn)
        val_system_accuracy, val_system_loss, _ = evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, val_loader, expert_fns, loss_fn)
        print('####### TEST ########')
        _, _, test_metrics = evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, test_loader, expert_fns, loss_fn)

        if method == "Joint Sparse Framework":
            save_log(JSF_LOG, test_metrics["System Accuracy"])
            if val_system_accuracy > best_val_system_accuracy:
                best_val_system_accuracy = val_system_accuracy
                best_metrics = test_metrics

        elif method == "Our Approach":
            save_log(HAIT_LOG, test_metrics["System Accuracy"])
            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
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

    return best_metrics["System Accuracy"], best_metrics["Classifier Coverage"]


def get_accuracy_of_best_expert(seed, expert_fns):
    cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed, small_version=False)
    _, _, test_loader = cifar_dl.get_data_loader()

    targets = torch.tensor([]).long()
    subclass_idxs = []

    with torch.no_grad():
        for i, (_, batch_targets, batch_subclass_idxs) in enumerate(test_loader):
            targets = torch.cat((targets, batch_targets))
            subclass_idxs.extend(batch_subclass_idxs)

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(subclass_idxs))

    expert_accuracies = []
    for idx in range(NUM_EXPERTS):
        preds = expert_preds[idx]
        acc = accuracy_score(targets, preds)
        expert_accuracies.append(acc)

    print(f'Best Expert Accuracy: {max(expert_accuracies)}\n')

    return max(expert_accuracies)


def get_accuracy_of_average_expert(seed, expert_fns):
    cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed, small_version=False)
    _, _, test_loader = cifar_dl.get_data_loader()

    targets = torch.tensor([]).long()
    subclass_idxs = []

    with torch.no_grad():
        for i, (_, batch_targets, batch_subclass_idxs) in enumerate(test_loader):
            targets = torch.cat((targets, batch_targets))
            subclass_idxs.extend(batch_subclass_idxs)

    avg_expert = Cifar100AverageExpert(expert_fns)
    avg_expert_preds = avg_expert.predict(subclass_idxs)
    avg_expert_acc = accuracy_score(targets, avg_expert_preds)
    print(f'Average Expert Accuracy: {avg_expert_acc}\n')

    return avg_expert_acc


def train_full_automation_one_epoch(feature_extractor, classifier, train_loader, optimizer, scheduler):
    # switch to train mode
    feature_extractor.eval()
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

    with torch.no_grad():
        for i, (batch_input, batch_targets, _) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            targets = torch.cat((targets, batch_targets))

    log_output = torch.log(classifier_outputs + 1e-7)
    full_automation_loss = nn.NLLLoss()(log_output, targets.long())

    classifier_outputs = classifier_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    classifier_preds = np.argmax(classifier_outputs, 1)
    full_automation_accuracy = get_accuracy(classifier_preds, targets)

    return full_automation_accuracy, full_automation_loss


def run_full_automation(seed):
    print(f'Training full automation baseline')

    feature_extractor = Resnet().to(device)

    classifier = Network(output_size=NUM_CLASSES,
                         softmax_sigmoid="softmax").to(device)

    cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed, small_version=False)
    train_loader, val_loader, test_loader = cifar_dl.get_data_loader()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

    best_val_system_loss = 100
    best_test_system_accuracy = None

    for epoch in tqdm(range(1, EPOCHS + 1)):
        train_full_automation_one_epoch(feature_extractor, classifier, train_loader, optimizer, scheduler)

        val_system_accuracy, val_system_loss = evaluate_full_automation_one_epoch(feature_extractor, classifier, val_loader)
        test_system_accuracy, test_system_loss, = evaluate_full_automation_one_epoch(feature_extractor, classifier, test_loader)
        save_log(CLS_LOG, test_system_accuracy)

        if val_system_loss < best_val_system_loss:
            best_val_system_loss = val_system_loss
            best_test_system_accuracy = test_system_accuracy

    print(f'Full Automation Accuracy: {best_test_system_accuracy}\n')
    return best_test_system_accuracy


def train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler):
    # switch to train mode
    feature_extractor.eval()
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

    feature_extractor = Resnet().to(device)

    allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                softmax_sigmoid="softmax").to(device)

    classifiers = []
    for _ in range(NUM_EXPERTS + 1):
        classifier = Network(output_size=NUM_CLASSES,
                             softmax_sigmoid="softmax").to(device)
        classifiers.append(classifier)

    cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed, small_version=False)
    train_loader, val_loader, test_loader = cifar_dl.get_data_loader()

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
        val_moae_accuracy, val_moae_loss = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, val_loader)
        test_moae_accuracy, test_moae_loss = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, test_loader)
        save_log(CLS_TEAM_LOG, test_moae_accuracy)
        if val_moae_loss < best_val_system_loss:
            best_val_system_loss = val_moae_loss
            best_test_system_accuracy = test_moae_accuracy

    print(f'Mixture of Artificial Experts Accuracy: {best_test_system_accuracy}\n')
    return best_test_system_accuracy


def train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler, expert_fns):
    # switch to train mode
    feature_extractor.eval()
    allocation_system.train()

    for i, (batch_input, batch_targets, batch_subclass_idxs) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        expert_batch_preds = np.empty((NUM_EXPERTS, len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_subclass_idxs))

        batch_features = feature_extractor(batch_input)
        batch_outputs_allocation_system = allocation_system(batch_features)

        # compute and record loss
        batch_loss = mixture_of_human_experts_loss(allocation_system_output=batch_outputs_allocation_system,
                                                   human_expert_preds=expert_batch_preds, targets=batch_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if USE_LR_SCHEDULER:
            scheduler.step()


def evaluate_mohe_one_epoch(feature_extractor, allocation_system, data_loader, expert_fns):
    feature_extractor.eval()
    allocation_system.eval()

    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    subclass_idxs = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_subclass_idxs) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_allocation_system_outputs = allocation_system(batch_features)

            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets))
            subclass_idxs.extend(batch_subclass_idxs)

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(subclass_idxs))

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


def run_mohe(seed, expert_fns):
    print(f'Training Mixture of human experts baseline')

    feature_extractor = Resnet().to(device)

    allocation_system = Network(output_size=NUM_EXPERTS,
                                softmax_sigmoid="softmax").to(device)

    cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, seed=seed, small_version=False)
    train_loader, val_loader, test_loader = cifar_dl.get_data_loader()

    parameters = allocation_system.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

    best_val_system_loss = 100
    best_test_system_accuracy = None

    for epoch in range(1, EPOCHS + 1):
        print("-" * 20, f'Epoch {epoch}', "-" * 20)

        train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler, expert_fns)
        val_mohe_accuracy, val_mohe_loss = evaluate_mohe_one_epoch(feature_extractor, allocation_system, val_loader, expert_fns)
        test_mohe_accuracy, test_mohe_loss = evaluate_mohe_one_epoch(feature_extractor, allocation_system, test_loader, expert_fns)
        save_log(EXPERT_TEAM_LOG, test_mohe_accuracy)

        if val_mohe_loss < best_val_system_loss:
            best_val_system_loss = val_mohe_loss
            best_test_system_accuracy = test_mohe_accuracy

    print(f'Mixture of Human Experts Accuracy: {best_test_system_accuracy}\n')
    return best_test_system_accuracy


if __name__ == '__main__':
    NUM_EXPERTS = 12

    # ########################################################################
    # # 单一分类器只需要单独训练五次
    # full_automation_accuracies = []
    # for exp_times in [1, 2, 3, 4, 5]:
    #     seed = [42, 233, 666, 2025, 3407][exp_times - 1]
    #     print(f'Seed: {seed}')
    #     print("-"*40)
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     CLS_LOG = LOG + f'cls_{exp_times}.txt'
    #     full_automation_accuracy = run_full_automation(seed)
    #     full_automation_accuracies.append(full_automation_accuracy)

    # mean_full_automation_accuracy = np.mean(full_automation_accuracies)
    # mean_full_automation_accuracy_std = np.std(full_automation_accuracies)
    # file_path = f"./log/CIFAR-100-exp/HAIT-result-exp-CIFAR-100.txt"
    # with open(file_path, 'a') as file:
    #     file.write(f"EXPERT NUMBER: One classifier\n")
    #     file.write(f"Classifier Team acc: {mean_full_automation_accuracy:.8f} {mean_full_automation_accuracy_std:.8f}\n")
    #     file.write(f"-------------------------------------------------------------------------------\n")

    # ########################################################################
    # for idx, num_experts in enumerate([2, 4, 6, 8, 10, 12]):
    #     moae_accuracies = []
    #     for exp_times in [1, 2, 3, 4, 5]:
    #         seed = [42, 233, 666, 2025, 3407][exp_times - 1]
    #         print(f'Seed: {seed}')
    #         print("-"*40)
    #         np.random.seed(seed)
    #         random.seed(seed)
    #         CLS_TEAM_LOG = LOG + f'cls_team_{num_experts}_{exp_times}.txt'
    #         moae_accuracy = run_moae(seed)
    #         moae_accuracies.append(moae_accuracy)
    #     mean_moae_accuracy = np.mean(moae_accuracies)
    #     mean_moae_accuracy_std = np.std(moae_accuracies)

    # moae没有专家能力不同规模单独训练五次
    # file_path = f"./log/CIFAR-100-exp/HAIT-result-exp-CIFAR-100.txt"
    # with open(file_path, 'a') as file:
    #     file.write(f"EXPERT NUMBER: {num_experts} classifier\n")
    #     file.write(f"Classifier Team acc: {mean_moae_accuracy:.8f} {mean_moae_accuracy_std:.8f}\n")
    #     file.write(f"-------------------------------------------------------------------------------\n")

    # ########################################################################
    for idx, num_experts in enumerate([2, 12]):
        print(f'Number of Experts: {num_experts}')
        NUM_EXPERTS = num_experts
        for (mean, std) in [(25, 5), (50, 5), (75, 5)]:
            if num_experts == 2 and mean != 75:
                continue
            best_expert_accuracies = []
            avg_expert_accuracies = []
            our_approach_accuracies = []
            our_approach_coverages = []
            jsf_accuracies = []
            jsf_coverages = []
            mohe_accuracies = []

            for exp_times in [1, 2, 3, 4, 5]:
                seed = [42, 233, 666, 2025, 3407][exp_times - 1]
                print(f'Seed: {seed}')
                print("-" * 40)
                np.random.seed(seed)
                random.seed(seed)
                subclass_idxs = list(range(0, 100))
                expert_strengths = []
                for _ in range(num_experts):
                    num_perfect_predictions = int(np.clip(np.random.normal(mean, std), 0, 100))
                    strengths = random.sample(subclass_idxs, num_perfect_predictions)
                    expert_strengths.append(strengths)

                expert_fns = []
                for i, strengths in enumerate(expert_strengths):
                    cifar100_expert = Cifar100Expert(strengths=strengths)
                    expert_fns.append(cifar100_expert.predict)

                # best_expert_accuracy = get_accuracy_of_best_expert(seed, expert_fns)
                # best_expert_accuracies.append(best_expert_accuracy)

                # avg_expert_accuracy = get_accuracy_of_average_expert(seed, expert_fns)
                # avg_expert_accuracies.append(avg_expert_accuracy)
                EXPERT_TEAM_LOG = LOG + f'expert_team_{num_experts}_{mean}_{exp_times}.txt'
                mohe_accuracy = run_mohe(seed, expert_fns)
                mohe_accuracies.append(mohe_accuracy)
                JSF_LOG = LOG + f'JSF_{num_experts}_{mean}_{exp_times}.txt'
                jsf_accuracy, jsf_coverage = run_team_performance_optimization("Joint Sparse Framework", seed, expert_fns)
                jsf_accuracies.append(jsf_accuracy)
                HAIT_LOG = LOG + f'HAIT_{num_experts}_{mean}_{exp_times}.txt'
                our_approach_accuracy, our_approach_coverage = run_team_performance_optimization("Our Approach", seed, expert_fns)
                our_approach_accuracies.append(our_approach_accuracy)

            # mean_best_expert_accuracy = np.mean(best_expert_accuracies)
            # mean_best_expert_accuracy_std = np.std(best_expert_accuracies)

            # mean_avg_expert_accuracy = np.mean(avg_expert_accuracies)
            # mean_avg_expert_accuracy_std = np.std(avg_expert_accuracies)

            mean_mohe_accuracy = np.mean(mohe_accuracies)
            mean_mohe_accuracy_std = np.std(mohe_accuracies)

            mean_jsf_accuracy = np.mean(jsf_accuracies)
            mean_jsf_accuracy_std = np.std(jsf_accuracies)

            mean_our_approach_accuracy = np.mean(our_approach_accuracies)
            mean_our_approach_accuracy_std = np.std(our_approach_accuracies)
            # file_path = f"./log/CIFAR-100-exp/HAIT-result-exp-CIFAR-100.txt"
            # with open(file_path, 'a') as file:
            #     file.write(f"EXPERT NUMBER: {num_experts}\n")
            #     file.write(f"EXPERT ABILITY: ({mean}, {std})\n")
            #     # file.write(f"Best Expert acc: {mean_best_expert_accuracy:.8f} {mean_best_expert_accuracy_std:.8f}\n")
            #     # file.write(f"Random Expert acc: {mean_avg_expert_accuracy:.8f} {mean_avg_expert_accuracy_std:.8f}\n")
            #     file.write(f"Expert Team acc: {mean_mohe_accuracy:.8f} {mean_mohe_accuracy_std:.8f}\n")
            #     file.write(f"JSF acc: {mean_jsf_accuracy:.8f} {mean_jsf_accuracy_std:.8f}\n")
            #     file.write(f"HAIT acc: {mean_our_approach_accuracy:.8f} {mean_our_approach_accuracy_std:.8f}\n")
            #     file.write(f"-------------------------------------------------------------------------------\n")

    # table_list = []

    # mean_full_automation_accuracy = np.mean(full_automation_accuracies)
    # mean_moae_accuracy = np.mean(moae_accuracies)
    # table_list.append(['--------', 'Full Automation', mean_full_automation_accuracy])
    # table_list.append(['--------', 'MOAE', mean_moae_accuracy])
    # table_list.append(['--------', '--------', '--------'])

    # for num_experts in range(4,11):
    #   mean_best_expert_accuracy = np.mean(best_expert_accuracies[num_experts])
    #   table_list.append([num_experts, 'Best Expert', mean_best_expert_accuracy])

    #   mean_avg_expert_accuracy = np.mean(avg_expert_accuracies[num_experts])
    #   table_list.append([num_experts, 'Random Expert', mean_avg_expert_accuracy])

    #   mean_our_approach_accuracy = np.mean(our_approach_accuracies[num_experts])
    #   table_list.append([num_experts, 'Our Approach', mean_our_approach_accuracy])

    #   mean_jsf_accuracy = np.mean(jsf_accuracies[num_experts])
    #   table_list.append([num_experts, 'JSF', mean_jsf_accuracy])

    #   mean_mohe_accuracy = np.mean(mohe_accuracies[num_experts])
    #   table_list.append([num_experts, 'MOHE', mean_mohe_accuracy])

    #   table_list.append(['--------', '--------', '--------'])

    # print(tabulate(table_list, headers=['Number of Experts', 'Method', 'Accuracy']))

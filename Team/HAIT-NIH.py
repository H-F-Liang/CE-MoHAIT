import os
import random
from itertools import chain
from typing import Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_to_test_Labels = "../data/NIH/four_findings_expert_labels_test_labels.csv"
path_to_val_Labels = "../data/NIH/four_findings_expert_labels_validation_labels.csv"
test_labels = pd.read_csv(path_to_test_Labels)
val_labels = pd.read_csv(path_to_val_Labels)

ground_truth_labels = pd.concat([test_labels, val_labels])
ground_truth_labels["Fracture_Label"] = ground_truth_labels["Fracture"].map(dict(YES=1, NO=0))
ground_truth_labels["Pneumothorax_Label"] = ground_truth_labels["Pneumothorax"].map(dict(YES=1, NO=0))
ground_truth_labels["Airspace_Opacity_Label"] = ground_truth_labels["Airspace opacity"].map(dict(YES=1, NO=0))
ground_truth_labels["Nodule_Or_Mass_Label"] = ground_truth_labels["Nodule or mass"].map(dict(YES=1, NO=0))

path_to_individual_reader = "../data/NIH/four_findings_expert_labels_individual_readers.csv"
individual_readers = pd.read_csv(path_to_individual_reader)

individual_readers["Fracture_Expert_Label"] = individual_readers["Fracture"].map(dict(YES=1, NO=0))
individual_readers["Pneumothorax_Expert_Label"] = individual_readers["Pneumothorax"].map(dict(YES=1, NO=0))
individual_readers["Airspace_Opacity_Expert_Label"] = individual_readers["Airspace opacity"].map(dict(YES=1, NO=0))
individual_readers["Nodule_Or_Mass_Expert_Label"] = individual_readers["Nodule/mass"].map(dict(YES=1, NO=0))

individual_readers["Fracture_GT_Label"] = individual_readers["Image ID"].map(pd.Series(ground_truth_labels["Fracture_Label"].values, index=ground_truth_labels["Image Index"]).to_dict())
individual_readers["Pneumothorax_GT_Label"] = individual_readers["Image ID"].map(pd.Series(ground_truth_labels["Pneumothorax_Label"].values, index=ground_truth_labels["Image Index"]).to_dict())
individual_readers["Airspace_Opacity_GT_Label"] = individual_readers["Image ID"].map(
    pd.Series(ground_truth_labels["Airspace_Opacity_Label"].values, index=ground_truth_labels["Image Index"]).to_dict())
individual_readers["Nodule_Or_Mass_GT_Label"] = individual_readers["Image ID"].map(pd.Series(ground_truth_labels["Nodule_Or_Mass_Label"].values, index=ground_truth_labels["Image Index"]).to_dict())

individual_readers["Fracture_Correct"] = (individual_readers['Fracture_Expert_Label'] == individual_readers['Fracture_GT_Label']).astype(int)
individual_readers["Pneumothorax_Correct"] = (individual_readers['Pneumothorax_Expert_Label'] == individual_readers['Pneumothorax_GT_Label']).astype(int)
individual_readers["Airspace_Opacity_Correct"] = (individual_readers['Airspace_Opacity_Expert_Label'] == individual_readers['Airspace_Opacity_GT_Label']).astype(int)
individual_readers["Nodule_Or_Mass_Correct"] = (individual_readers['Nodule_Or_Mass_Expert_Label'] == individual_readers['Nodule_Or_Mass_GT_Label']).astype(int)

individual_readers.to_csv("../data/NIH/labels.csv")

NUM_CLASSES = 2
DROPOUT = 0.00
NUM_HIDDEN_UNITS = 30
LR = 5e-4
USE_LR_SCHEDULER = False
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 10


class NIH_Dataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.image_ids = data["Image ID"].values
        self.targets = data["GT_Label"].values

        self.tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.images = []
        for filename in self.image_ids:
            img = Image.open("../data/NIH/" + filename)
            img = img.convert("RGB")
            img = img.resize((224, 224))
            img = self.tfms(img)
            img = img.to(device)
            self.images.append(img)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        filename, target = self.image_ids[index], self.targets[index]
        img = self.images[index]
        return img, target, filename

    def __len__(self) -> int:
        return len(self.images)


class NIH_K_Fold_Dataloader:
    def __init__(self, k=10, labelerIds=[4323195249, 4295194124], target="Airspace_Opacity", train_batch_size=8, test_batch_size=8,
                 seed=42):
        self.k = k
        self.labelerIds = labelerIds
        self.target = target
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
        self.k_fold_datasets = []
        self.k_fold_patient_ids = []

        individual_labels = pd.read_csv("../data/NIH/labels.csv")

        # get image ids for those aimages labelled by both radiologists
        common_image_ids = individual_labels["Image ID"].values.tolist()
        for labelerId in self.labelerIds:
            expert_labels = individual_labels[individual_labels["Reader ID"] == labelerId]
            expert_image_ids = expert_labels["Image ID"].values.tolist()
            common_image_ids = np.intersect1d(common_image_ids, expert_image_ids)

        # filter labels by common images ids
        self.expert_labels = individual_labels[individual_labels["Image ID"].isin(common_image_ids)][
            ["Reader ID", "Image ID", self.target + "_Expert_Label", self.target + "_GT_Label", "Patient ID"]]

        self.expert_labels.columns = ["Reader ID", "Image ID", "Expert_Label", "GT_Label", "Patient ID"]

        # transform data for stratification. Calculate the performance of each radiologists for each patient
        self.expert_labels["Expert_Correct"] = self.expert_labels["Expert_Label"] == self.expert_labels["GT_Label"]

        patient_ids = self.expert_labels["Patient ID"].unique()
        num_patient_images = self.expert_labels.drop_duplicates(subset=["Image ID"]).groupby(by="Patient ID", as_index=False).count()["Image ID"]
        self.patient_performance = pd.DataFrame({"Patient ID": patient_ids, "Num Patient Images": num_patient_images})

        for labeler_id in self.labelerIds:
            sum = self.expert_labels[self.expert_labels["Reader ID"] == labeler_id][["Patient ID", "Expert_Correct"]].groupby(by="Patient ID",
                                                                                                                              as_index=False).sum()
            sum.columns = ["Patient ID", f'{labeler_id}_num_correct']
            self.patient_performance = pd.merge(self.patient_performance, sum, left_on="Patient ID", right_on="Patient ID")
            self.patient_performance[f'{labeler_id}_perf'] = self.patient_performance[f'{labeler_id}_num_correct'] / self.patient_performance['Num Patient Images']

        # create target variable used for stratification. Target variable is the combination of radiologist#1 performance and radiologist#2 performance
        self.patient_performance["target"] = self.patient_performance[f'{self.labelerIds[0]}_perf'].astype(str) + "_" + self.patient_performance[
            f'{self.labelerIds[1]}_perf'].astype(str)

        self._init_k_folds()

    def _init_k_folds(self):
        self.labels = self.expert_labels.drop_duplicates(subset=["Image ID"])
        self.labels = self.labels.fillna(0)
        self.labels = self.labels[["Patient ID", "Image ID", "GT_Label"]]

        kf_cv = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)

        fold_data_idxs = [fold_test_idxs for (_, fold_test_idxs) in kf_cv.split(self.patient_performance["Patient ID"].values, self.patient_performance["target"].values)]

        for fold_idx in range(len(fold_data_idxs)):
            test_fold_idx = fold_idx
            test_fold_data_idxs = fold_data_idxs[test_fold_idx]

            # use next 2 folds for validation set
            val_folds_idxs = [(test_fold_idx + 1 + i) % 10 for i in range(2)]
            val_fold_data_idxs = [fold_data_idxs[val_fold_idx] for val_fold_idx in val_folds_idxs]
            val_fold_data_idxs = list(chain.from_iterable(val_fold_data_idxs))

            # use next 7 folds for training set
            train_folds_idxs = [(test_fold_idx + 3 + i) % 10 for i in range(7)]
            train_folds_data_idxs = [fold_data_idxs[train_fold_idx] for train_fold_idx in train_folds_idxs]
            train_folds_data_idxs = list(chain.from_iterable(train_folds_data_idxs))

            train_patient_ids = self.patient_performance["Patient ID"].iloc[train_folds_data_idxs]
            val_patient_ids = self.patient_performance["Patient ID"].iloc[val_fold_data_idxs]
            test_patient_ids = self.patient_performance["Patient ID"].iloc[test_fold_data_idxs]

            expert_train = self.labels[self.labels["Patient ID"].isin(train_patient_ids)]
            expert_val = self.labels[self.labels["Patient ID"].isin(val_patient_ids)]
            expert_test = self.labels[self.labels["Patient ID"].isin(test_patient_ids)]

            # check that patients are not shared across training, validation and test split
            overlap = expert_train[expert_train["Patient ID"].isin(expert_val["Patient ID"])]
            assert len(overlap) == 0, "Train and Val Patient Ids overlap"

            overlap = expert_train[expert_train["Patient ID"].isin(expert_test["Patient ID"])]
            assert len(overlap) == 0, "Train and Test Patient Ids overlap"

            overlap = expert_val[expert_val["Patient ID"].isin(expert_test["Patient ID"])]
            assert len(overlap) == 0, "Val and Test Patient Ids overlap"

            expert_train = expert_train[["Image ID", "GT_Label"]]
            expert_val = expert_val[["Image ID", "GT_Label"]]
            expert_test = expert_test[["Image ID", "GT_Label"]]
            self.k_fold_datasets.append((expert_train, expert_val, expert_test))

    def get_data_loader_for_fold(self, fold_idx):
        expert_train, expert_val, expert_test = self.k_fold_datasets[fold_idx]

        expert_train_dataset = NIH_Dataset(expert_train)
        expert_val_dataset = NIH_Dataset(expert_val)
        expert_test_dataset = NIH_Dataset(expert_test)

        train_loader = torch.utils.data.DataLoader(dataset=expert_train_dataset, batch_size=self.train_batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset=expert_val_dataset, batch_size=self.test_batch_size, shuffle=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=expert_test_dataset, batch_size=self.test_batch_size, shuffle=True, drop_last=False)
        return train_loader, val_loader, test_loader


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
    #   epoch: int = current epoch (not used, just to have the same function parameters as with JSF loss)
    #   classifier_output: softmax probabilities as class probabilities,  nxm matrix with n=batch size, m=number of classes
    #   allocation_system_output: softmax outputs as weights,  nx(m+1) matrix with n=batch size, m=number of experts + 1 for machine
    #   expert_preds: nxm matrix with expert predictions with n=number of experts, m=number of classes
    #   targets: targets as 1-dim vector with n length with n=batch_size

    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, NUM_CLASSES)).to(classifier_output.device)  # set up zero-initialized tensor to store team predictions
    team_probs = team_probs + allocation_system_output[:, 0].reshape(-1, 1) * classifier_output  # add the weighted classifier prediction to the team prediction
    for idx in range(NUM_EXPERTS):  # continue with human experts
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[expert_preds[idx].astype(int)]).to(classifier_output.device)
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

        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        del self.resnet.fc

        if "resnet.pth" in os.listdir():
            print('load Resnet-18 checkpoint resnet.pth')
            print(self.resnet.load_state_dict(
                torch.load("resnet.pth"),
                strict=False))
        else:
            print('load Resnet-18 pretrained on ImageNet')

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


class NihExpert:
    """A class used to represent an Expert on NIH ChestX-ray data.

    Parameters
    ----------
    labeler_id : int
        the Reader ID to specify which radiologist the expert object represents
    target : str
        the target to make predictions for

    Attributes
    ----------
    labeler_id : int
        the Reader ID to specify which radiologist the expert object represents
    target : str
        the target to make predictions for
    image_id_to_prediction : dict of {int : str}
        a dictionary that maps the image id to the prediction the radiologist made for the specified target

    Methods
    -------
    predict(image_ids)
        makes a prediction for the given image ids
    """

    def __init__(self, labeler_id: int, target: str):
        self.labelerId = labeler_id
        self.target = target

        individual_labels = pd.read_csv("../data/NIH/labels.csv")

        expert_labels = individual_labels[individual_labels["Reader ID"] == self.labelerId][
            ["Image ID", self.target + "_Expert_Label", self.target + "_GT_Label"]]
        expert_labels = expert_labels.fillna(0)

        self.image_id_to_prediction = pd.Series(expert_labels[self.target + "_Expert_Label"].values,
                                                index=expert_labels["Image ID"]).to_dict()

    def predict(self, image_ids):
        """Returns the experts predictions for the given image ids. Works only for image ids that are labeled by the expert

        Parameters
        ----------
        image_ids : list of int
            the image ids to get the radiologists predictions for

        Returns
        -------
        list of int
            returns a list of 0 or 1 that represent the radiologists prediction for the specified target
        """
        return [self.image_id_to_prediction[image_id] for image_id in image_ids]

    def predict_unlabeled_data(self, image_ids):
        """Returns the experts predictions for the given image ids. Works for all image ids, returns -1 if not labeled by expert

        Parameters
        ----------
        image_ids : list of int
            the image ids to get the radiologists predictions for

        Returns
        -------
        list of int
            returns a list of 0 or 1 that represent the radiologists prediction for the specified target, or -1 if no prediction
        """
        return [self.image_id_to_prediction[image_id] if image_id in self.image_id_to_prediction else -1 for image_id in image_ids]


class NihAverageExpert:
    def __init__(self, expert_fns=[]):
        self.expert_fns = expert_fns
        self.num_experts = len(self.expert_fns)

    def predict(self, filenames):
        all_experts_predictions = [expert_fn(filenames) for expert_fn in self.expert_fns]
        predictions = [None] * len(filenames)

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

    for i, (batch_input, batch_targets, batch_filenames) in enumerate(train_loader):
        batch_targets = batch_targets.to(device)

        expert_batch_preds = np.empty((NUM_EXPERTS, len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_filenames))

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
    filenames = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_filenames) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)
            batch_allocation_system_outputs = allocation_system(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets))
            filenames.extend(batch_filenames)

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(filenames))

    classifier_outputs = classifier_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifier_preds = np.argmax(classifier_outputs, 1)
    preds = np.vstack((classifier_preds, expert_preds)).T
    system_preds = preds[range(len(preds)), allocation_system_decisions.astype(int)]

    system_accuracy, system_loss, metrics = get_metrics(epoch, allocation_system_outputs, classifier_outputs, expert_preds, targets, loss_fn)

    return system_accuracy, system_loss, system_preds, allocation_system_decisions, targets


def run_team_performance_optimization(method, seed, expert_fns):
    print(f'Team Performance Optimization with {method}')

    if method == "Joint Sparse Framework":
        loss_fn = joint_sparse_framework_loss
        allocation_system_activation_function = "sigmoid"


    elif method == "Our Approach":
        loss_fn = our_loss
        allocation_system_activation_function = "softmax"

    feature_extractor = Resnet().to(device)

    nih_dataloader = NIH_K_Fold_Dataloader(
        K,
        LABELER_IDS,
        TARGET,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        seed
    )

    overall_allocation_system_decisions = []
    overall_system_preds = []
    overall_targets = []

    for fold_idx in range(K):
        print(f'Running fold {fold_idx + 1} out of {K}')

        classifier = Network(output_size=NUM_CLASSES,
                             softmax_sigmoid="softmax").to(device)

        allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                    softmax_sigmoid=allocation_system_activation_function).to(device)

        train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        parameters = list(classifier.parameters()) + list(allocation_system.parameters())
        optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

        best_val_system_accuracy = 0
        best_val_system_loss = 100
        best_metrics = None

        for epoch in tqdm(range(1, EPOCHS + 1)):
            train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler, expert_fns, loss_fn)

            val_system_accuracy, val_system_loss, _, _, _ = evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, val_loader, expert_fns, loss_fn)
            _, _, test_system_preds, test_allocation_system_decisions, test_targets = evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, test_loader, expert_fns, loss_fn)

            if method == "Joint Sparse Framework":
                if val_system_accuracy > best_val_system_accuracy:
                    best_val_system_accuracy = val_system_accuracy
                    best_epoch_system_preds = test_system_preds
                    best_epoch_allocation_system_decisions = test_allocation_system_decisions
                    best_epoch_targets = test_targets

            elif method == "Our Approach":
                if val_system_loss < best_val_system_loss:
                    best_val_system_loss = val_system_loss
                    best_epoch_system_preds = test_system_preds
                    best_epoch_allocation_system_decisions = test_allocation_system_decisions
                    best_epoch_targets = test_targets

        overall_system_preds.extend(list(best_epoch_system_preds))
        overall_allocation_system_decisions.extend(list(best_epoch_allocation_system_decisions))
        overall_targets.extend(list(best_epoch_targets))

    system_accuracy = get_accuracy(overall_system_preds, overall_targets)
    classifier_coverage = np.sum([1 for dec in overall_allocation_system_decisions if dec == 0])

    return system_accuracy, classifier_coverage


def get_accuracy_of_best_expert(seed, expert_fns):
    nih_dataloader = NIH_K_Fold_Dataloader(
        K,
        LABELER_IDS,
        TARGET,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        seed
    )

    targets = []
    filenames = []

    for fold_idx in range(K):
        print(f'Running fold {fold_idx + 1} out of {K}')
        _, _, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        with torch.no_grad():
            for i, (_, batch_targets, batch_filenames) in enumerate(test_loader):
                targets.extend(list(batch_targets.numpy()))
                filenames.extend(batch_filenames)

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(filenames))

    expert_accuracies = []
    for idx in range(NUM_EXPERTS):
        preds = expert_preds[idx]
        acc = accuracy_score(targets, preds)
        expert_accuracies.append(acc)

    print(f'Best Expert Accuracy: {max(expert_accuracies)}\n')

    return max(expert_accuracies)


def get_accuracy_of_average_expert(seed, expert_fns):
    nih_dataloader = NIH_K_Fold_Dataloader(
        K,
        LABELER_IDS,
        TARGET,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        seed
    )

    targets = []
    filenames = []

    for fold_idx in range(K):
        print(f'Running fold {fold_idx + 1} out of {K}')
        _, _, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        with torch.no_grad():
            for i, (_, batch_targets, batch_filenames) in enumerate(test_loader):
                targets.extend(list(batch_targets.numpy()))
                filenames.extend(batch_filenames)

    avg_expert = NihAverageExpert(expert_fns)
    avg_expert_preds = avg_expert.predict(filenames)
    avg_expert_acc = accuracy_score(targets, avg_expert_preds)
    print(f'Average Expert Accuracy: {avg_expert_acc}\n')

    return avg_expert_acc


def train_full_automation_one_epoch(epoch, feature_extractor, classifier, train_loader, optimizer, scheduler):
    # switch to train mode
    feature_extractor.eval()
    classifier.train()

    for i, (batch_input, batch_targets, _) in enumerate(train_loader):
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


def evaluate_full_automation_one_epoch(epoch, feature_extractor, classifier, data_loader):
    feature_extractor.eval()
    classifier.eval()

    classifier_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)

    with torch.no_grad():
        for i, (batch_input, batch_targets, _) in enumerate(data_loader):
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

    return full_automation_loss, classifier_preds, targets


def run_full_automation(seed):
    print(f'Training full automation baseline')

    feature_extractor = Resnet().to(device)

    nih_dataloader = NIH_K_Fold_Dataloader(
        K,
        LABELER_IDS,
        TARGET,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        seed
    )

    overall_classifier_preds = []
    overall_targets = []

    for fold_idx in range(K):
        print(f'Running fold {fold_idx + 1} out of {K}')

        classifier = Network(output_size=NUM_CLASSES,
                             softmax_sigmoid="softmax").to(device)

        train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        parameters = list(classifier.parameters())
        optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

        best_val_system_loss = 100

        for epoch in tqdm(range(1, EPOCHS + 1)):
            train_full_automation_one_epoch(epoch, feature_extractor, classifier, train_loader, optimizer, scheduler)

            val_system_loss, _, _ = evaluate_full_automation_one_epoch(epoch, feature_extractor, classifier, val_loader)
            _, test_classifier_preds, test_targets = evaluate_full_automation_one_epoch(epoch, feature_extractor, classifier, test_loader)

            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
                best_epoch_classifier_preds = test_classifier_preds
                best_epoch_targets = test_targets

        overall_classifier_preds.extend(list(best_epoch_classifier_preds))
        overall_targets.extend(list(best_epoch_targets))

    classifier_accuracy = get_accuracy(overall_classifier_preds, overall_targets)

    return classifier_accuracy


def train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler):
    # switch to train mode
    feature_extractor.eval()
    allocation_system.train()
    for classifier in classifiers:
        classifier.train()

    for i, (batch_input, batch_targets, batch_filenames) in enumerate(train_loader):
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
    filenames = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_filenames) in enumerate(data_loader):
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
            filenames.extend(batch_filenames)

    moae_loss = mixture_of_ai_experts_loss(allocation_system_output=allocation_system_outputs,
                                           classifiers_outputs=classifiers_outputs, targets=targets.long())

    classifiers_outputs = classifiers_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifiers_preds = np.argmax(classifiers_outputs, 2).T
    team_preds = classifiers_preds[range(len(classifiers_preds)), allocation_system_decisions.astype(int)]

    return moae_loss, team_preds, targets


def run_moae(seed):
    print(f'Training Mixture of artificial experts baseline')

    feature_extractor = Resnet().to(device)

    nih_dataloader = NIH_K_Fold_Dataloader(
        K,
        LABELER_IDS,
        TARGET,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        seed
    )

    overall_system_preds = []
    overall_targets = []

    for fold_idx in range(K):
        print(f'Running fold {fold_idx + 1} out of {K}')

        allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                    softmax_sigmoid="softmax").to(device)

        classifiers = []
        for _ in range(NUM_EXPERTS + 1):
            classifier = Network(output_size=NUM_CLASSES,
                                 softmax_sigmoid="softmax").to(device)
            classifiers.append(classifier)

        train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        parameters = list(allocation_system.parameters())
        for classifier in classifiers:
            parameters += list(classifier.parameters())

        optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

        best_val_system_loss = 100

        for epoch in tqdm(range(1, EPOCHS + 1)):
            train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler)

            val_system_loss, _, _ = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, val_loader)
            _, test_system_preds, test_targets = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, test_loader)

            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
                best_epoch_system_preds = test_system_preds
                best_epoch_targets = test_targets

        overall_system_preds.extend(list(best_epoch_system_preds))
        overall_targets.extend(list(best_epoch_targets))

    system_accuracy = get_accuracy(overall_system_preds, overall_targets)

    print(f'Mixture of Artificial Experts Accuracy: {system_accuracy}\n')
    return system_accuracy


def train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler, expert_fns):
    # switch to train mode
    feature_extractor.eval()
    allocation_system.train()

    for i, (batch_input, batch_targets, batch_filenames) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        expert_batch_preds = np.empty((NUM_EXPERTS, len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_filenames))

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
    targets = torch.tensor([]).long().to(device)
    filenames = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_filenames) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_allocation_system_outputs = allocation_system(batch_features)

            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets.float()))
            filenames.extend(batch_filenames)

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(filenames))

    mohe_loss = mixture_of_human_experts_loss(allocation_system_output=allocation_system_outputs,
                                              human_expert_preds=expert_preds, targets=targets.long())

    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    expert_preds = expert_preds.T
    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    team_preds = expert_preds[range(len(expert_preds)), allocation_system_decisions.astype(int)]

    return mohe_loss, team_preds, targets


def run_mohe(seed, expert_fns):
    print(f'Training Mixture of human experts baseline')

    feature_extractor = Resnet().to(device)

    nih_dataloader = NIH_K_Fold_Dataloader(
        K,
        LABELER_IDS,
        TARGET,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        seed
    )

    overall_system_preds = []
    overall_targets = []

    for fold_idx in range(K):
        print(f'Running fold {fold_idx + 1} out of {K}')

        allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                    softmax_sigmoid="softmax").to(device)

        train_loader, val_loader, test_loader = nih_dataloader.get_data_loader_for_fold(fold_idx)

        optimizer = torch.optim.Adam(allocation_system.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

        best_val_system_loss = 100

        for epoch in tqdm(range(1, EPOCHS + 1)):
            train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler, expert_fns)

            val_system_loss, _, _ = evaluate_mohe_one_epoch(feature_extractor, allocation_system, val_loader, expert_fns)
            _, test_system_preds, test_targets = evaluate_mohe_one_epoch(feature_extractor, allocation_system, test_loader, expert_fns)

            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
                best_epoch_system_preds = test_system_preds
                best_epoch_targets = test_targets

        overall_system_preds.extend(list(best_epoch_system_preds))
        overall_targets.extend(list(best_epoch_targets))

    system_accuracy = get_accuracy(overall_system_preds, overall_targets)

    print(f'Mixture of Human Experts Accuracy: {system_accuracy}\n')
    return system_accuracy


if __name__ == '__main__':
    NUM_EXPERTS = 2
    K = 10

    # choose radiologist_pair (last three digits are the IDS used in the paper)
    # for TARGET in ["Airspace_Opacity", "Pneumothorax", "Fracture"]:
    for TARGET in ["Fracture"]:
        # for ids in [[4295342357, 4295349121], [4323195249, 4295194124], [4295342357, 4295354117], [4323195249, 4295232296]]:
        for ids in [[4295342357, 4295354117], [4323195249, 4295232296]]:
            # for ids in [[4323195249, 4295367682]]:
            LABELER_IDS = ids
            # LABELER_IDS = [4295342357, 4295349121]
            # LABELER_IDS = [4323195249, 4295194124]
            # LABELER_IDS = [4295342357, 4295354117]
            # LABELER_IDS = [4323195249, 4295232296]

            best_expert_accuracies = []
            avg_expert_accuracies = []
            our_approach_accuracies = []
            our_approach_coverages = []
            jsf_accuracies = []
            jsf_coverages = []
            full_automation_accuracies = []
            moae_accuracies = []
            mohe_accuracies = []

            for seed in [42, 233, 666, 2025, 3407]:
                print(f'Seed: {seed}')
                print("-" * 40)
                np.random.seed(seed)
                random.seed(seed)

                expert_fns = []
                for labelerId in list(LABELER_IDS):
                    nih_expert = NihExpert(labeler_id=labelerId, target=TARGET)
                    expert_fns.append(nih_expert.predict)

                best_expert_accuracy = get_accuracy_of_best_expert(seed, expert_fns)
                best_expert_accuracies.append(best_expert_accuracy)

                avg_expert_accuracy = get_accuracy_of_average_expert(seed, expert_fns)
                avg_expert_accuracies.append(avg_expert_accuracy)

                our_approach_accuracy, our_approach_coverage = run_team_performance_optimization("Our Approach", seed, expert_fns)
                our_approach_accuracies.append(our_approach_accuracy)
                our_approach_coverages.append(our_approach_coverage)

                jsf_accuracy, jsf_coverage = run_team_performance_optimization("Joint Sparse Framework", seed, expert_fns)
                jsf_accuracies.append(jsf_accuracy)
                jsf_coverages.append(jsf_coverage)

                full_automation_accuracy = run_full_automation(seed)
                full_automation_accuracies.append(full_automation_accuracy)

                moae_accuracy = run_moae(seed)
                moae_accuracies.append(moae_accuracy)

                mohe_accuracy = run_mohe(seed, expert_fns)
                mohe_accuracies.append(mohe_accuracy)
                print("-" * 40)

            mean_best_expert_accuracy = np.mean(best_expert_accuracies)
            mean_best_expert_accuracy_std = np.std(best_expert_accuracies)
            # mean_best_expert_coverage = 0.00

            mean_avg_expert_accuracy = np.mean(avg_expert_accuracies)
            mean_avg_expert_accuracy_std = np.std(avg_expert_accuracies)
            # mean_avg_expert_coverage = 0.00

            mean_our_approach_accuracy = np.mean(our_approach_accuracies)
            mean_our_approach_accuracy_std = np.std(our_approach_accuracies)
            # mean_our_approach_coverage = np.mean(our_approach_coverages)

            mean_our_approach_coverage_std = np.std(our_approach_coverages)

            mean_jsf_accuracy = np.mean(jsf_accuracies)
            mean_jsf_accuracy_std = np.std(jsf_accuracies)
            # mean_jsf_coverage = np.mean(jsf_coverages)

            mean_full_automation_accuracy = np.mean(full_automation_accuracies)
            mean_full_automation_accuracy_std = np.std(full_automation_accuracies)
            # mean_full_automation_coverage = 100.00

            mean_moae_accuracy = np.mean(moae_accuracies)
            mean_moae_accuracy_std = np.std(moae_accuracies)
            # mean_moae_coverage = 100.00

            mean_mohe_accuracy = np.mean(mohe_accuracies)
            mean_mohe_accuracy_std = np.std(mohe_accuracies)
            # mean_mohe_coverage = 0.00
            # print(tabulate([['Our Approach', mean_our_approach_accuracy, mean_our_approach_coverage],
            #                 ['JSF', mean_jsf_accuracy, mean_jsf_coverage],
            #                 ['--------', '--------', '--------'],
            #                 ['Full Automation', mean_full_automation_accuracy, mean_full_automation_coverage],
            #                 ['MOAE', mean_moae_accuracy, mean_moae_coverage],
            #                 ['MOHE', mean_mohe_accuracy, mean_mohe_coverage],
            #                 ['Random Expert', mean_avg_expert_accuracy, mean_avg_expert_coverage],
            #                 ['Best Expert', mean_best_expert_accuracy, mean_best_expert_coverage]],
            #                headers=['Method', 'Accuracy', 'Coverage']))
            # print(tabulate([['Our Approach', mean_our_approach_accuracy, mean_our_approach_coverage]],
            #             headers=['Method', 'Accuracy', 'Coverage']))
            # 文件路径
            file_path = f"./log/IJCAI-22/all-exp-NIH.txt"

            # 将数据追加写入文件
            with open(file_path, 'a') as file:
                file.write(f"TARGET: {TARGET}\n")
                file.write(f"IDS: {LABELER_IDS[0]} AND {LABELER_IDS[1]}\n")
                file.write(f"Best expert: {mean_best_expert_accuracy:.8f} {mean_best_expert_accuracy_std:.8f}\n")
                file.write(f"Avg expert: {mean_avg_expert_accuracy:.8f} {mean_avg_expert_accuracy_std:.8f}\n")
                file.write(f"JSF: {mean_jsf_accuracy:.8f} {mean_jsf_accuracy_std:.8f}\n")
                file.write(f"Full automation: {mean_full_automation_accuracy:.8f} {mean_full_automation_accuracy_std:.8f}\n")
                file.write(f"MoAE: {mean_moae_accuracy:.8f} {mean_moae_accuracy_std:.8f}\n")
                file.write(f"MoHE: {mean_mohe_accuracy:.8f} {mean_mohe_accuracy_std:.8f}\n")
                file.write(f"IJCAI: {mean_our_approach_accuracy:.8f} {mean_our_approach_accuracy_std:.8f}\n")
                file.write(f"-------------------------------------------------------------------------------\n")

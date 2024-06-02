import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from glob import glob
import torch
from torch.utils.data import DataLoader, Dataset
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')

test_healthy_path = "./AML_project/SVD/test/healthy"
test_pathology_path = "./AML_project/SVD/test/pathology"
train_healthy_path = "./AML_project/SVD/train/healthy"
train_pathology_path = "./AML_project/SVD/train/pathology"

train_file_names_health = os.listdir(train_healthy_path)
test_file_names_health = os.listdir(test_healthy_path)

train_file_names_patho = os.listdir(train_pathology_path)
test_file_names_patho = os.listdir(test_pathology_path)

import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(831)

sr = 20000


def load_audio(file_names, path, sr=22050):
    audios = []
    for audio in tqdm(file_names):
        # librosa?? ?????? ?????? ?��?
        file_path = os.path.join(path, audio)
        an_audio, _ = librosa.load(file_path, sr=sr)
        audios.append(an_audio)
    return audios


def load_train_test_data(train_path, test_path, train_file_names, test_file_names, sr=22050):
    # ??? ?????? ?��?
    print("Loading training data...")
    train_data = load_audio(train_file_names, train_path, sr)
    print("Training data loaded.")

    # ???? ?????? ?��?
    print("Loading testing data...")
    test_data = load_audio(test_file_names, test_path, sr)
    print("Testing data loaded.")

    return train_data, test_data

train_data_patho, test_data_patho = load_train_test_data(train_pathology_path, test_pathology_path, train_file_names_patho, test_file_names_patho)

train_labels_patho = np.ones(len(train_data_patho))
test_labels_patho = np.ones(len(test_data_patho))

train_data_healthy, test_data_healthy = load_train_test_data(train_healthy_path, test_healthy_path, train_file_names_health, test_file_names_health)


train_labels_healthy = np.zeros(len(train_data_healthy)) 
test_labels_healthy = np.zeros(len(test_data_healthy))   



train_data = train_data_healthy + train_data_patho

train_labels = np.concatenate([train_labels_healthy, train_labels_patho])

test_data = test_data_healthy + test_data_patho

test_labels = np.concatenate([test_labels_healthy, test_labels_patho])


def pad_sequences(data, max_length, padding_value=0):
    padded_data = []
    for seq in data:
        # ???????? ????? ??? ??????? ��???? ?��? ???
        if len(seq) < max_length:
            padded_seq = np.pad(seq, (0, max_length - len(seq)), mode='constant', constant_values=padding_value)
        # ???????? ??? ??????? ?????? ???
        else:
            padded_seq = seq[:max_length]
        padded_data.append(padded_seq)
    return np.array(padded_data)


max_length = max(max(len(audio) for audio in train_data), max(len(audio) for audio in test_data))
train_data_padded = pad_sequences(train_data, max_length)
test_data_padded = pad_sequences(test_data, max_length)

print("Padded train data shape:", np.array(train_data_padded).shape)
print("Padded test data shape:", np.array(test_data_padded).shape)




def extract_mfcc_features(audio_data, sr=22050, n_mfcc=40):
    mfcc_features = []
    for audio in audio_data:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_features.append(mfcc)
    return mfcc_features

def extract_mel_features(audio_data, sr=22050, n_mels=40):
    mel_features = []
    for audio in audio_data:
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_features.append(mel)
    return mel_features


train_mfcc_features = extract_mfcc_features(train_data_padded)
test_mfcc_features = extract_mfcc_features(test_data_padded)

train_mfcc_features = np.array(train_mfcc_features)
test_mfcc_features = np.array(test_mfcc_features)


train_mel_features = extract_mel_features(train_data_padded)
test_mel_features = extract_mel_features(test_data_padded)

train_mel_features = np.array(train_mel_features)
test_mel_features = np.array(test_mel_features)








import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(IdentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(F3)
        self.conv_shortcut = nn.Conv2d(in_channels, F3, kernel_size=(1, 1))
        self.bn_shortcut = nn.BatchNorm2d(F3)
        self.activation = nn.ReLU()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.conv_shortcut(shortcut)
        shortcut = self.bn_shortcut(shortcut)

        x += shortcut
        x = self.activation(x)

        return x


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride):
        super(ConvolutionalBlock, self).__init__()
        F1, F2, F3 = filters
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=(1, 1), stride=stride)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(F3)
        self.conv_shortcut = nn.Conv2d(in_channels, F3, kernel_size=(1, 1), stride=stride)
        self.bn_shortcut = nn.BatchNorm2d(F3)
        self.activation = nn.ReLU()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.conv_shortcut(shortcut)
        shortcut = self.bn_shortcut(shortcut)

        x += shortcut
        x = self.activation(x)

        return x
    

class ResNet50_(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.conv_block1 = ConvolutionalBlock(64, [64, 64, 256], 3, 1)
        self.id_block1_1 = IdentityBlock(256, [64, 64, 256], 3)
        self.id_block1_2 = IdentityBlock(256, [64, 64, 256], 3)

        self.conv_block2 = ConvolutionalBlock(256, [128, 128, 512], 3, 2)
        self.id_block2_1 = IdentityBlock(512, [128, 128, 512], 3)
        self.id_block2_2 = IdentityBlock(512, [128, 128, 512], 3)
        self.id_block2_3 = IdentityBlock(512, [128, 128, 512], 3)

        self.conv_block3 = ConvolutionalBlock(512, [256, 256, 1024], 3, 2)
        self.id_block3_1 = IdentityBlock(1024, [256, 256, 1024], 3)
        self.id_block3_2 = IdentityBlock(1024, [256, 256, 1024], 3)
        self.id_block3_3 = IdentityBlock(1024, [256, 256, 1024], 3)
        self.id_block3_4 = IdentityBlock(1024, [256, 256, 1024], 3)
        self.id_block3_5 = IdentityBlock(1024, [256, 256, 1024], 3)

        self.conv_block4 = ConvolutionalBlock(1024, [512, 512, 2048], 3, 2)
        self.id_block4_1 = IdentityBlock(2048, [512, 512, 2048], 3)
        self.id_block4_2 = IdentityBlock(2048, [512, 512, 2048], 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv_block1(x)
        x = self.id_block1_1(x)
        x = self.id_block1_2(x)

        x = self.conv_block2(x)
        x = self.id_block2_1(x)
        x = self.id_block2_2(x)
        x = self.id_block2_3(x)

        x = self.conv_block3(x)
        x = self.id_block3_1(x)
        x = self.id_block3_2(x)
        x = self.id_block3_3(x)
        x = self.id_block3_4(x)
        x = self.id_block3_5(x)

        x = self.conv_block4(x)
        x = self.id_block4_1(x)
        x = self.id_block4_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def get_data_loader(features, labels, batch_size=16, shuffle=True):
    dataset = CustomDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


class EnsembleModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EnsembleModel, self).__init__()
        self.resnet1 = ResNet50_(num_classes=num_classes)
        self.resnet2 = ResNet50_(num_classes=num_classes)
        
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, mfcc_inputs, melspectrogram_inputs):
        
        mfcc_features = self.resnet1(mfcc_inputs)
        
        melspectrogram_features = self.resnet2(melspectrogram_inputs)
        
        combined_features = torch.cat((mfcc_features, melspectrogram_features), dim=1)
        
        output = self.fc(combined_features)
        return output
    




def evaluate_model(loader_mfcc, loader_mel, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (mfcc_inputs, labels_mfcc), (mel_inputs, labels_mel) in zip(loader_mfcc, loader_mel):
            mfcc_inputs, labels_mfcc = mfcc_inputs.to(device), labels_mfcc.to(device)
            mel_inputs, labels_mel = mel_inputs.to(device), labels_mel.to(device)

            outputs = model(mfcc_inputs, mel_inputs)
            _, predicted = outputs.max(1)
            total += labels_mfcc.size(0)
            correct += predicted.eq(labels_mfcc).sum().item()

    model_accuracy = 100. * correct / total
    return model_accuracy

import copy

def train_ensemble_model(train_loader_mfcc, train_loader_mel, val_loader_mfcc, val_loader_mel,
                         test_loader_mfcc, test_loader_mel, model, num_epochs=20, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_model_wts = model.state_dict()
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for (mfcc_inputs, labels_mfcc), (mel_inputs, labels_mel) in zip(train_loader_mfcc, train_loader_mel):
            mfcc_inputs, labels_mfcc = mfcc_inputs.to(device), labels_mfcc.to(device)
            mel_inputs, labels_mel = mel_inputs.to(device), labels_mel.to(device)

            optimizer.zero_grad()

            outputs = model(mfcc_inputs, mel_inputs)
            loss = criterion(outputs, labels_mfcc)  
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels_mfcc.size(0)
            correct += predicted.eq(labels_mfcc).sum().item()

        epoch_loss = running_loss / len(train_loader_mfcc)
        accuracy = 100. * correct / total
        
        val_accuracy = evaluate_model(val_loader_mfcc, val_loader_mel, model)
        test_accuracy = evaluate_model(test_loader_mfcc, test_loader_mel, model)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}%")

        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

    
    model.load_state_dict(best_model_wts)
    print("Loaded the best model weights with Validation Accuracy: {:.2f}%".format(best_val_accuracy))

    return model



from sklearn.model_selection import KFold


def k_fold_cross_validation(train_mfcc_features, train_mel_features, train_labels,
                            test_mfcc_features, test_mel_features, test_labels,
                            num_epochs=20, k=5, learning_rate=0.001):
    kf = KFold(n_splits=k, shuffle=True)
    best_model = None
    best_accuracy = 0.0

    for fold, (train_index, test_index) in enumerate(kf.split(train_mfcc_features)):
        print(f"Fold {fold+1}/{k}")
        model = EnsembleModel(num_classes=2)
        train_loader_mfcc = get_data_loader(train_mfcc_features[train_index], train_labels[train_index])
        train_loader_mel = get_data_loader(train_mel_features[train_index], train_labels[train_index])

        vaild_loader_mfcc = get_data_loader(train_mfcc_features[test_index], train_labels[test_index])
        vaild_loader_mel = get_data_loader(train_mel_features[test_index], train_labels[test_index])

        test_loader_mfcc = get_data_loader(test_mfcc_features, test_labels)
        test_loader_mel = get_data_loader(test_mel_features, test_labels)
        

        model = train_ensemble_model(train_loader_mfcc, train_loader_mel,
                                     vaild_loader_mfcc, vaild_loader_mel,
                                     test_loader_mfcc, test_loader_mel,
                                     model, num_epochs=num_epochs, learning_rate=learning_rate)

        
        train_accuracy = evaluate_model(train_loader_mfcc, train_loader_mel, model)
        vaild_acc = evaluate_model(vaild_loader_mfcc, vaild_loader_mel, model)
        print(f"Train Accuracy (Fold {fold+1}): {train_accuracy:.2f}, Vaildation Accuracy (Fold {fold+1}): {vaild_acc:.2f}%")



        if vaild_acc > best_accuracy:
            best_model = model
            best_accuracy = vaild_acc

    print(f"Best Vaildation Accuracy: {best_accuracy:.2f}%")
    return best_model

epochs_arr = [30]
lr_arr = [0.0001, 0.001, 0.01]

for cur_epoch in epochs_arr:
    for cur_lr in lr_arr:
        print(f'epoch : {cur_epoch}, learning rate : {cur_lr}')
        best_model = k_fold_cross_validation(train_mfcc_features, train_mel_features, train_labels,
                                     test_mfcc_features, test_mel_features, test_labels,
                                     num_epochs=cur_epoch, k=5, learning_rate=cur_lr)


        test_loader_mfcc = get_data_loader(test_mfcc_features, test_labels)
        test_loader_mel = get_data_loader(test_mel_features, test_labels)

        test_score = evaluate_model(test_loader_mfcc, test_loader_mel, best_model)
        print(f"best model's Test score : {test_score}%\n\n\n")
        print("-----------------------------------------------------------------\n\n\n")
        

        

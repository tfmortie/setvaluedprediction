"""
Data module for paper "Set-valued prediction in hierarchical classification with constrained representation complexity"

Author: Thomas Mortier
Date: January 2022
"""
import ast
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

""" general protein dataset file """


class ProteinDataset(Dataset):
    def __init__(self, struct_path, svm_path):
        f = open(struct_path, "r")
        struct = None
        for line in f:
            struct = ast.literal_eval(line.strip())
        f.close()
        # create label mapping based on struct
        num_classes = len(struct[0])
        self.lbl_to_path = []
        for i in range(1, num_classes + 1):
            path = []
            for n in struct:
                if i in n:
                    path.append(str(n))
            self.lbl_to_path.append(";".join(path))
        # first get dimensionality
        f = open(svm_path)
        d, n = 0, 0
        for line in f:
            d = max(int(line.strip().split(" ")[-1].split(":")[0]), d)
            n += 1
        f.close()
        # now process data
        self.X = np.zeros((n, d + 1))
        self.y = []
        f = open(svm_path)
        for i, line in enumerate(f):
            line = line.strip().split(" ")
            # process label
            self.y.append(self.lbl_to_path[int(line[0]) - 1])
            for t in line[1:]:
                t_splitted = t.split(":")
                self.X[i, int(t_splitted[0])] = float(t_splitted[1])
        f.close()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        prot = self.X[idx]
        prot = torch.Tensor(prot)
        return prot, self.y[idx]


""" general protein dataloaders """


def ProteinDataloaders(args):
    trainval_dataset = ProteinDataset(
        args.datapath + "/hierarchy_full.txt",
        args.datapath + "/proteins_train_tfidfoh.svm",
    )
    test_dataset = ProteinDataset(
        args.datapath + "/hierarchy_full.txt",
        args.datapath + "/proteins_test_tfidfoh.svm",
    )
    trainval_dataloader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    return trainval_dataloader, test_dataloader, trainval_dataset.y


""" general bacteria dataset file """


class BacteriaDataset(Dataset):
    def __init__(self, dim, struct_path, svm_path):
        f = open(struct_path, "r")
        struct = None
        for line in f:
            struct = ast.literal_eval(line.strip())
        f.close()
        # create label mapping based on struct
        num_classes = len(struct[0])
        self.lbl_to_path = []
        for i in range(1, num_classes + 1):
            path = []
            for n in struct:
                if i in n:
                    path.append(str(n))
            self.lbl_to_path.append(";".join(path))
        # first get dimensionality
        f = open(svm_path)
        n = 0
        for line in f:
            n += 1
        f.close()
        # now process data
        self.X = np.zeros((n, dim))
        self.y = []
        f = open(svm_path)
        for i, line in enumerate(f):
            line = line.strip().split(" ")
            # process label
            self.y.append(self.lbl_to_path[int(line[0]) - 1])
            for t in line[1:]:
                t_splitted = t.split(":")
                self.X[i, int(t_splitted[0])] = float(t_splitted[1])
        f.close()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        bact = self.X[idx]
        bact = torch.Tensor(bact)
        return bact, self.y[idx]


""" general bacteria dataloaders """


def BacteriaDataloaders(args):
    trainval_dataset = BacteriaDataset(
        args.dim,
        args.datapath + "/hierarchy_full.txt",
        args.datapath + "/bacteria_train_tfidf.svm",
    )
    test_dataset = BacteriaDataset(
        args.dim,
        args.datapath + "/hierarchy_full.txt",
        args.datapath + "/bacteria_test_tfidf.svm",
    )
    trainval_dataloader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    return trainval_dataloader, test_dataloader, trainval_dataset.y


""" general Caltech dataset file """


class CaltechDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.transform = transform
        df = pd.read_csv(csv_path)
        self.X = df["path"]
        self.y = df["label"]

    def __len__(self):
        return len(self.X.index)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


""" general Caltech dataloaders """


def CaltechDataloaders(args):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((args.dim, args.dim)),
            transforms.ToTensor(),
        ]
    )
    trainval_dataset = CaltechDataset(args.datapath + "/TRAINVAL.csv", transform)
    test_dataset = CaltechDataset(args.datapath + "/TEST.csv", transform)
    trainval_dataloader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    return trainval_dataloader, test_dataloader, trainval_dataset.y


""" general Plantclef dataset file """


class PlantclefDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.transform = transform
        df = pd.read_csv(csv_path)
        self.X = df["path"]
        self.y = df["label"]

    def __len__(self):
        return len(self.X.index)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


""" general Caltech dataloaders """


def PlantclefDataloaders(args):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((args.dim, args.dim)),
            transforms.ToTensor(),
        ]
    )
    trainval_dataset = PlantclefDataset(args.datapath + "/TRAINVAL.csv", transform)
    test_dataset = PlantclefDataset(args.datapath + "/TEST.csv", transform)
    trainval_dataloader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    return trainval_dataloader, test_dataloader, trainval_dataset.y


""" dictionary representing dataset->dataloaders mapper """
GET_DATASETLOADER = {
    "CAL101": CaltechDataloaders,
    "CAL256": CaltechDataloaders,
    "PLANTCLEF": PlantclefDataloaders,
    "PROT": ProteinDataloaders,
    "BACT": BacteriaDataloaders,
}

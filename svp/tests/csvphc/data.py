"""
Data module for experiments "Distribution-free set-valued prediction in hierarchical classification with finite sample guarantees"

Author: Thomas Mortier
Date: February 2024
"""
import ast
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, random_split

""" general protein dataset file """


class ProteinDataset(Dataset):
    def __init__(self, struct_path=None, svm_path=None, X=None, y=None):
        if struct_path:
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
        elif X is not None and y is not None:
            self.X = X
            self.y = y
        else:
            raise ValueError("Either struct_path and svm_path or X and y must be provided.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        prot = self.X[idx]
        prot = torch.Tensor(prot)
        return prot, self.y[idx]


""" general protein dataloaders """


def ProteinDataloaders(args):
    # extract train and test data and merge
    trainval_dataset = ProteinDataset(
        struct_path=args.datapath + "/hierarchy_full.txt",
        svm_path=args.datapath + "/proteins_train_tfidfoh.svm",
    )
    test_dataset = ProteinDataset(
        struct_path=args.datapath + "/hierarchy_full.txt",
        svm_path=args.datapath + "/proteins_test_tfidfoh.svm",
    ) 
    # now split in train and calibration 
    generator = torch.Generator().manual_seed(args.randomseeddata)
    train_dataset, val_dataset = random_split(trainval_dataset, [args.trainratio, 1-args.trainratio], generator=generator)
    tune_dataset = None
    if args.tunelk:
        cal_dataset, tune_dataset, test_dataset = random_split(test_dataset, [(1-args.testratio)/2, (1-args.testratio)/2, args.testratio], generator=generator)
    else:
        cal_dataset, test_dataset = random_split(test_dataset, [1-args.testratio, args.testratio], generator=generator)
    # get the loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    cal_dataloader = torch.utils.data.DataLoader(
        cal_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    tune_dataloader = None
    if tune_dataset is not None:
        tune_dataloader = torch.utils.data.DataLoader(
            tune_dataset,
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

    return train_dataloader, val_dataloader, cal_dataloader, tune_dataloader, test_dataloader, trainval_dataset.y


""" general bacteria dataset file """


class BacteriaDataset(Dataset):
    def __init__(self, dim=None, struct_path=None, svm_path=None, X=None, y=None):
        if struct_path:
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
        elif X is not None and y is not None:
            self.X = X
            self.y = y
        else:
            raise ValueError("Either struct_path and svm_path or X and y must be provided.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        bact = self.X[idx]
        bact = torch.Tensor(bact)
        return bact, self.y[idx]


""" general bacteria dataloaders """


def BacteriaDataloaders(args):
    # extract train and test data and merge
    trainval_dataset = BacteriaDataset(
        dim=args.dim,
        struct_path=args.datapath + "/hierarchy_full.txt",
        svm_path=args.datapath + "/bacteria_train_tfidf.svm",
    )
    test_dataset = BacteriaDataset(
        dim=args.dim,
        struct_path=args.datapath + "/hierarchy_full.txt",
        svm_path=args.datapath + "/bacteria_test_tfidf.svm",
    ) 
    # now split in train and calibration 
    generator = torch.Generator().manual_seed(args.randomseeddata)
    train_dataset, val_dataset = random_split(trainval_dataset, [args.trainratio, 1-args.trainratio], generator=generator)
    tune_dataset = None
    if args.tunelk:
        cal_dataset, tune_dataset, test_dataset = random_split(test_dataset, [(1-args.testratio)/2, (1-args.testratio)/2, args.testratio], generator=generator)
    else:
        cal_dataset, test_dataset = random_split(test_dataset, [1-args.testratio, args.testratio], generator=generator)
    # get the loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    cal_dataloader = torch.utils.data.DataLoader(
        cal_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    tune_dataloader = None
    if tune_dataset is not None:
        tune_dataloader = torch.utils.data.DataLoader(
            tune_dataset,
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

    return train_dataloader, val_dataloader, cal_dataloader, tune_dataloader, test_dataloader, trainval_dataset.y


""" general Caltech dataset file """


class CaltechDataset(Dataset):
    def __init__(self, csv_path=None, X=None, y=None, transform=None):
        self.transform = transform
        if csv_path:
            df = pd.read_csv(csv_path)
            self.X = df["path"]
            self.y = df["label"]
        elif X is not None and y is not None:
            self.X = X
            self.y = y
        else:
            raise ValueError("Either csv_path or X and y must be provided.")

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
    # extract train and test data and merge
    trainval_dataset = CaltechDataset(csv_path=args.datapath + "/TRAINVAL.csv", transform=transform)
    test_dataset = CaltechDataset(csv_path=args.datapath + "/TEST.csv", transform=transform) 
    # now split in train and calibration 
    generator = torch.Generator().manual_seed(args.randomseeddata)
    train_dataset, val_dataset = random_split(trainval_dataset, [args.trainratio, 1-args.trainratio], generator=generator)
    tune_dataset = None
    if args.tunelk:
        cal_dataset, tune_dataset, test_dataset = random_split(test_dataset, [(1-args.testratio)/2, (1-args.testratio)/2, args.testratio], generator=generator)
    else:
        cal_dataset, test_dataset = random_split(test_dataset, [1-args.testratio, args.testratio], generator=generator)
    # get the loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    cal_dataloader = torch.utils.data.DataLoader(
        cal_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    tune_dataloader = None
    if tune_dataset is not None:
        tune_dataloader = torch.utils.data.DataLoader(
            tune_dataset,
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

    return train_dataloader, val_dataloader, cal_dataloader, tune_dataloader, test_dataloader, trainval_dataset.y


""" general Plantclef dataset file """


class PlantclefDataset(Dataset):
    def __init__(self, csv_path=None, X=None, y=None, transform=None):
        self.transform = transform
        if csv_path:
            df = pd.read_csv(csv_path)
            self.X = df["path"]
            self.y = df["label"]
        elif X is not None and y is not None:
            self.X = X
            self.y = y
        else:
            raise ValueError("Either csv_path or X and y must be provided.")

    def __len__(self):
        return len(self.X.index)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


""" general Plantclef dataloaders """


def PlantclefDataloaders(args):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((args.dim, args.dim)),
            transforms.ToTensor(),
        ]
    )
    # extract train and test data and merge
    trainval_dataset = PlantclefDataset(csv_path=args.datapath + "/TRAINVAL.csv", transform=transform)
    test_dataset = PlantclefDataset(csv_path=args.datapath + "/TEST.csv", transform=transform) 
    # now split in train and calibration 
    generator = torch.Generator().manual_seed(args.randomseeddata)
    train_dataset, val_dataset = random_split(trainval_dataset, [args.trainratio, 1-args.trainratio], generator=generator)
    tune_dataset = None
    if args.tunelk:
        cal_dataset, tune_dataset, test_dataset = random_split(test_dataset, [(1-args.testratio)/2, (1-args.testratio)/2, args.testratio], generator=generator)
    else:
        cal_dataset, test_dataset = random_split(test_dataset, [1-args.testratio, args.testratio], generator=generator)
    # get the loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    cal_dataloader = torch.utils.data.DataLoader(
        cal_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    tune_dataloader = None
    if tune_dataset is not None:
        tune_dataloader = torch.utils.data.DataLoader(
            tune_dataset,
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

    return train_dataloader, val_dataloader, cal_dataloader, tune_dataloader, test_dataloader, trainval_dataset.y

""" general Cifar10 dataset file """


class CifarDataset(Dataset):
    def __init__(self, csv_path=None, X=None, y=None, transform=None):
        self.transform = transform
        if csv_path:
            df = pd.read_csv(csv_path)
            self.X = df["path"]
            self.y = df["label"]
        elif X is not None and y is not None:
            self.X = X
            self.y = y
        else:
            raise ValueError("Either csv_path or X and y must be provided.")

    def __len__(self):
        return len(self.X.index)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


""" general Cifar10 dataloaders """


def CifarDataloaders(args):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((args.dim, args.dim)),
            transforms.ToTensor(),
        ]
    )
    # extract train and test data and merge
    trainval_dataset = CifarDataset(csv_path=args.datapath + "/TRAINVAL.csv", transform=transform)
    test_dataset = CifarDataset(csv_path=args.datapath + "/TEST.csv", transform=transform) 
    # now split in train and calibration 
    generator = torch.Generator().manual_seed(args.randomseeddata)
    train_dataset, val_dataset = random_split(trainval_dataset, [args.trainratio, 1-args.trainratio], generator=generator)
    tune_dataset = None
    if args.tunelk:
        cal_dataset, tune_dataset, test_dataset = random_split(test_dataset, [(1-args.testratio)/2, (1-args.testratio)/2, args.testratio], generator=generator)
    else:
        cal_dataset, test_dataset = random_split(test_dataset, [1-args.testratio, args.testratio], generator=generator)
    # get the loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    cal_dataloader = torch.utils.data.DataLoader(
        cal_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    tune_dataloader = None
    if tune_dataset is not None:
        tune_dataloader = torch.utils.data.DataLoader(
            tune_dataset,
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

    return train_dataloader, val_dataloader, cal_dataloader, tune_dataloader, test_dataloader, trainval_dataset.y


""" dictionary representing dataset->dataloaders mapper """
GET_DATASETLOADER = {
    "CIFAR10": CifarDataloaders,
    "CAL101": CaltechDataloaders,
    "CAL256": CaltechDataloaders,
    "PLANTCLEF": PlantclefDataloaders,
    "PROT": ProteinDataloaders,
    "BACT": BacteriaDataloaders,
}

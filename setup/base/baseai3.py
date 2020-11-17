import configparser
import matplotlib.pyplot as plt
import argparse
import os
import abc
import sys
import torch
import torch.nn as nn
import time
import numpy as np
from torch import optim
from torch.utils import data
from torchvision import transforms
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tool.utils import makedir

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseNet(nn.Module):
    def __init__(self, name=None, id=None, device=DEVICE):
        super(BaseNet, self).__init__()
        self.name = name
        self.id = id
        self.device = device
        # self.mark = int(time.time())

    def paraminit(self):
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        pass


class BaseDataset(data.Dataset):
    def __init__(self):
        self.dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        pass


class BaseTrain(metaclass=abc.ABCMeta):
    def __init__(self, trainName=None, cfgfile=None, index=None, basepath=None):

        # optim.Adam
        self.cfgfile = cfgfile
        self.basepath = basepath
        self.parser = argparse.ArgumentParser(description="base class for network training")
        self.args = self._argparser()

        if self.args.name == None:
            if trainName != None:
                self.trainName = trainName
            else:
                raise ValueError
        else:
            self.trainName = self.args.name

        self.index = self.args.index if self.args.index else index
        self.trainID = f"{self.trainName}_{self.index}"

        self._cfginit(cfgfile)

        self._loginit()
        self.title()
        self._deviceinit()
        self.set_module()
        self._moduleinit()
        self._datasetinit()

        self.lossDict = {1: "nn.MSELoss()",
                         2: "nn.CrossEntropyLoss()",
                         3: "nn.BCELoss()",
                         4: "nn.BCEWithLogitsLoss()",
                         5: "nn.MSELoss(reduction='sum')"}
        self.optimDict = {1: f"optim.Adam(self.net.parameters())",
                          2: f"optim.SGD(self.net.parameters(), lr={self.lr})"}

        # self._paraminit()

    def _argparser(self):
        self.parser.add_argument("-n", "--name", type=str,
                            default=None, help="the trainid name to train")
        self.parser.add_argument("-i", "--index", type=int,
                            default=None, help="the netfile index to train")
        self.parser.add_argument("-m", "--module", type=str,
                            default=None, help="the module name")
        self.parser.add_argument("-e", "--epoch", type=int,
                            default=None, help="number of epochs")
        self.parser.add_argument("-b", "--batchsize", type=int,
                            default=None, help="mini-batch size")
        self.parser.add_argument("-w", "--numworkers", type=int, default=None,
                            help="number of threads used during batch generation")
        self.parser.add_argument("-r", "--lr", type=float, default=None,
                            help="learning rate for gradient descent")
        self.parser.add_argument("-p", "--checkpoint", type=int,
                            default=None, help="print frequency")
        self.parser.add_argument("-t", "--threshold", type=float, default=None,
                            help="interval between evaluations on validation set")
        self.parser.add_argument("-a", "--alpha", type=float,
                            default=None, help="ratio of conf and offset loss")
        self.parser.add_argument("-s", "--save", type=int,
                            default=None, help="if need save")
        self.parser.add_argument("-c", "--cudanum", type=int,
                            default=0, help="the number of cuda")
        self.parser.add_argument("-l", "--lossid", type=int,
                            default=None, help="the key index of loss")
        self.parser.add_argument("-o", "--optimid", type=int,
                            default=None, help="the key index of optimizer")
        self.parser.add_argument("-f", "--train", type=int,
                            default=None, help="the key index of optimizer")
        return self.parser.parse_args()

    def _cfginit(self, cfgfile):
        self.paramDict = {}
        self.config = configparser.ConfigParser()
        self.config.read(cfgfile)

        saveDir_ = self.config.get(self.trainName, "SAVE_DIR")

        self.module = self.args.module if self.args.module else self.config.get(self.trainName, "MODULE")
        self.imgDir = self.config.get(self.trainName, "IMG_DIR")
        self.imgTestDir = self.config.get(self.trainName, "IMGTEST_DIR")
        self.labelDir = self.config.get(self.trainName, "LABEL_DIR")

        self.saveDir = os.path.join(self.basepath, saveDir_)

        self.save = self.args.save if (self.args.save != None) else self.config.getint(self.trainName, "SAVE")
        self.epoch = self.args.epoch if self.args.epoch else self.config.getint(self.trainName, "EPOCH")

        self.alpha = self.args.alpha if self.args.alpha else self.config.getfloat(self.trainName, "ALPHA")

        self.batchSize = self.args.batchsize if self.args.batchsize else self.config.getint(self.trainName, "BATCHSIZE")
        self.numWorkers = self.args.numworkers if self.args.numworkers else self.config.getint(self.trainName, "NUMWORKERS")
        self.checkPoint = self.args.checkpoint if self.args.checkpoint else self.config.getint(self.trainName, "CHECKPOINT")

        self.threshold = self.args.threshold if self.args.threshold else self.config.getfloat(self.trainName, "THRESHOLD")
        self.lr = self.args.lr if self.args.lr else self.config.getfloat(self.trainName, "LEARNING_RATE")
        self.lossid = self.args.lossid if self.args.lossid else self.config.getint(self.trainName, "LOSSID")
        self.optimid = self.args.optimid if self.args.optimid else self.config.getint(self.trainName, "OPTIMID")

        self.posnum = self.config.getint(self.trainName, "POSNUM")
        self.negnum = self.config.getint(self.trainName, "NEGNUM")

        self.subSaveDir = os.path.join(self.saveDir, f"{self.trainName}")
        self.subNetSaveDir = os.path.join(self.subSaveDir, f"{self.trainID}")

        if self.save and self.args.train:
            makedir(self.subSaveDir)
            makedir(self.subNetSaveDir)

        self.transform = transforms.ToTensor()

    def _deviceinit(self):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.args.cudanum}")

    def paraminit(self, **kwargs):
        self.paramDict['netName'] = self.net.__class__.__name__
        self.paramDict['trainName'] = self.trainName
        self.paramDict['trainID'] = self.trainID
        self.paramDict['cfgfile'] = self.cfgfile
        self.paramDict['device'] = self.device
        self.paramDict['basepath'] = self.basepath
        self.paramDict['loss'] = self.lossDict[self.lossid]
        self.paramDict['optimizer'] = self.optimDict[self.optimid]
        self.paramDict['imgDir'] = self.imgDir
        self.paramDict['imgTestDir'] = self.imgTestDir
        self.paramDict['labelDir'] = self.labelDir
        self.paramDict['saveDir'] = self.saveDir
        self.paramDict['save'] = self.save
        self.paramDict['epoch'] = self.epoch
        self.paramDict['alpha'] = self.alpha
        self.paramDict['batchSize'] = self.batchSize
        self.paramDict['numWorkers'] = self.numWorkers
        self.paramDict['checkPoint'] = self.checkPoint
        self.paramDict['threshold'] = self.threshold
        self.paramDict['lr'] = self.lr
        self.paramDict['posnum'] = self.posnum
        self.paramDict['negnum'] = self.negnum
        self.paramDict['transform'] = self.transform
        for key, value in kwargs.items():
            self.paramDict[key] = value
        paramtitle = f"{self.trainID} Parameters"
        self.printlog(f"{paramtitle:^50}", newline=False)
        self.printlog(self.paramDict)

    def _moduleinit(self):
        self.netfile = os.path.join(self.subSaveDir, f"{self.trainID}.pt")
        self.netfile_backup = os.path.join(self.subSaveDir, f"{self.trainID}.pt.bak")
        self.net.to(self.device)
        if os.path.exists(self.netfile):
            self.net.load_state_dict(torch.load(self.netfile))
            print("load successfully")
            print()

    def _loginit(self, test=True):
        self.logFile = os.path.join(self.subSaveDir, f"{self.trainID}.log")
        self.logTXTFile = os.path.join(self.subSaveDir, f"{self.trainID}.txt")

        self.logFile_backup = os.path.join(
            self.subSaveDir, f"{self.trainID}_.log.bak")
        self.logFileTest_backup = os.path.join(
            self.subSaveDir, f"{self.trainID}_test.log.bak")
        if os.path.exists(self.logFile):
            self.logDict = torch.load(self.logFile)
        else:
            self.logDict = {"i": [], "j": [], "k": [], "loss": [], "accuracy": []}
        if test:
            self.logFileTest = os.path.join(
                self.subSaveDir, f"{self.trainID}_test.log")
            if os.path.exists(self.logFileTest):
                self.logDictTest = torch.load(self.logFileTest)
            else:
                self.logDictTest = {"i": [], "j": [], "k": [], "accuracy": []}

    def _log(self, isTest: bool = False, **kwargs):
        if isTest:
            for key, value in kwargs.items():
                self.logDictTest[key].append(value)
            torch.save(self.logDictTest, self.logFileTest)
            time.sleep(0.1)
            torch.save(self.logDictTest, self.logFileTest_backup)
        else:
            for key, value in kwargs.items():
                self.logDict[key].append(value)
            torch.save(self.logDict, self.logFile)
            time.sleep(0.1)
            torch.save(self.logDict, self.logFile_backup)

    def title(self):
        starttime = time.strftime("%b/%d/%Y %H:%M:%S", time.localtime())
        titlestr = "\n" \
                   f"{'*' * 80}\n" \
                   f"*{self.trainID:^78s}*\n" \
                   f"*{' '.join(sys.argv):^78s}*\n" \
                   f"*{starttime:^78s}*\n" \
                   f"{'*' * 80}"
        self.printlog(titlestr)

    def printlog(self, *args, newline=True):
        if self.save and self.args.train:
            with open(self.logTXTFile, 'a', encoding='utf-8') as file:
                for arg in args:
                    if isinstance(arg, dict):
                        for key, value in arg.items():
                            print(f"{key:<11s}: ", end="")
                            print(value)
                            print(f"{key:<11s}: ", end="", file=file)
                            print(value, file=file)
                    else:
                        print(arg)
                        print(arg, file=file)
                if newline:
                    print(file=file)
                    print()
        else:
            for arg in args:
                if isinstance(arg, dict):
                    for key, value in arg.items():
                        print(f"{key:<11s}: ", end="")
                        print(value)
                else:
                    print(arg)
            if newline:
                print()

    def saveBatch(self):
        if self.save and self.args.train:
            torch.save(self.net.state_dict(), self.netfile)
            time.sleep(0.1)
            torch.save(self.net.state_dict(), self.netfile_backup)
            self._log(i=self.i, j=self.j, k=self.k, loss=self.loss.item(), accuracy=self.accuracy.item())

    def saveEpoch(self):
        if self.save and self.args.train:
            self.plot("loss", "accuracy")
            self.netfile_epoch = os.path.join(self.subNetSaveDir, f"{self.trainName}_{self.index}_{self.i}.pt")
            torch.save(self.net.state_dict(), self.netfile_epoch)

    def onehot(self, a, cls=2):
        b = torch.zeros(a.size(0), cls).scatter_(-1, a.view(-1, 1).long(), 1).to(self.device)
        return b

    def plot(self, *args, isTest=False):
        for item in args:
            plotName = f"plot_{self.trainID}_{item}.png"  # jpg in linux wrong
            plotPath = os.path.join(self.subSaveDir, plotName)
            if isTest:
                y = np.array(self.logDictTest[item])
            else:
                y = np.array(self.logDict[item])
            plt.clf()
            plt.title(item)
            plt.plot(y)
            plt.savefig(plotPath)

    def train(self, *args, **kwargs):
        print("train start")
        self.dataloader = data.DataLoader(self.dataset, self.batchSize, shuffle=True, num_workers=self.numWorkers)

        try:
            self.dataloaderLen = len(self.dataloader)
            # global i, j, k
            i = 0
            j = 0
            k = 0

            if len(self.logDict["i"]) > 0:
                i = self.logDict["i"][-1]
                j = self.logDict["j"][-1] + 1
                # k = len(self.logDict["i"]) * self.checkPoint
                k = self.logDict["k"][-1]
                if j >= self.dataloaderLen:
                    i += 1
                    j = 0
            self.i = i
            self.j = j
            self.k = k
            self.trainImpl()
        except Exception as e:
            self.printlog(e)
            exit(-1)

    # @abc.abstractmethod
    def trainImpl(self):

        while (self.i < self.epoch):
            print(self.net.name)
            self.net.train()
            for dataitem in self.dataloader:
                output = self.get_output(dataitem)

                self.loss = self.get_loss(output)

                self.step()

                # print((self.j+1), self.checkPoint)
                if 0 == (self.j + 1) % self.checkPoint or self.j == self.dataloaderLen - 1:
                    self.accuracy = self.get_accuracy(output)
                    result = self.get_result()
                    self.printlog(result, newline=False)
                    self.saveBatch()

                self.j += 1
                self.k += 1
                if self.j == self.dataloaderLen:
                    self.j = 0
                    break

            self.saveEpoch()
            self.test()

            self.i += 1

    def step(self):
        self._optimizer.zero_grad()
        self.loss.backward()
        self._optimizer.step()

    @property
    def _optimizer(self):
        return eval(self.optimDict[self.optimid])

    @property
    def _lossFn(self):
        return eval(self.lossDict[self.lossid])

    @abc.abstractmethod
    def set_module(self, *args, **kwargs):
        self.net = BaseNet(*args, **kwargs)

    @abc.abstractmethod
    def _datasetinit(self):
        self.dataset = []
        if 0 == len(self.dataset):
            print("dataset must be initialed")
            exit(-1)

    def test(self, *args, **kwargs):
        return

    def get_result(self, *args, **kwargs):
        result = f"epoch: {self.i:>4d}, batch: {self.j:>4d}, totalbatch: {self.k:>6d}, loss: {self.loss.item():.4f}, accuracy: {self.accuracy.item():.4f}"
        return result

    @abc.abstractmethod
    def get_loss(self, *args, **kwargs):
        loss = torch.Tensor([])
        if 0 == loss.size(0):
            print("get_loss function should be overrided")
            exit(-1)
        return loss

    @abc.abstractmethod
    def get_output(self, *args, **kwargs):
        output = torch.Tensor([])
        if 0 == output.size(0):
            print("get_loss function should be overrided")
            exit(-1)
        return output

    # @abc.abstractmethod
    def get_accuracy(self, *args, **kwargs):
        accuracy = torch.Tensor([])
        if 0 == accuracy.size(0):
            print("get_accuracy function should be overrided")
            exit(-1)
        return accuracy

    # @abc.abstractmethod
    def get_property(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def get_precision(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def get_recall(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def detect(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def analyze(self, *args, **kwargs):
        pass

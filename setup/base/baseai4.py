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
import traceback
from torch import optim
from torch.utils import data
from torchvision import transforms
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tool.utils import makedir

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseNet, self).__init__()
        self.name = self.__class__.__name__

    def paraminit(self):
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)

    def forward(self, *args, **kwargs):
        pass


class BaseDataset(data.Dataset):
    def __init__(self):
        self.dataset = []

    def __str__(self):
        return self.__class__.__name__

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        pass


class BaseTrain(metaclass=abc.ABCMeta):
    def __init__(self, ID=None, cfgfile=None, number=None, basepath=None):

        # optim.Adam
        self.cfgfile = cfgfile
        self.basepath = basepath

        self._argparserinit()


        if self.args.ID == None:
            if ID != None:
                self.ID = ID
            else:
                raise ValueError
        else:
            self.ID = self.args.ID

        self.number = self.args.number if self.args.number else number
        self.trainID = f"{self.ID}_{self.number}"

        self._cfginit(cfgfile)

        self._loginit()

        self._deviceinit()
        self.set_module()
        self._moduleinit()
        self._datasetinit()
        self._dataloaderinit()
        self.trainName = self.__class__.__name__
        self.netName = self.net.__class__.__name__
        self.lossDict = {1: "nn.MSELoss()",
                         2: "nn.CrossEntropyLoss()",
                         3: "nn.BCELoss()",
                         4: "nn.BCEWithLogitsLoss()",
                         5: "nn.MSELoss(reduction='sum')"}
        self.optimDict = {1: f"optim.Adam(self.net.parameters())",
                          2: f"optim.SGD(self.net.parameters(), lr={self.lr})"}
        self._title()
        self.loginfo()

    def __str__(self):
        return self.__class__.__name__

    def _argparserinit(self):
        parser = argparse.ArgumentParser(description="base class for network training")
        parser.add_argument("-i", "--ID", type=str,
                                 default=None, help="the trainid name to train")
        parser.add_argument("-n", "--number", type=int,
                                 default=None, help="the netfile index number to train")
        parser.add_argument("-m", "--module", type=str,
                                 default=None, help="the module name")
        parser.add_argument("-e", "--epoch", type=int,
                                 default=None, help="number of epochs")
        parser.add_argument("-b", "--batchsize", type=int,
                                 default=None, help="mini-batch size")
        parser.add_argument("-w", "--numworkers", type=int, default=None,
                                 help="number of threads used during batch generation")
        parser.add_argument("-r", "--lr", type=float, default=None,
                                 help="learning rate for gradient descent")
        parser.add_argument("-p", "--checkpoint", type=int,
                                 default=None, help="print frequency")
        parser.add_argument("-t", "--threshold", type=float, default=None,
                                 help="interval between evaluations on validation set")
        parser.add_argument("-a", "--alpha", type=float,
                                 default=None, help="ratio of conf and offset loss")
        parser.add_argument("-s", "--save", type=int,
                                 default=None, help="if need save")
        parser.add_argument("-c", "--cudanum", type=int,
                                 default=0, help="the number of cuda")
        parser.add_argument("-l", "--lossid", type=int,
                                 default=None, help="the key index of loss")
        parser.add_argument("-o", "--optimid", type=int,
                                 default=None, help="the key index of optimizer")
        parser.add_argument("-f", "--train", type=int,
                                 default=None, help="the key index of optimizer")
        self.args = parser.parse_args()

    def _cfginit(self, cfgfile):

        self.config = configparser.ConfigParser()
        self.config.read(cfgfile)

        saveDir_ = self.config.get(self.ID, "SAVEDIR")

        self.module = self.args.module if self.args.module else self.config.get(self.ID, "MODULE")
        self.imgDir = self.config.get(self.ID, "IMGDIR")

        self.saveDir = os.path.join(self.basepath, saveDir_)

        self.save = self.args.save if (self.args.save != None) else self.config.getint(self.ID, "SAVE")
        self.epoch = self.args.epoch if self.args.epoch else self.config.getint(self.ID, "EPOCH")

        self.alpha = self.args.alpha if self.args.alpha else self.config.getfloat(self.ID, "ALPHA")

        self.batchSize = self.args.batchsize if self.args.batchsize else self.config.getint(self.ID, "BATCHSIZE")
        self.numWorkers = self.args.numworkers if self.args.numworkers else self.config.getint(self.ID,
                                                                                               "NUMWORKERS")
        self.checkPoint = self.args.checkpoint if self.args.checkpoint else self.config.getint(self.ID,
                                                                                               "CHECKPOINT")

        self.threshold = self.args.threshold if self.args.threshold else self.config.getfloat(self.ID,
                                                                                              "THRESHOLD")
        self.lr = self.args.lr if self.args.lr else self.config.getfloat(self.ID, "LR")
        self.lossid = self.args.lossid if self.args.lossid else self.config.getint(self.ID, "LOSSID")
        self.optimid = self.args.optimid if self.args.optimid else self.config.getint(self.ID, "OPTIMID")

        self.subSaveDir = os.path.join(self.saveDir, f"{self.ID}")
        self.subNetSaveDir = os.path.join(self.subSaveDir, f"{self.trainID}")

        if self.save and self.args.train:
            makedir(self.subSaveDir)
            makedir(self.subNetSaveDir)

        self.transform = transforms.ToTensor()

        varc = {k.lower(): v for k, v in vars(self).items()}
        configitems = self.config.items(self.ID)
        for configkey, configvalue in configitems:
            if configkey.lower() not in varc.keys():
                print(f"{configkey}={configvalue}")
                try:
                    exec(f"self.{configkey}={configvalue}")
                except:
                    exec(f"self.{configkey}='{configvalue}'")

    def _deviceinit(self):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.args.cudanum}")

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
        self.logParam = os.path.join(self.subSaveDir, f"{self.trainID}_param.log")
        self.logTXTFile = os.path.join(self.subSaveDir, f"{self.trainID}.txt")
        # self.logflag = True
        # if os.path.exists(self.logTXTFile):
        #     with open(self.logTXTFile, 'w', encoding='utf-8') as file:

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

    def _title(self):
        if not os.path.exists(self.logTXTFile):
            # torch.save(self.logDict, self.logFile)
            title = f"{self.trainID} Module"
            self.printlog(f"{title:^50}")
            self.printlog(self.net)
        starttime = time.strftime("%b/%d/%Y %H:%M:%S", time.localtime())

        titlestr = "\n" \
                   f"{'*' * 80}\n" \
                   f"*{self.trainID:^78s}*\n" \
                   f"*{' '.join(sys.argv):^78s}*\n" \
                   f"*{starttime:^78s}*\n" \
                   f"{'*' * 80}"
        self.printlog(titlestr)

    def loginfo(self, **kwargs):

        title = f"{self.trainID} Docs"
        self.printlog(f"{title:^50}")
        self.printlog(f"Net_DOC:\n\t{self.net.__doc__.strip()}",newline=True)
        self.printlog(f"Dataset_DOC:\n\t{self.dataset.__doc__.strip()}", newline=True)
        self.printlog(f"Train_DOC:\n\t{self.__doc__.strip()}", newline=True)

        title = f"{self.trainID} Parameters"
        self.printlog(f"{title:^50}")
        varc = vars(self).copy()
        poplist = ['net']

        for var in varc:
            if var.startswith('log') or "__" in var:
                poplist.append(var)
        for popitem in poplist:
            varc.pop(popitem)

        paramdict = {}
        diffdict = {}
        if os.path.exists(self.logParam):
            paramdict = torch.load(self.logParam)

        # print(paramdict['dataset'], varc['dataset'])
        # typelist = [int, float, list, set, dict, str, tuple]
        typelist = [int, float, dict, str]
        for key in varc.keys():
            if type(varc[key]) not in typelist:
                continue
            if key not in paramdict.keys() or paramdict[key]!=varc[key]:
                diffdict[key] = varc[key]

        self.printlog(diffdict, newline=False)

        if self.save and self.args.train:
            torch.save(varc, self.logParam)

    def printlog(self, *args, newline=True):
        if self.save and self.args.train:
            with open(self.logTXTFile, 'a', encoding='utf-8') as file:
                for arg in args:
                    if isinstance(arg, dict):
                        argkeys = sorted(arg)
                        for key in argkeys:
                            print(f"{key:<11s}: ", end="")
                            print(arg[key])
                            print(f"{key:<11s}: ", end="", file=file)
                            print(arg[key], file=file)
                    else:
                        print(arg, end=' ')
                        print(arg, file=file, end=' ')
                if newline:
                    print(file=file)
                    print()
        else:
            for arg in args:
                if isinstance(arg, dict):
                    argkeys = sorted(arg)
                    for key in argkeys:
                        print(f"{key:<11s}: ", end="")
                        print(arg[key])
                else:
                    print(arg,end=' ')
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
            self.netfile_epoch = os.path.join(self.subNetSaveDir, f"{self.ID}_{self.number}_{self.i}.pt")
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

    def tryfunc(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            exc = traceback.format_exc()
            self.printlog(exc)
            # traceback.print_exc()
            # traceback.format_exc()
            # self.printlog(f"error:, {os.path.basename(__file__)}, {e}, {excinfo[2].tb_lineno}")
            # print(sys.exc_info())
            exit(-1)

    def train(self):
        self.tryfunc(self._train)

    def step(self):
        self.tryfunc(self._step)

    def _train(self, *args, **kwargs):
        # print(1 / 0)
        print("train start")
        self.dataloader = data.DataLoader(self.dataset, self.batchSize, shuffle=True, num_workers=self.numWorkers)
        self.dataloaderLen = len(self.dataloader)
        self.i = 0
        self.j = 0
        self.k = 0

        if len(self.logDict["i"]) > 0:
            self.i = self.logDict["i"][-1]
            self.j = self.logDict["j"][-1] + 1
            self.k = self.logDict["k"][-1]
            if self.j >= self.dataloaderLen:
                self.i += 1
                self.j = 0
        self._trainImpl()


    # @abc.abstractmethod
    def _trainImpl(self):

        while (self.i < self.epoch):
            # print(self.net.name)
            self.net.train()
            for dataitem in self.dataloader:
                output = self.get_output(dataitem)

                self.loss = self.get_loss(output)

                self.step()

                if 0 == (self.j + 1) % self.checkPoint or self.j == self.dataloaderLen - 1:
                    self.accuracy = self.get_accuracy(output)
                    result = self.get_result()
                    self.printlog(result, newline=True)
                    self.saveBatch()

                self.j += 1
                self.k += 1
                if self.j == self.dataloaderLen:
                    self.j = 0
                    break

            self.saveEpoch()
            self.test()

            self.i += 1

    def _step(self):
        self._optimizer.zero_grad()
        self.loss.backward()
        self._optimizer.step()

    def test(self):
        self.tryfunc(self._test)

    def _test(self):
        with torch.no_grad():
            self.net.eval()
            self._testImpl()

    def _testImpl(self):

        for dataitem in self.testdataloader:
            output = self.get_output(dataitem)
            self.accuracy = self.get_accuracy(output)
            result = f"test:{self.get_result()}"
            self.printlog(result, newline=True)
            # break


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
        self.testdataset = []
        if 0 == len(self.dataset):
            print("dataset must be initialed")
            exit(-1)
    def _dataloaderinit(self):
        self.dataloader = data.DataLoader(self.dataset, self.batchSize, shuffle=True, num_workers=self.numWorkers)
        self.testdataloader = data.DataLoader(self.testdataset, self.batchSize, shuffle=True, num_workers=self.numWorkers)



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

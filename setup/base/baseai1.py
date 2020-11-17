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

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# FILEPATH = os.path.dirname(__file__)
# CODESPATH = os.path.dirname(FILEPATH)
# BASEPATH = os.path.dirname(CODESPATH)

# if CODESPATH not in sys.path:
#     sys.path.append(CODESPATH)

from tool.utils import makedir

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# BASEPATH = "/home/wen/Projects"
BASEPATH = "D:/Projects"


class BaseNet(nn.Module):
    def __init__(self, name=None, device=DEVICE):
        super(BaseNet, self).__init__()
        self.name = name
        self.device = device
        self.mark = int(time.time())

    def paraminit(self):
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseTrain(metaclass=abc.ABCMeta):
    def __init__(self, netName, net, cfgfile, basepath=BASEPATH):

        self.cfgfile = cfgfile
        self.basepath = basepath
        parser = argparse.ArgumentParser(
            description="base class for network training")
        self.args = self._argparser(parser)

        if self.args.name == None:
            if netName != None:
                self.netName = netName
            else:
                raise ValueError
        else:
            self.netName = self.args.name

        self._deviceinit()
        self._cfginit(cfgfile)

        self._moduleinit(net)

    def _deviceinit(self):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

    def _cfginit(self, cfgfile):
        config = configparser.ConfigParser()
        config.read(cfgfile)

        saveDir_ = config.get(self.netName, "SAVE_DIR")

        self.imgDir = config.get(self.netName, "IMG_DIR")
        self.imgTestDir = config.get(self.netName, "IMGTEST_DIR")
        self.labelDir = config.get(self.netName, "LABEL_DIR")

        self.saveDir = os.path.join(self.basepath, saveDir_)
        # self.imgDir = os.path.join(self.basepath, imgDir_)
        # self.imgTestDir = os.path.join(self.basepath, imgTestDir_)
        # self.labelDir = os.path.join(self.basepath, labelDir_)

        self.save = self.args.save if self.args.save else config.getboolean(self.netName, "SAVE")
        self.epoch = self.args.epoch if self.args.epoch else config.getint(self.netName, "EPOCH")

        self.alpha = self.args.alpha if self.args.alpha else config.getfloat(self.netName, "ALPHA")

        self.batchSize = self.args.batchsize if self.args.batchsize else config.getint(self.netName, "BATCHSIZE")
        self.numWorkers = self.args.numworkers if self.args.numworkers else config.getint(self.netName, "NUMWORKERS")
        self.checkPoint = self.args.checkpoint if self.args.checkpoint else config.getint(self.netName, "CHECKPOINT")

        self.threshold = self.args.threshold if self.args.threshold else config.getfloat(self.netName, "THRESHOLD")
        self.posnum = config.getint(self.netName, "POSNUM")
        self.negnum = config.getint(self.netName, "NEGNUM")
        self.subSaveDir = os.path.join(self.saveDir, self.netName)
        makedir(self.subSaveDir)
        # print(imgTestDir_)

    def _argparser(self, parser):
        parser.add_argument("-n", "--name", type=str,
                            default=None, help="the netfile name to train")
        parser.add_argument("-e", "--epoch", type=int,
                            default=None, help="number of epochs")
        parser.add_argument("-b", "--batchsize", type=int,
                            default=None, help="mini-batch size")
        parser.add_argument("-w", "--numworkers", type=int, default=None,
                            help="number of threads used during batch generation")
        parser.add_argument("-l", "--lr", type=float, default=None,
                            help="learning rate for gradient descent")
        parser.add_argument("-c", "--checkpoint", type=int,
                            default=None, help="print frequency")
        parser.add_argument("-t", "--threshold", type=int, default=None,
                            help="interval between evaluations on validation set")
        parser.add_argument("-a", "--alpha", type=float,
                            default=None, help="ratio of conf and offset loss")
        parser.add_argument("-s", "--save", type=bool,
                            default=None, help="if need save")
        return parser.parse_args()

    def _moduleinit(self, module):
        if module:
            self.netfile = os.path.join(self.subSaveDir, f"{self.netName}.pt")
            self.netfile_backup = os.path.join(
                self.subSaveDir, f"{self.netName}_backup.pt")
            self.net = module().to(self.device)
            if os.path.exists(self.netfile):
                self.net.load_state_dict(torch.load(self.netfile))
                print("load successfully")
        else:
            raise NotImplementedError

    def _log(self, isTest: bool = False, **kwargs):
        if isTest:
            for key, value in kwargs.items():
                self.logDictTest[key].append(value)
            torch.save(self.logDictTest, self.logFileTest)
            torch.save(self.logDictTest, self.logFileTest_backup)
            torch.save(self.logDict, self.logFile_backup)
        else:
            for key, value in kwargs.items():
                self.logDict[key].append(value)
            torch.save(self.logDict, self.logFile)

    def _loginit(self):
        self.logFile = os.path.join(self.subSaveDir, f"{self.netName}.log")
        self.logFileTest = os.path.join(
            self.subSaveDir, f"{self.netName}_test.log")
        self.logFile_backup = os.path.join(
            self.subSaveDir, f"{self.netName}_backup.log")
        self.logFileTest_backup = os.path.join(
            self.subSaveDir, f"{self.netName}_test_backup.log")
        if os.path.exists(self.logFile):
            self.logDict = torch.load(self.logFile)
        else:
            self.logDict = {"i": [], "j": [], "loss": [], "accuracy": []}

        if os.path.exists(self.logFileTest):
            self.logDictTest = torch.load(self.logFileTest)
        else:
            self.logDictTest = {"i": [], "j": [], "accuracy": []}

    def onehot(self, a, cls=2):
        b = torch.zeros(a.size(0), cls).scatter_(-1,
                                                 a.view(-1, 1).long(), 1).to(self.device)
        return b

    def plot(self, *args, isTest=False):
        for item in args:
            plotName = f"plot_{item}.png"  # jpg in linux wrong
            plotPath = os.path.join(self.subSaveDir, plotName)
            if isTest:
                y = np.array(self.logDictTest[item])
            else:
                y = np.array(self.logDict[item])
            plt.clf()
            plt.title(item)
            plt.plot(y)
            plt.savefig(plotPath)

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _datasetinit(self):
        pass

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def get_output(self, *args, **kwargs):
        raise NotImplementedError

    def get_property(self, *args, **kwargs):
        raise NotImplementedError

    def get_accuracy(self, *args, **kwargs):
        raise NotImplementedError

    def get_precision(self, *args, **kwargs):
        raise NotImplementedError

    def get_recall(self, *args, **kwargs):
        raise NotImplementedError

    def detect(self, *args, **kwargs):
        raise NotImplementedError

    def analyze(self, *args, **kwargs):
        raise NotImplementedError

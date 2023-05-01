import os
import os.path as osp
import random
from xml.etree import ElementTree as ET
import cv2
import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


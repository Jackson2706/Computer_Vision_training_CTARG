import os
import os.path as osp
import random
from xml.etree import ElementTree as ET
import cv2
import torch.utils.data as data
import torch
import numpy as np


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

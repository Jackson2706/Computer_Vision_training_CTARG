from typing import Any
from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, \
                            RandomMirror, ToPercentCoords, Resize, SubtractMeans
from make_datapath import make_datapath_list
from extract_inform_annotation import Anno_xml
from lib import *


class DataTransform:
    def __init__(self, input_size, color_mean):
        self.data_tranform = {
            "train": Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(), 
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),

            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_tranform[phase](img, boxes, labels)
    

if __name__ == "__main__":
    classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
            ]
    root_path =  "data/VOCdevkit/VOC2012"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path=root_path)

    anno_xml = Anno_xml(classes)
    idx = 0
    img_file_path =  train_img_list[idx]

    # Read the image
    img = cv2.imread(img_file_path)
    height, width, channels = img.shape

    # Read bbox, label
    annotation_infor = anno_xml(train_annotation_list[idx], width,height)

    #show the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    #tranform image, phase train
    color_mean = [104, 117, 123]
    input_size = 300
    phase = "train"
    transform = DataTransform(input_size=input_size, color_mean=color_mean)
    img_tranformed, boxes, labels = transform(img=img, phase=phase, boxes=annotation_infor[:,:4], labels=annotation_infor[:, 4])
    plt.imshow(cv2.cvtColor(img_tranformed, cv2.COLOR_BGR2RGB))
    plt.show()

    #tranform image, phase val
    phase = "val"
    img_tranformed, boxes, labels = transform(img=img, phase=phase, boxes=annotation_infor[:,:4], labels=annotation_infor[:, 4])
    plt.imshow(cv2.cvtColor(img_tranformed, cv2.COLOR_BGR2RGB))
    plt.show()


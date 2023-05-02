from lib import *
from transform import DataTransform
from extract_inform_annotation import Anno_xml
from make_datapath import make_datapath_list

class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, tranform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.tranform = tranform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)

        return img, gt
    
    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape

        anno_file_path = self.anno_list[index]
        anno_infor = self.anno_xml(anno_file_path, width, height)

        img, boxes, labels = self.tranform(img, self.phase, anno_infor[:, :4], anno_infor[:, 4])

        # convert BGR to RGB
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,1,0)

        #ground truth
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # print(boxes)
        # print(gt)

        return img, gt, height, width
    
def my_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])  #sample[0] = img
        targets.append(torch.FloatTensor(sample[1])) #sample[1] = annotation
    
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


if __name__ == "__main__":
    classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
            ]
    root_path =  "data/VOCdevkit/VOC2012"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path=root_path)

    #tranform image, phase train
    color_mean = [104, 117, 123]
    input_size = 300

    train_dataset = MyDataset(train_img_list, train_annotation_list, phase="train",
                              tranform=DataTransform(input_size=input_size, color_mean=color_mean), anno_xml= Anno_xml(classes=classes))
    val_dataset = MyDataset(val_img_list,val_annotation_list, phase='val', 
                            tranform=DataTransform(input_size=input_size, color_mean=color_mean), anno_xml=Anno_xml(classes=classes))
    
    # print(len(train_dataset))   
    # print(train_dataset.__getitem__(1))
    batch_size = 4
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, collate_fn=my_collate_fn)
    val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_data_loader,
        "val": val_data_loader
    }

    batch_iter = iter(dataloader_dict["val"])
    images, targets = next(batch_iter)
    print(images.size())
    print(len(targets))
    print(targets[0].size())
# data loader
# network : SSD
# loss: multiboxloss
#optimizer
# training, validation

from lib import *
from make_datapath import make_datapath_list
from dataset import MyDataset, my_collate_fn
from transform import DataTransform
from extract_inform_annotation import Anno_xml
from model import SSD
from multiboxloss import MultiBoxLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
cudnn.benchmark = True

#dataloader
root_path = "./data/VOCdevkit/VOC2012"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(root_path=root_path)
classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

color_mean = (104, 117, 123)
input_size = 300

train_dataset = MyDataset(img_list=train_img_list, anno_list=train_anno_list, phase="train",
                            tranform= DataTransform(input_size=input_size, color_mean=color_mean),
                            anno_xml= Anno_xml(classes=classes))

val_dataset = MyDataset(img_list=train_img_list, anno_list=train_anno_list, phase="val",
                            tranform= DataTransform(input_size=input_size, color_mean=color_mean),
                            anno_xml= Anno_xml(classes=classes))

batch_size = 32

train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=True, collate_fn=my_collate_fn)

val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                 shuffle=False, collate_fn=my_collate_fn)

dataloader_dict = {
    "train": train_dataloader,
    "val": val_dataloader
}


#network
cfg = {
    "num_classes": 21, #VOC data include 20 class + 1 background class
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
}

net = SSD(phase="train", cfg=cfg)

vgg_weights = torch.load("./data/weights/vgg16_reducedfc.pth")

net.vgg.load_state_dict(vgg_weights)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# Loss: Multiboxloss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)

#optimizer
optimizer = optimizer.SGD(params=net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

#training & validation
def train(net, dataloader_dict, criterion, optimizer, num_epochs):
    net.to(device)
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print("---"*20)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("---"*20)

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Train)")

            else:
                if (epoch+1)% 10 == 0:
                    net.eval()
                    print("---"*10)
                    print("(evalidation)")
                else:
                    continue

            for images, targets in dataloader_dict[phase]:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                # innit optimizer
                optimizer.zero_grad()

                #forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c


                    if phase == "train":
                        loss.backward()
                    
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)

                        optimizer.step()

                        if (iteration % 10 == 0):
                            t_iter_end = time.time
                            duration = t_iter_end - t_iter_start
                            print(f"Iteration {iteration} || Loss: {loss.item():.4f} || 10iter: {duration:.4f} sec")
                            
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    
                    else:
                        epoch_val_loss += loss.item()
        t_epoch_end = time.time()
        print("---"*20)
        print(f"Epoch {epoch+1} || epoch_train_loss: {epoch_train_loss:.4f} || epoch_val_loss: {epoch_val_loss:.4f}")
        print(f"Duration: {(t_epoch_end-t_epoch_start):.4f} sec")
        t_epoch_start = time.time()

        log_epoch = {
            "Epoch": epoch+1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss    
        }            

        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_log.csv")
        
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0


        if((epoch+1) % 10):
            torch.save(net.state_dict(), "./data/weights/ssd300"+ str(epoch+1) +".pth")

num_epoch = 30

train(net=net,dataloader_dict=dataloader_dict, criterion=criterion, optimizer=optimizer, num_epochs=num_epoch)

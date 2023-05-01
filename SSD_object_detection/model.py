from lib import *

def create_vgg():
    layers = []
    in_channel = 3
    
    cfgs = [64, 64, "M", 128, 128, "M", 
            256, 256, 256, "MC", 512, 512, 512, "M",
            512, 512, 512]
    
    for cfg in cfgs:
        if cfg == 'M':  # floor
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif cfg == "MC": #ceiling
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            layers += [nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1),
                          nn.ReLU(inplace=True)]
            in_channel = cfg

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def extra():
    layers = []
    in_channel = 1024

    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]
    
    layers+= [nn.Conv2d(in_channels=in_channel, out_channels=cfgs[0], kernel_size=1)] 
    layers+= [nn.Conv2d(in_channels=cfgs[0], out_channels=cfgs[1], kernel_size=3, stride=2, padding=1)] 
    layers+= [nn.Conv2d(in_channels=cfgs[1], out_channels=cfgs[2], kernel_size=1)] 
    layers+= [nn.Conv2d(in_channels=cfgs[2], out_channels=cfgs[3], kernel_size=3, stride=2, padding=1)] 
    layers+= [nn.Conv2d(in_channels=cfgs[3], out_channels=cfgs[4], kernel_size=1)] 
    layers+= [nn.Conv2d(in_channels=cfgs[4], out_channels=cfgs[5], kernel_size=3)] 
    layers+= [nn.Conv2d(in_channels=cfgs[5], out_channels=cfgs[6], kernel_size=1)] 
    layers+= [nn.Conv2d(in_channels=cfgs[6], out_channels=cfgs[7], kernel_size=3)]

    return nn.ModuleList(layers)

def create_loc_conf(num_classes = 21, bbox_ratio_num=[4, 6, 6, 6, 4, 4]):
    loc_layer = []
    conf_layer = []
    
    #Source 1
    ## loc
    loc_layer += [nn.Conv2d(in_channels=512, out_channels=bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
    ## conf
    conf_layer += [nn.Conv2d(in_channels=512, out_channels=bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]

    #Source 2
    ##loc
    loc_layer += [nn.Conv2d(in_channels=1024, out_channels=bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
    ## conf
    conf_layer += [nn.Conv2d(in_channels=1024, out_channels=bbox_ratio_num[1]*num_classes, kernel_size=3, padding=1)]

    #Source 3
    ##loc
    loc_layer += [nn.Conv2d(in_channels=512, out_channels=bbox_ratio_num[2]*4, kernel_size=3, padding=1)]
    ##conf
    conf_layer += [nn.Conv2d(in_channels=512, out_channels=bbox_ratio_num[2]* num_classes,kernel_size=3, padding=1)]

    #Source 4
    ##loc 
    loc_layer += [nn.Conv2d(in_channels=256, out_channels=bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    ##conf
    conf_layer += [nn.Conv2d(in_channels=256, out_channels=bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]

    #Source 5
    ##loc 
    loc_layer += [nn.Conv2d(in_channels=256, out_channels=bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    ##conf
    conf_layer += [nn.Conv2d(in_channels=256, out_channels=bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]

    #Source 6
    ##loc 
    loc_layer += [nn.Conv2d(in_channels=256, out_channels=bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    ##conf
    conf_layer += [nn.Conv2d(in_channels=256, out_channels=bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layer), nn.ModuleList(conf_layer)
if __name__ == "__main__":
    vgg = create_vgg()
    loc, conf = create_loc_conf()
    extras = extra()
    print(conf)



        
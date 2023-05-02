from lib import *

cfg = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],
    "min_size": [30, 60, 111, 162, 213, 264],
    "max_size": [60, 111, 162, 213, 264, 315],
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]

}


class DefBox:
    def __init__(self, cfg):
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        
    def create_defbox(self):
        def_box_list = []
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.img_size/self.steps[k]
                cx = (i+0.5)/f_k
                cy = (j+0.5)/f_k

                #small square box    
                s_k = self.min_size[k]/self.img_size
                def_box_list += [cx, cy, s_k, s_k]

                #big square box
                s_k_= sqrt(s_k*(self.max_size[k]/self.img_size))
                def_box_list += [cx, cy, s_k_, s_k_]

                for ar in self.aspect_ratios[k]:
                    def_box_list+= [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    def_box_list += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        
        output = torch.Tensor(def_box_list).view(-1, 4)
        output.clamp_(max=1, min=0)

        return output
    

if __name__ == "__main__":
    defBox = DefBox(cfg=cfg)
    dbox_list = defBox.create_defbox()
    print(pd.DataFrame(dbox_list.numpy()))
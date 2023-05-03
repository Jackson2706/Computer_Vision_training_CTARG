# Jaccard
# Hard negative mining: negative default boxes = 3 * positive default boxes
# Loss in regression: MSE -> F.SmomthL1Loss
# Loss in classification (multi class): Cross Entropy loss

from lib import *
from utils.box_utils import match

class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_threshold = 0.5, neg_pos = 3, device = "cpu"):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.neg_pos = neg_pos
        self.device = device

    def forward(self, predictions, targets):
        loc_data, conf_data, dbox_list = predictions

        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = loc_data.size(2)

        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4)

        for  idx in range(num_batch):
            truths = targets[idx][:, :, :-1].to(self.device)
            labels = targets[idx][:, :, -1].to(self.device)

            dbox = dbox_list.to(self.device)
            variances = [0.1, 0.2]
            match(self.jaccard_threshold, truths, dbox, variances, labels, loc_t, conf_t_label, idx)

        # SmoothL1Loss
        pos_mask = conf_t_label>0
        # loc_dat(num_batch, 8732, 4)
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # positive dbox, loc_data
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)

        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

        # loc_conf : Cross_entropy
        batch_conf = conf_data.view(-1, num_classes)
        loss_conf = F.cross_entropy(batch_conf, conf_t_label, reduction="none")

        # hard negative mining
        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_conf = loss_conf.view(num_batch, -1)

        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # -> idx_rank la thong so de biet do lon loss nam o vi tri bao nhieu

        num_neg = torch.clamp(num_pos* self.neg_pos, max=num_pos)

        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueuze(2).expand_as(conf_data)


        conf_t_pre = conf_data[(pos_idx_mask+ neg_idx_mask).gt(0)].view(-1,num_classes)
        conf_t_label_ = conf_t_label[(pos_idx_mask+neg_idx_mask).gt(0)]

        loss_conf = F.cross_entropy(conf_t_pre, conf_t_label_, reduction="sum" )

        # total_loss = loss_loc + loss_conf

        N = num_pos.sum()
        loss_loc = loss_loc/N
        loss_conf = loss_conf/N

        return loss_loc, loss_conf
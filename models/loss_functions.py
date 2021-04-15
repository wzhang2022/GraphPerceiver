import torch.nn as nn
import torch
from einops import rearrange


class SoftAUC(nn.Module):
    def __init__(self):
        super(SoftAUC, self).__init__()

    def forward(self, logits, labels):
        bs, _ = logits.shape
        assert labels.shape == (bs,)
        num_pos = labels.sum(dtype=int).item()
        if num_pos == 0 or num_pos == bs:
            return logits.mean() * 0
        else:
            pos_indices = torch.nonzero(labels, as_tuple=True)[0]
            neg_indices = torch.nonzero(~labels, as_tuple=True)[0]
            scores = logits[:, 1] - logits[:, 0]
            soft_indicators = scores[pos_indices].unsqueeze(0) - scores[neg_indices].unsqueeze(1)
            soft_auc = torch.sigmoid(soft_indicators).mean()
            return 1-soft_auc


class CombinedCEandAUCLoss(nn.Module):
    def __init__(self, ce_weighted=True, auc_weight=1.0):
        super(CombinedCEandAUCLoss, self).__init__()
        self.auc_weight = auc_weight
        ce_weight = torch.as_tensor([1232 / 32901, 1]) if ce_weighted else None
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean", weight=ce_weight)
        self.softauc_loss = SoftAUC()

    def forward(self, logits, labels):
        return self.ce_loss(logits, labels) + self.auc_weight * self.softauc_loss(logits, labels)


class MultitaskCrossEntropyLoss(nn.Module):
    def __init__(self, num_tasks):
        super(MultitaskCrossEntropyLoss, self).__init__()
        self.num_tasks = num_tasks
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=2)

    def forward(self, logits, labels):
        # logits: (bs, num_tasks)
        output = torch.zeros((logits.shape[0] * self.num_tasks, 2), device=logits.device)
        output[:, 1] = rearrange(logits, "b n -> (b n)")
        labels = torch.nan_to_num(labels, nan=2).long()
        labels = rearrange(labels, "b n -> (b n)")
        return self.ce_loss(output, labels)

import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
from torch.nn import functional as F
import torch.nn.init as init

class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train=0.1):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train
        self.weights = torch.tensor([1.0, 10.0])

    def forward(self, masked_pred_E, nosoft_pred_E, true_E, CEloss=False):
        """ Compute train metrics
        masked_pred_E : tensor -- (bs, n, n, de)
        true_E : tensor -- (bs, n, n, de)
        """
        self.weights = self.weights.to(masked_pred_E.device)
        if CEloss == True:
            masked_pred_E = nosoft_pred_E

        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        target = flat_true_E[:, 1]
        pred = flat_pred_E[:, 1]

        if CEloss == True:
            loss_fn = nn.CrossEntropyLoss(weight=self.weights)
            loss_train = loss_fn(flat_pred_E, flat_true_E)
        else:
            # num_pos = target.sum()
            # num_neg = target.shape[0] - num_pos
            # weight = torch.empty_like(target)
            # weight[target == 1] = num_neg / (num_pos + num_neg)
            # weight[target == 0] = num_pos / (num_pos + num_neg)
            loss_train = F.binary_cross_entropy(pred, target)
            # pos_weight = torch.tensor([10.0]).to(masked_pred_E.device)  # 通常 pos_weight > 1
            # criterion = WeightedFocalLoss(alpha=pos_weight, gamma=2, reduction='mean')
            # loss_train = criterion(pred, target)

        return loss_train

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
        # 使用 Kaiming 初始化方法初始化权重
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        # 初始化偏置项为零
        if m.bias is not None:
            init.zeros_(m.bias)

    for name, param in m.named_parameters():
        if 'weight' in name:
            if 'attention' in name:  # 初始化attention层的权重
                init.xavier_uniform_(param)
            else:  # 初始化其他层的权重
                init.kaiming_uniform_(param)
        elif 'bias' in name:
            init.zeros_(param)


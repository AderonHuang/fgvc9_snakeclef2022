import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class AMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_features
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        delt_costh = torch.zeros(costh.size(), device='cuda').scatter_(1, lb_view, self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        return costh_m_s


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, bias=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # def forward(self, input, label):
    #     cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    #     sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
    #     phi = cosine * self.cos_m - sine * self.sin_m
    #     phi = torch.where(cosine > self.th, phi, cosine - self.mm)
    #     # --------------------------- convert label to one-hot ---------------------------
    #     # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
    #     one_hot = torch.zeros(cosine.size(), device='cuda')
    #     one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    #     # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    #     output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    #     output *= self.s
    #     # print(output)

    #     return output

    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # aa_sum = torch.sum(torch.exp(cosine * self.s), dim=1, keepdim=True) - torch.exp(cosine * self.s)
        # top = torch.exp(phi * self.s)
        # div = top / (top + aa_sum)
        return phi * self.s


class CircleLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=256, m=0.25):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(num_classes, in_features))
        self.s = s
        self.m = m
        self._num_classes = num_classes
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __call__(self, bn_feat, targets):

        sim_mat = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)

        targets = F.one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits


class OhemSoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(OhemSoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target, keep_ratio=0.7):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        loss_sorted, idx = torch.sort(loss, descending=True)
        keep_num = int(keep_ratio * len(loss_sorted))
        loss_keep = loss_sorted[:keep_num]
        return loss_keep.mean()

class OhemLogitsAdjustmentSoftTargetCrossEntropy(nn.Module):
    def __init__(self, classes_dist, tao=1.0):
        super(OhemLogitsAdjustmentSoftTargetCrossEntropy, self).__init__()
        sum_dist = sum(classes_dist)
        self.cls_dist_ts = torch.Tensor([i/sum_dist for i in classes_dist]).cuda()
        self.tao = tao

    def forward(self, x, target, keep_ratio=0.7):
        x = x + torch.log(self.cls_dist_ts ** self.tao + 1e-12)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        loss_sorted, idx = torch.sort(loss, descending=True)
        keep_num = int(keep_ratio * len(loss_sorted))
        loss_keep = loss_sorted[:keep_num]
        return loss_keep.mean()


def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class PolyLoss(_Loss):
    def __init__(self,
                 softmax: bool = True,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes, 
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)


class SeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    class_counts: The list which has number of samples for each class. 
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts: Union[list, np.array], p: float = 0.8):
        super().__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        
        self.eps = 1.0e-6

    def forward(self, logits, targets):
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()

    
class DistibutionAgnosticSeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 for default following the paper.
    """

    def __init__(self, p: float = 0.8):
        super().__init__()
        self.eps = 1.0e-6
        self.p = p
        self.s = None
        self.class_counts = None

    def forward(self, logits, targets):
        if self.class_counts is None:
            self.class_counts = targets.sum(axis=0) + 1 # to prevent devided by zero.
        else:
            self.class_counts += targets.sum(axis=0)

        conditions = self.class_counts[:, None] > self.class_counts[None, :]
        trues = (self.class_counts[None, :] / self.class_counts[:, None]) ** self.p
        falses = torch.ones(len(self.class_counts), len(self.class_counts))
        
        conditions = conditions.cuda()
        trues = trues.cuda()
        falses = falses.cuda()
        self.s = torch.where(conditions, trues, falses)

        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


# end training
import torch
class FocalLoss:
    def __init__(self, alpha_t=0.25, gamma=2):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        super(FocalLoss,self).__init__()
#         assert alpha_t<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
#         print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha_t))
#         self.alpha_t = torch.zeros(100)
#         self.alpha_t[0] += alpha_t
#         self.alpha_t[1:] += (1-alpha_t) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
#         self.alpha_t = torch.tensor(self.alpha_t) 
        self.alpha_t = torch.tensor(alpha_t)
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets)

        elif self.alpha_t is not None and self.gamma == 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha_t)

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha_t is not None and self.gamma != 0:
#             print(self.alpha_t.device)
#             print(outputs.device)
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha_t, reduction='none')
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss
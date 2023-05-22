import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_with_logits(pred, labels, reduction='mean'):
    x = (-labels * F.log_softmax(pred, -1))
    if reduction == 'sum':
        return x.sum()
    elif reduction == 'mean':
        return x.mean()
    else:
        raise NotImplementedError()

def get_ce_loss(cfg, n_classes):
    add_dice = cfg.affordance.add_dice
    class_weights = cfg.affordance.ce_class_weights
    _criterion = nn.BCEWithLogitsLoss if n_classes == 1 else nn.CrossEntropyLoss

    # Only CE
    # -> CE loss w/weight
    if not add_dice and n_classes > 1:
        assert len(class_weights) == n_classes, "Number of weights [%d] != n_classes [%d]" % (
            len(class_weights),
            n_classes,
        )
        affordance_loss = _criterion(weight=torch.tensor(class_weights))
    else:
        # either BCE w or w/o dice
        # or CE w dice
        affordance_loss = _criterion()
    return affordance_loss


# https://github.com/chrisdxie/uois/blob/515c92f63bc83411be21da8449d22660863affbd/src/losses.py#L34
class CosineSimilarityLossWithMask(nn.Module):
    """Compute Cosine Similarity loss"""

    def __init__(self, weighted=False):
        super(CosineSimilarityLossWithMask, self).__init__()
        self.CosineSimilarity = nn.CosineSimilarity(dim=1)
        self.weighted = weighted

    def forward(self, x, target, mask=None):
        """Compute masked cosine similarity loss
        @param x: a [N x C x H x W] torch.FloatTensor of values
        @param target: a [N x C x H x W] torch.FloatTensor of values
        @param mask: a [N x H x W] torch.FloatTensor with values in {0, 1, 2, ..., K+1}, where K is number of objects. {0} are background/table.
                                   Could also be None
        """
        # Shape: [N x H x W]. values are in [0, 1]
        temp = 0.5 * (1 - self.CosineSimilarity(x, target))
        if mask is None:
            # return mean
            return torch.sum(temp) / target.numel()

        # Compute tabletop objects mask
        # Shape: [N x H x W]
        OBJECTS_LABEL = 1
        binary_object_mask = mask.clamp(0, OBJECTS_LABEL).long() == OBJECTS_LABEL

        if torch.sum(binary_object_mask) > 0:
            if self.weighted:
                # Compute pixel weights
                # Shape: [N x H x W]. weighted mean over pixels
                weight_mask = torch.zeros_like(mask).float()
                unique_object_labels = torch.unique(mask)
                unique_object_labels = unique_object_labels[unique_object_labels >= OBJECTS_LABEL]
                for obj in unique_object_labels:
                    num_pixels = torch.sum(mask == obj, dtype=torch.float)
                    # inversely proportional to number of pixels
                    weight_mask[mask == obj] = 1 / num_pixels
            else:
                # mean over observed pixels
                weight_mask = binary_object_mask.float()
            loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask)
        else:
            # print("all gradients are 0...")
            # just 0. all gradients will be 0
            loss = torch.tensor(0.0, dtype=torch.float, device=x.device)

        bg_mask = ~binary_object_mask
        if torch.sum(bg_mask) > 0:
            bg_loss = 0.1 * torch.sum(temp * bg_mask.float()) / torch.sum(bg_mask.float())
        else:
            # just 0
            bg_loss = torch.tensor(0.0, dtype=torch.float, device=x.device)

        return loss + bg_loss


def tresh_tensor(logits, threshold=0.5, keepdim=False):
    if logits.shape[1] == 1 or len(logits.shape) == 3:
        probs = F.sigmoid(logits)
        if threshold == 0.5:
            pred = probs.round().byte()
        else:
            pred = probs > threshold
    else:
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1, keepdim=keepdim).byte()
    return pred, probs


def compute_dice_score(logits, gt, threshold=0.5):
    pred, _ = tresh_tensor(logits, threshold)
    true_positives = ((pred == 1) & (gt == 1)).sum().float()
    false_positives = ((pred == 1) & (gt == 0)).sum().float()
    false_negatives = ((pred == 0) & (gt == 1)).sum().float()
    nomin = 2 * true_positives
    denom = nomin + false_positives + false_negatives
    dice_score = nomin / max(denom, 1e-6)
    return dice_score


def compute_mIoU(logits, gt, threshold=0.5):
    # logits.shape = [BS, 2, img_size, img_size]
    # gt.shape = [BS, 128, 128]
    pred, _ = tresh_tensor(logits, threshold)
    intersection = ((pred == 1) & (gt == 1)).sum().float()
    union = ((pred == 1) | (gt == 1)).sum().float()
    return intersection / max(union, 1e-6)


# https://github.com/kevinzakka/form2fit/blob/099a4ceac0ec60f5fbbad4af591c24f3fff8fa9e/form2fit/code/ml/losses.py#L305
def compute_dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        # B, 2, H, W
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2.0 * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss

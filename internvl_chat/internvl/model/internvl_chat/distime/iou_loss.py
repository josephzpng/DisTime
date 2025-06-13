import torch
import torch.nn as nn

def iou_target_ori(input_offsets, target_offsets, eps=1e-6):
    """ iou for 1d"""
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    # assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    # assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.max(lp, lg)
    rkis = torch.min(rp, rg)

    overlap = torch.maximum(torch.Tensor([0]).to(input_offsets.device), rkis - lkis)

    # iou
    # intsctk = rkis + lkis

    # unionk = (lp + rp) + (lg + rg) - intsctk
    # iouk = intsctk / unionk.clamp(min=eps)
    iouk = overlap / ((rp - lp + rg - lg - overlap).clamp(min=eps))

    return iouk

def iou_loss(input_offsets, target_offsets, eps=1e-6):
    iou_loss = iou_target_ori(input_offsets, target_offsets, eps)
    iou_loss = iou_loss.clamp(min=eps)
    iou_loss = 1 - iou_loss

    return iou_loss


def diou_loss_1d(
        input_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        reduction: str = 'none',
        eps: float = 1e-6,
        weighted=None,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid

    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.maximum(lp, lg)
    rkis = torch.minimum(rp, rg)

    # iou
    overlap = torch.maximum(torch.Tensor([0]).to(input_offsets.device), rkis - lkis)
    iouk = overlap / ((rp - lp) + (rg - lg) - overlap).clamp(min=eps)

    # smallest enclosing box
    lc = torch.minimum(lp, lg)
    rc = torch.maximum(rp, rg)
    len_c = rc - lc

    # offset between centers
    rho_p = 0.5 * (rp + lp) 
    rho_g = 0.5 * (rg + lg)
    rho = rho_p - rho_g

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))
    if weighted is not None:
        loss = loss * weighted

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss



def diou_loss_1d_v2(
        input_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        reduction: str = 'none',
        eps: float = 1e-6,
        weighted=None,
) -> torch.Tensor:
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()

    input_offsets = torch.where(torch.isnan(input_offsets), torch.zeros_like(input_offsets), input_offsets)
    target_offsets = torch.where(torch.isnan(target_offsets), torch.zeros_like(target_offsets), target_offsets)

    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.maximum(lp, lg)
    rkis = torch.minimum(rp, rg)

    # iou
    overlap = torch.maximum(torch.Tensor([0]).to(input_offsets.device), rkis - lkis)
    iouk = overlap / ((rp - lp) + (rg - lg) - overlap).clamp(min=eps)

    # smallest enclosing box
    lc = torch.minimum(lp, lg)
    rc = torch.maximum(rp, rg)
    len_c = rc - lc

    # offset between centers
    rho_p = 0.5 * (rp + lp) 
    rho_g = 0.5 * (rg + lg)
    rho = rho_p - rho_g

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))
    if weighted is not None:
        loss = loss * weighted

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
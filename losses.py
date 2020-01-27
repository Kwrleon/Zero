import torch

def tversky_coeff(pred, target, alpha=0.3, beta=0.7, smooth=1.0, hard=False, reduction="mean"):
    img_flat = pred.view(pred.size(0),-1)
    msk_flat = target.view(target.size(0),-1)
    
    if hard:
        img_flat = torch.round(img_flat)
        
    intersection = (img_flat * msk_flat).sum(dim=-1)
    fps = (img_flat * (1 - msk_flat)).sum(dim=-1)
    fns = ((1 - img_flat) * msk_flat).sum(dim=-1)
    
    denominator = intersection + alpha * fps + beta * fns
    tversky_per_image = (intersection + smooth)/(denominator + smooth)
    if reduction == "mean":
        return tversky_per_image.mean()
    if reduction == "sum":
        return tversky_per_image.sum()
    return tversky_per_image

def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1.0, reduction="mean"):
    """""""""""""""
    https://arxiv.org/pdf/1706.05721.pdf
    α and β control the magnitude of penalties for FPs and FNs, respectively
    α = β = 0.5 => dice coeff
    α = β = 1   => tanimoto coeff
    α + β = 1   => F beta coeff
    """""""""""""""
    tversky_per_image = tversky_coeff(
                pred,
                target,
                alpha     = alpha,
                smooth    = smooth,
                hard      = False,
                reduction = "none")
    loss_per_image = 1.0 - tversky_per_image
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image
    
  
import torch
from torchmetrics import PearsonCorrCoef

def CCCLoss(preds, targets,eps: float = 1e-8):
        # Means
    mu_x = torch.mean(preds)
    mu_y = torch.mean(targets)
    # Centered values
    xm = preds - mu_x
    ym = targets - mu_y
    # Covariance
    cov_xy = torch.mean(xm * ym)
    # Variances
    var_x = torch.mean(xm * xm)
    var_y = torch.mean(ym * ym)

    # Concordance Correlation Coefficient
    numerator = 2.0 * cov_xy
    denominator = var_x + var_y + (mu_x - mu_y) ** 2 + eps
    ccc_val = numerator / denominator

    # CCC loss = 1 − CCC (so minimum is 0 when preds == targets exactly)
    ccc_loss = 1.0 - ccc_val
    return ccc_loss

def InverseCorrelationLoss(preds,targets):
    tags = targets.view(preds.shape)
    
    return 1- torch.corrcoef(torch.stack((preds[...,0],tags[...,0])))[0,1]

"""
Prism losses — 算法改写:
  新增 contrastive_node_loss: 相邻节点embedding对比正则化
  新增 mixup_masked_mae: 对mixup增强后的数据计算加权MAE
  contrastive温度参数随训练步数退火, 促进更紧的embedding聚类
"""
import torch
import numpy as np
from .. import _dbg

_contrast_step = [0]  # 全局步数追踪


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(
        masked_mse(preds=preds, labels=labels,
                   null_val=null_val))


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def contrastive_node_loss(node_embeddings, adj_matrix,
                          tau_init=0.5, tau_min=0.07,
                          anneal_rate=0.001):
    """Prism特有: 相邻节点embedding对比正则化
    对于图中相邻的节点对, 拉近其embedding;
    对于非相邻节点, 推远其embedding。
    温度参数tau随训练步数指数退火, 促进更紧的聚类。

    Args:
        node_embeddings: [N, D] 归一化后的节点embedding
        adj_matrix: [N, N] 邻接矩阵 (非零表示相邻)
        tau_init: 初始温度
        tau_min: 最低温度
        anneal_rate: 退火速率
    """
    _contrast_step[0] += 1
    step = _contrast_step[0]
    # 温度退火: tau = max(tau_min, tau_init * exp(-rate * step))
    tau = max(tau_min, tau_init * np.exp(-anneal_rate * step))
    N = node_embeddings.shape[0]
    # 相似度矩阵
    sim_matrix = torch.mm(node_embeddings,
                          node_embeddings.t()) / tau
    # 构造正样本mask: 相邻节点
    if isinstance(adj_matrix, np.ndarray):
        adj_tensor = torch.from_numpy(adj_matrix).float().to(
            node_embeddings.device)
    else:
        adj_tensor = adj_matrix.float()
    positive_mask = (adj_tensor > 0).float()
    # 去除自环
    diag_mask = 1.0 - torch.eye(N, device=node_embeddings.device)
    positive_mask = positive_mask * diag_mask
    num_positives = positive_mask.sum()
    if num_positives < 1:
        _dbg("contrastive_loss", "no positive pairs, skip",
             "loss")
        return torch.tensor(0.0, device=node_embeddings.device,
                            requires_grad=True)
    # InfoNCE: 对每个节点, 相邻节点为正样本, 其余为负样本
    exp_sim = torch.exp(sim_matrix) * diag_mask
    # 分母: 所有非自身节点的exp(sim)之和
    denominator = exp_sim.sum(dim=1, keepdim=True)
    # log概率
    log_prob = sim_matrix - torch.log(
        denominator + 1e-8)
    # 只取正样本对的均值
    loss = -(log_prob * positive_mask).sum() / (
        num_positives + 1e-8)
    _dbg("contrastive_tau", f"{tau:.4f}", "loss")
    _dbg("contrastive_loss_val", f"{loss.item():.6f}",
         "loss")
    _dbg("contrastive_positives",
         f"{int(num_positives.item())}", "loss")
    return loss


def mixup_masked_mae(preds, labels, lam, null_val=np.nan):
    """Prism特有: Mixup增强后的加权MAE
    当训练使用mixup时, loss也需要按mixup系数加权。
    lam是mixup的插值系数, loss = lam * L(pred, y1) + (1-lam) * L(pred, y2)
    这里简化为: 直接对混合后的label计算MAE, 但用lam调节mask的严格程度。
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = torch.abs(preds - labels)
    # lam加权: 高lam意味着更接近原始样本, 给予更多权重
    weight = lam * torch.ones_like(residual)
    loss = residual * mask * weight
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    _dbg("mixup_lam", f"{lam:.4f}", "loss")
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

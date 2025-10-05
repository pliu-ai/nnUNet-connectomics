import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import tifffile

class DC_and_BCD_Loss(nn.Module):
    def __init__(self, bce_weight=1.0, mse_weight=1.0):
        """
        Combines BCE Loss for the first two channels and MSE Loss for the third channel.

        Args:
            bce_weight (float): Weight for BCE Loss.
            mse_weight (float): Weight for MSE Loss.
        """
        super(DC_and_BCD_Loss, self).__init__()
        self.dc = SoftDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # BCE Loss for binary classification
        self.mse_loss = nn.MSELoss()           # MSE Loss for regression
        self.bce_weight = bce_weight           # Weight for BCE Loss
        self.mse_weight = mse_weight           # Weight for MSE Loss

    def forward(self, net_output, target, heatmap):
        """
        Computes the combined loss.

        Args:
            net_output (Tensor): The network output, shape [batch_size, 3, H, W, D].
            target (Tensor): Ground truth for BCE Loss, shape [batch_size, 2, H, W, D].
            heatmap (Tensor): Heatmap for MSE Loss, shape [batch_size, 1, H, W, D].

        Returns:
            Tensor: Combined loss.
        """
        # Compute BCE Loss for the first two channels
        # apply sigmoid to net_output
        #net_output = torch.sigmoid(net_output)

        bce_loss_channel_0 = self.bce_loss(net_output[:, 0], target[:, 0].float())
        bce_loss_channel_1 = self.bce_loss(net_output[:, 1], target[:, 1].float())
        bce_loss = bce_loss_channel_0 + bce_loss_channel_1

        # Compute MSE Loss for the third channel
        mse_loss = self.mse_loss(net_output[:, 2], heatmap[:, 0])
        # Save predictions to disk
        #print(f"min heatmap: {heatmap[:, 0].min()}, max heatmap: {heatmap[:, 0].max()}")
        #print(f"min net_output: {net_output[:, 2].min()}, max net_output: {net_output[:, 2].max()}")
        # Combine BCE Loss and MSE Loss with respective weights
        total_loss = self.bce_weight * bce_loss + 2 * mse_loss
        #total_loss = 100 * mse_loss
        return total_loss


class DC_and_BCG_Loss(nn.Module):
    def __init__(self, bce_weight=1.0, mse_weight=1.0):
        """
        Combines BCE Loss for the first two channels and MSE Loss for the third channel.

        Args:
            bce_weight (float): Weight for BCE Loss.
            mse_weight (float): Weight for MSE Loss.
        """
        super(DC_and_BCD_Loss, self).__init__()
        self.dc = SoftDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # BCE Loss for binary classification
        self.mse_loss = nn.MSELoss()           # MSE Loss for regression
        self.bce_weight = bce_weight           # Weight for BCE Loss
        self.mse_weight = mse_weight           # Weight for MSE Loss

    def forward(self, net_output, target, heatmap):
        """
        Computes the combined loss.

        Args:
            net_output (Tensor): The network output, shape [batch_size, 3, H, W, D].
            target (Tensor): Ground truth for BCE Loss, shape [batch_size, 2, H, W, D].
            heatmap (Tensor): Heatmap for MSE Loss, shape [batch_size, 1, H, W, D].

        Returns:
            Tensor: Combined loss.
        """
        # Compute BCE Loss for the first two channels
        # apply sigmoid to net_output
        #net_output = torch.sigmoid(net_output)

        bce_loss_channel_0 = self.bce_loss(net_output[:, 0], target[:, 0].float())
        bce_loss_channel_1 = self.bce_loss(net_output[:, 1], target[:, 1].float())
        bce_loss = bce_loss_channel_0 + bce_loss_channel_1

        # Compute MSE Loss for the third channel
        mse_loss = self.mse_loss(torch.sigmoid(net_output[:, 2]), heatmap[:, 0])
        # Save predictions to disk
        #print(f"min heatmap: {heatmap[:, 0].min()}, max heatmap: {heatmap[:, 0].max()}")
        #print(f"min net_output: {net_output[:, 2].min()}, max net_output: {net_output[:, 2].max()}")
        # Combine BCE Loss and MSE Loss with respective weights
        total_loss = self.bce_weight * bce_loss + 2 * mse_loss
        #total_loss = 100 * mse_loss
        return total_loss


class DC_and_CE_BCD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_BCD_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.mse_loss = nn.MSELoss()           # MSE Loss for regression

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, heatmap: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        seg_output = net_output[:, :3]
        heatmap_output = net_output[:, 3:]
        print(f"seg_output shape: {seg_output.shape},heatmap_output shape: {heatmap_output.shape}")
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Compute Segmentation Loss
        dc_loss = self.dc(seg_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(seg_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        seg_loss = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        
        # Compute heatmap loss
        heatmap_loss = 2 * self.mse_loss(heatmap_output, heatmap[:, 0:1])
        
        result = seg_loss + heatmap_loss
        
        return result
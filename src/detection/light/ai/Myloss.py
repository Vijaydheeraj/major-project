import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16


class L_color(nn.Module):
    """
    Computes the color loss by measuring the difference in RGB channel means.

    This loss function calculates the difference in the mean values of the RGB channels of an image.
    The color loss is defined as the Euclidean distance between these means.

    Args:
        None

    Forward Args:
        x (torch.Tensor): The input tensor of shape (B, C, H, W) representing the image.
            B is the batch size, C is the number of channels (RGB = 3), H and W are the height and width.

    Returns:
        torch.Tensor: A scalar tensor representing the computed color loss.
    """
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):
    """
    Computes the spatial consistency loss to maintain image structure.

    This loss function ensures that the spatial features (structure) of the image are preserved
    by applying spatial convolution kernels to both the original and enhanced images.

    Args:
        None

    Forward Args:
        org (torch.Tensor): The original image tensor of shape (B, C, H, W).
        enhance (torch.Tensor): The enhanced image tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: A tensor representing the spatial consistency loss.
    """

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)

        return E


class L_exp(nn.Module):
    """
    Exposure control loss to maintain image brightness at a desired level.

    This loss function calculates the deviation of the image's mean brightness from a target value.
    It controls the exposure of the image to ensure it matches the desired brightness.

    Args:
        patch_size (int): Size of the pooling region used to compute the average brightness.
        mean_val (float): The target mean brightness value.

    Forward Args:
        x (torch.Tensor): The input tensor representing the image with shape (B, C, H, W).

    Returns:
        torch.Tensor: A scalar tensor representing the exposure control loss.
    """

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    """
    Total variation loss to smooth the enhanced image.

    This loss function encourages smoothness in the enhanced image by penalizing large gradients
    along both the horizontal and vertical directions.

    Args:
        TVLoss_weight (float): The weight for the total variation loss.

    Forward Args:
        x (torch.Tensor): The input tensor representing the image with shape (B, C, H, W).

    Returns:
        torch.Tensor: A scalar tensor representing the total variation loss.
    """
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    """
    Computes the loss based on the spatial arrangement of the RGB channels.

    This loss function calculates the spatial distance between the RGB channels in the image.

    Args:
        None

    Forward Args:
        x (torch.Tensor): The input tensor representing the image with shape (B, C, H, W).

    Returns:
        torch.Tensor: A scalar tensor representing the spatial arrangement loss.
    """
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)

        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    """
    Computes perceptual loss using VGG16 to capture high-level image features.

    This loss function uses a pretrained VGG16 model to extract high-level features and compare them 
    between the original and enhanced images, encouraging perceptual similarity.

    Args:
        None

    Forward Args:
        x (torch.Tensor): The input tensor representing the image with shape (B, C, H, W).

    Returns:
        torch.Tensor: The feature map at the last layer used for perceptual loss.
    """
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3

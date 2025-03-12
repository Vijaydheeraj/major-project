import torch
import torch.nn as nn
import torch.nn.functional as f


class enhance_net_nopool(nn.Module):
    """
    EnhanceNet model for low-light image enhancement.
    
    This model aims to enhance low-light images by learning mappings 
    from the low-light input image to its enhanced version.
    """
    def __init__(self):
        """
        Initializes the EnhanceNet model with convolutional layers, 
        non-linear activation functions, and upsampling layers.
        """
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        """
        Forward pass of the EnhanceNet model.
        
        Args:
            x (torch.Tensor): Input tensor representing an image batch. 
                              The tensor should have shape (batch_size, 3, height, width).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - The first tensor is the enhanced image after the first enhancement layer.
                - The second tensor is the fully enhanced image.
                - The third tensor contains the split outputs from the last convolutional layer.
        """
        # Apply convolution layers and ReLU activations
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        # Concatenate feature maps from different layers and process
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        # Output from the final convolution layer and apply tanh for enhancement
        x_r = f.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        # Split the output into multiple feature maps
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # Apply transformations based on the learned feature maps
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        #Concatenate the split outputs for additional information
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        # Return both enhanced images and the split features
        return enhance_image_1, enhance_image, r

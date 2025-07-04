import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import math

num_frequencies = 20  # consider increase, Number of frequency bands for positional encoding

def positional_encoding(points, num_frequencies, include_input=True, log_sampling=True):
    """
    Apply positional encoding to 3D coordinates (NeRF-style).

    Args:
        points: Tensor of shape [N, 3], N is the number of points, and 3 is (x, y, z).
        num_frequencies: Number of frequency bands for encoding.
        include_input: Whether to include the original coordinates in the output.
        log_sampling: Whether to use logarithmic frequency sampling (default: True).

    Returns:
        Tensor of shape [N, 3 * (2 * num_frequencies) + (3 if include_input else 0)].
    """
    # Determine frequency bands
    if log_sampling:
        frequencies = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies).to(points.device)
    else:
        frequencies = torch.linspace(1.0, 2. ** (num_frequencies - 1), num_frequencies).to(points.device)

    # Compute positional encodings
    encoded = []
    if include_input:
        encoded.append(points)  # Include the original coordinates

    for freq in frequencies:
        encoded.append(torch.sin(points * freq * math.pi))  # Scale by π for NeRF-style encoding
        encoded.append(torch.cos(points * freq * math.pi))

    # Concatenate original coordinates and encoded features
    return torch.cat(encoded, dim=-1)

class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim, num_frequencies,include_input=True):
        super(generator, self).__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input  # Include raw input coordinates in PE

        # Update the input dimension based on positional encoding
        self.encoded_dim = self.point_dim * (2 * self.num_frequencies)  # Sin and cos
        if self.include_input:
            self.encoded_dim += self.point_dim  # Add raw input coordinates

        self.input_dim = self.z_dim + self.encoded_dim

        # Input processing block
        self.input_block = nn.Sequential(
            nn.Linear(self.input_dim, self.gf_dim * 8, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        # Residual blocks with consistent dimensions for residual connections
        # Block 1 (gf_dim * 8)
        self.res_block1_1 = nn.Sequential(
            nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        self.res_block1_2 = nn.Sequential(
            nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        # Transition block 1 (gf_dim * 8 → gf_dim * 4)
        self.transition1 = nn.Sequential(
            nn.Linear(self.gf_dim * 8, self.gf_dim * 4, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        # Block 2 (gf_dim * 4)
        self.res_block2_1 = nn.Sequential(
            nn.Linear(self.gf_dim * 4, self.gf_dim * 4, bias=True),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(self.gf_dim * 4, self.gf_dim * 4, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        # Transition block 2 (gf_dim * 4 → gf_dim * 2)
        self.transition2 = nn.Sequential(
            nn.Linear(self.gf_dim * 4, self.gf_dim * 2, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        # Block 3 (gf_dim * 2)
        self.res_block3_1 = nn.Sequential(
            nn.Linear(self.gf_dim * 2, self.gf_dim * 2, bias=True),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(self.gf_dim * 2, self.gf_dim * 2, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        # Transition block 3 (gf_dim * 2 → gf_dim)
        self.transition3 = nn.Sequential(
            nn.Linear(self.gf_dim * 2, self.gf_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        # Final blocks
        self.final_block = nn.Sequential(
            nn.Linear(self.gf_dim, self.gf_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.02)
        )
        
        self.output_layer = nn.Linear(self.gf_dim, 1, bias=True)
        
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for all layers."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.size(1)
                fan_out = module.weight.size(0)
                
                if module in self.output_layer.modules():
                    # Xavier for output
                    std = math.sqrt(2.0 / (fan_in + fan_out))
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    nn.init.constant_(module.bias, 0.0)
                else:
                    # He Initialization for LeakyReLU
                    std = math.sqrt(2.0 / fan_in) * 0.5
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, points, z):
        """
        Args:
            points: Tensor of shape [N, 3], where N can vary
            z: Tensor of shape [1, z_dim] - single z vector for the file
        """
        # Get current batch size (can be different for last chunk)
        num_points = points.size(0)
        
        # Apply positional encoding to points
        points_encoded = positional_encoding(points, self.num_frequencies, include_input=self.include_input)  # [N, encoded_dim]
        
        # Expand z to match current points batch size
        zs = z.expand(num_points, -1)  # [N, z_dim]
        
        # Concatenate encoded points and z
        pointz = torch.cat([points_encoded, zs], dim=-1)  # [N, input_dim]
        
        # Input processing
        x = self.input_block(pointz)  # [N, gf_dim * 8]
        
        # First residual block (same dimension)
        res1 = self.res_block1_1(x)
        x = x + res1  # Residual connection
        
        # Second residual block (same dimension)
        res2 = self.res_block1_2(x)
        x = x + res2  # Residual connection
        
        # Transition from gf_dim * 8 to gf_dim * 4
        x = self.transition1(x)  # [N, gf_dim * 4]
        
        # Third residual block
        res3 = self.res_block2_1(x)
        x = x + res3  # Residual connection
        
        # Transition from gf_dim * 4 to gf_dim * 2
        x = self.transition2(x)  # [N, gf_dim * 2]
        
        # Fourth residual block
        res4 = self.res_block3_1(x)
        x = x + res4  # Residual connection
        
        # Transition from gf_dim * 2 to gf_dim
        x = self.transition3(x)  # [N, gf_dim]
        
        # Final processing
        x = self.final_block(x)  # [N, gf_dim]
        
        # Output layer with tanh
        out = self.output_layer(x)  # [N, 1], constrained to [-1, 1]
        
        return out

class INR_Network(nn.Module):
    def __init__(self, gf_dim, z_dim, point_dim):
        super(INR_Network, self).__init__()
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.num_frequencies = num_frequencies
        self.generator = generator(self.z_dim, self.point_dim, self.gf_dim, self.num_frequencies)
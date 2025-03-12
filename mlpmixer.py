from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os
import numpy as np


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, 
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # Uncomment this line and replace ? with correct values
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_1 = x
        x = self.norm1(x)
        # Transpose x across the sequence and channel dimensions
        x = x.transpose(1, 2)
        x = self.mlp_tokens(x)
        x = x.transpose(1, 2)
        x = x_1 + x
        x_2 = x
        x = self.norm2(x)
        x = self.mlp_channels(x)
        x = x_2 + x
        return x
    

class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu'):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """
        # step1: Go through the patch embedding
        # step 2 Go through the mixer blocks
        # step 3 go through layer norm
        # step 4 Global averaging spatially
        # Classification
        embeddings = self.patchemb(images)
        x = self.blocks(embeddings)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        
        return x

    def visualize(self, logdir):
        print("\n ---------------------------- Visualizing Model Weights ----------------------------- \n")
        
        first_mixer_block = self.blocks[0]  
        token_mixing_layer = first_mixer_block.mlp_tokens  
        W1 = token_mixing_layer.fc1.weight.detach().cpu().numpy()  

        # Get dimensions
        hidden_dim, num_patches = W1.shape 
        print(f'Size of W1: {hidden_dim}, {num_patches}')
        grid_size = int(np.sqrt(num_patches))  # Fixing type issue (torch -> numpy)
        plot_size = int(np.floor(np.sqrt(hidden_dim)))  

        # Normalize for visualization
        W1 = (W1 - W1.min()) / (W1.max() - W1.min())

        # Create subplots dynamically based on hidden_dim
        fig, axes = plt.subplots(plot_size, plot_size, figsize=(10, 10))

        for i in range(plot_size**2):  # Ensure we only plot available units
            row, col = divmod(i, plot_size)
            axes[row, col].imshow(W1[i].reshape(grid_size, grid_size), cmap="coolwarm")
            axes[row, col].axis("off")

        plt.suptitle("Token-Mixing MLP (W1) Weights")
        plt.tight_layout()

        # Ensure logdir exists
        os.makedirs(logdir, exist_ok=True)

        # Generate unique filename
        file_path = os.path.join(logdir, "mlpmixer_weights.png")
        i = 1
        while os.path.exists(file_path):
            file_path = os.path.join(logdir, f"mlpmixer_weights_{i}.png")
            i += 1

        # Save plot
        plt.savefig(file_path)
        plt.close()
        
        print(f"Visualization saved at: {file_path}")

        
        
            
        
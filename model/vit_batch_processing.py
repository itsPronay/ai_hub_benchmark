import torch
import torch.nn as nn
from .vit import ViT


class ViTbatchProcessing(nn.Module):
    """
    Batch processing wrapper for ViT model.
    Processes large datasets by splitting them into manageable batches.
    Accepts 5D input (total_samples, 1, bands, H, W) and processes in batches.
    """
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, 
                 mlp_dim, dropout, emb_dropout, mode, total_samples, batch_size):
        super().__init__()
        
        # Create ViT model internally
        self.vit = ViT(
            image_size=image_size,
            near_band=near_band,
            num_patches=num_patches,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mode=mode
        )
        
        # Batch processing parameters
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.num_batches = (total_samples + batch_size - 1) // batch_size
        
    def _reshape_5d_to_3d(self, x_5d):
        """
        Reshape 5D tensor to 3D format for ViT processing.
        
        Args:
            x_5d: Tensor of shape (batch, 1, bands, H, W)
        Returns:
            Tensor of shape (batch, bands, H*W)
        """
        b, c, bands, h, w = x_5d.shape
        x_4d = x_5d.squeeze(1)  # (batch, bands, H, W)
        x_3d = x_4d.reshape(b, bands, h * w)  # (batch, bands, H*W)
        return x_3d
    
    def _process_batch(self, batch_data):
        """
        Process a single batch through the ViT model.
        
        Args:
            batch_data: Tensor of shape (batch_size, 1, bands, H, W)
        Returns:
            Predictions of shape (batch_size, num_classes)
        """
        # Reshape from 5D to 3D
        batch_3d = self._reshape_5d_to_3d(batch_data)
        # Forward through ViT
        return self.vit(batch_3d)
        
    def forward(self, x):
        """
        Process full dataset in batches.
        
        Args:
            x: Input tensor of shape (total_samples, 1, bands, H, W)
        Returns:
            predictions: Tensor of shape (total_samples, num_classes)
        """
        batch_results = []
        
        # Process dataset in batches
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.total_samples)
            
            # Extract current batch
            batch_data = x[start_idx:end_idx]
            
            # Process batch
            batch_output = self._process_batch(batch_data)
            batch_results.append(batch_output)
        
        # Concatenate all batch results
        return torch.cat(batch_results, dim=0)

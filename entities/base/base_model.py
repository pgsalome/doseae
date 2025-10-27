"""
Base model class for all anatomical entities.
Provides common interface for models across different anatomical regions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for entity-specific models.
    All anatomical entity models should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model components
        self.encoder = None
        self.decoder = None
        self.attention_module = None
        self.auxiliary_heads = nn.ModuleDict()
        
        # Initialize model components
        self._build_model()
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        if self.encoder is None:
            raise NotImplementedError("Encoder not implemented")
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent representation
            
        Returns:
            Decoded output
        """
        if self.decoder is None:
            raise NotImplementedError("Decoder not implemented")
        return self.decoder(z)
    
    def apply_attention(self, features: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention mechanism to features.
        
        Args:
            features: Input features
            attention_mask: Optional attention mask
            
        Returns:
            Attended features
        """
        if self.attention_module is None:
            return features
        
        if attention_mask is not None:
            return self.attention_module(features, attention_mask)
        else:
            return self.attention_module(features)
    
    def compute_auxiliary_outputs(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary task outputs.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary containing auxiliary outputs
        """
        auxiliary_outputs = {}
        for name, head in self.auxiliary_heads.items():
            auxiliary_outputs[name] = head(features)
        return auxiliary_outputs
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from the attention module.
        
        Returns:
            Attention weights tensor or None
        """
        if hasattr(self.attention_module, 'attention_weights'):
            return self.attention_module.attention_weights
        return None
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the model.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Dictionary containing loss values
        """
        losses = {}
        
        # Main reconstruction loss
        if 'reconstruction' in outputs and 'target' in targets:
            losses['reconstruction'] = F.mse_loss(outputs['reconstruction'], targets['target'])
        
        # Auxiliary losses
        for name, output in outputs.items():
            if name.startswith('aux_') and name[4:] in targets:
                target_name = name[4:]
                if target_name == 'dose_level':
                    losses[name] = F.cross_entropy(output, targets[target_name])
                else:
                    losses[name] = F.mse_loss(output, targets[target_name])
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary information.
        
        Returns:
            Dictionary containing model summary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': self.__class__.__name__,
            'config': self.config
        }


class BaseAttentionModule(nn.Module, ABC):
    """
    Abstract base class for attention modules.
    """
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        """
        Initialize the attention module.
        
        Args:
            input_dim: Input feature dimension
            config: Configuration dictionary
        """
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.attention_weights = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the attention module.
        
        Args:
            x: Input features
            attention_mask: Optional attention mask
            
        Returns:
            Attended features
        """
        pass


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for encoders.
    """
    
    def __init__(self, input_channels: int, latent_dim: int, config: Dict[str, Any]):
        """
        Initialize the encoder.
        
        Args:
            input_channels: Number of input channels
            latent_dim: Latent dimension
            config: Configuration dictionary
        """
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded features
        """
        pass


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for decoders.
    """
    
    def __init__(self, latent_dim: int, output_channels: int, config: Dict[str, Any]):
        """
        Initialize the decoder.
        
        Args:
            latent_dim: Latent dimension
            output_channels: Number of output channels
            config: Configuration dictionary
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.config = config
    
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent representation
            
        Returns:
            Decoded output
        """
        pass

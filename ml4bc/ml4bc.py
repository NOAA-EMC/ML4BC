from typing import Optional
import torch
from huggingface_hub import PyTorchModelHubMixin
from encoder import Encoder
from decoder import Decoder
from processor import Processor

class GraphWeatherModel(torch.nn.Module, PyTorchModelHubMixin):
    """
    Weather Model utilizing Graph Neural Networks.

    This model predicts weather variables using a graph neural network architecture,
    transforming data from latitude/longitude grids to abstract latent features
    on an icosahedron grid.
    """

    def __init__(
        self,
        lat_lon_coords: list,
        resolution: int = 2,
        input_feature_dim: int = 78,
        auxiliary_feature_dim: int = 0,
        output_feature_dim: Optional[int] = None,
        node_hidden_dim: int = 256,
        edge_hidden_dim: int = 256,
        num_processor_blocks: int = 9,
        hidden_node_processor_dim: int = 256,
        hidden_edge_processor_dim: int = 256,
        hidden_node_processor_layers: int = 2,
        hidden_edge_processor_layers: int = 2,
        hidden_decoder_dim: int = 128,
        decoder_layers: int = 2,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
    ):
        """
        Initialize the Weather Forecasting Model.

        Args:
            ... (Same arguments as provided in the original code)
        """
        super().__init__()
        self.input_feature_dim = input_feature_dim
        if output_feature_dim is None:
            output_feature_dim = self.input_feature_dim

        # Encoder for transforming lat/lon data into an icosahedron grid
        self.encoder = Encoder(
            lat_lons=lat_lon_coords,
            resolution=resolution,
            input_dim=input_feature_dim + auxiliary_feature_dim,
            output_dim=node_hidden_dim,
            output_edge_dim=edge_hidden_dim,
            hidden_dim_processor_edge=hidden_edge_processor_dim,
            hidden_layers_processor_node=hidden_node_processor_layers,
            hidden_dim_processor_node=hidden_node_processor_dim,
            hidden_layers_processor_edge=hidden_edge_processor_layers,
            mlp_norm_type=norm_type,
            use_checkpointing=use_checkpointing,
        )

        # Processor for message passing and graph transformations
        self.processor = Processor(
            input_dim=node_hidden_dim,
            edge_dim=edge_hidden_dim,
            num_blocks=num_processor_blocks,
            hidden_dim_processor_edge=hidden_edge_processor_dim,
            hidden_layers_processor_node=hidden_node_processor_layers,
            hidden_dim_processor_node=hidden_node_processor_dim,
            hidden_layers_processor_edge=hidden_edge_processor_layers,
            mlp_norm_type=norm_type,
        )

        # Decoder for generating the forecast
        self.decoder = Decoder(
            lat_lons=lat_lon_coords,
            resolution=resolution,
            input_dim=node_hidden_dim,
            output_dim=output_feature_dim,
            output_edge_dim=edge_hidden_dim,
            hidden_dim_processor_edge=hidden_edge_processor_dim,
            hidden_layers_processor_node=hidden_node_processor_layers,
            hidden_dim_processor_node=hidden_node_processor_dim,
            hidden_layers_processor_edge=hidden_edge_processor_layers,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_decoder_dim,
            hidden_layers_decoder=decoder_layers,
            use_checkpointing=use_checkpointing,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the forecasted weather variables.

        Args:
            features: The input features, aligned with the order of lat_lon_coords

        Returns:
            The forecasted weather variables
        """
        x, edge_idx, edge_attr = self.encoder(features)
        x = self.processor(x, edge_idx, edge_attr)
        x = self.decoder(x, features[..., : self.input_feature_dim])
        return x

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Make predictions based on the input features."""
        return self.forward(features)

    def load_pretrained(self, path: str) -> None:
        """Load pre-trained weights or model from a specified path."""
        self.load_state_dict(torch.load(path))

    def save_model(self, path: str) -> None:
        """Save the model's weights or state to the specified path."""
        torch.save(self.state_dict(), path)

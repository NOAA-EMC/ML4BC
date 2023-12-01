import torch
import h3
import numpy as np
from torch_geometric.data import Data
import einops
from gnn import MLP, GraphProcessor

# Decoder graph module
class Decoder(torch.nn.Module):
    def __init__(
        self,
        lat_lons,
        resolution: int = 2,
        input_dim: int = 256,
        output_dim: int = 78,
        output_edge_dim: int = 256,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        mlp_norm_type: str = "LayerNorm",
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        use_checkpointing: bool = False,
    ):
        """
        Decoder from latent graph to lat/lon graph

        Args:
            lat_lons: List of (lat,lon) points
            resolution: H3 resolution level
            input_dim: Input node dimension
            output_dim: Output node dimension
            output_edge_dim: Edge dimension
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder: Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Whether to use gradient checkpointing or not
        """
        super().__init__()

        # Initialize parameters and attributes
        self.use_checkpointing = use_checkpointing
        self.num_latlons = len(lat_lons)
        self.base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
        self.num_h3 = len(self.base_h3_grid)
        self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        self.h3_to_index = {}
        h_index = len(self.base_h3_grid)
        for h in self.base_h3_grid:
            if h not in self.h3_to_index:
                h_index -= 1
                self.h3_to_index[h] = h_index
        self.h3_mapping = {}
        for h, value in enumerate(self.h3_grid):
            self.h3_mapping[h + self.num_h3] = value

        # Build the default graph structure
        nodes = torch.zeros(
            (len(lat_lons) + h3.num_hexagons(resolution), input_dim), dtype=torch.float
        )
        self.latlon_nodes = torch.zeros((len(lat_lons), input_dim), dtype=torch.float)
        edge_sources = []
        edge_targets = []
        self.h3_to_lat_distances = []
        for node_index, h_node in enumerate(self.h3_grid):
            h_points = h3.k_ring(self.h3_mapping[node_index + self.num_h3], 1)
            for h in h_points:
                distance = h3.point_dist(lat_lons[node_index], h3.h3_to_geo(h), unit="rads")
                self.h3_to_lat_distances.append([np.sin(distance), np.cos(distance)])
                edge_sources.append(self.h3_to_index[h])
                edge_targets.append(node_index + self.num_h3)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        self.h3_to_lat_distances = torch.tensor(self.h3_to_lat_distances, dtype=torch.float)

        # Initialize graph data structure
        self.graph = Data(x=nodes, edge_index=edge_index, edge_attr=self.h3_to_lat_distances)

        # Initialize MLPs and processors
        self.edge_encoder = MLP(
            2, output_edge_dim, hidden_dim_processor_edge, 2, mlp_norm_type, self.use_checkpointing
        )
        self.graph_processor = GraphProcessor(
            mp_iterations=1,
            in_dim_node=input_dim,
            in_dim_edge=output_edge_dim,
            hidden_dim_node=hidden_dim_processor_node,
            hidden_dim_edge=hidden_dim_processor_edge,
            hidden_layers_node=hidden_layers_processor_node,
            hidden_layers_edge=hidden_layers_processor_edge,
            norm_type=mlp_norm_type,
        )
        self.node_decoder = MLP(
            input_dim,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing,
        )

    def forward(
        self, processor_features: torch.Tensor, start_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            processor_features: Processed features in shape [B*Nodes, Features]
            start_features: Original input features to the encoder, with shape [B, Nodes, Features]

        Returns:
            Updated features for model
        """
        # Move the graph to device
        out = self.graph.to(processor_features.device)

        # Update edge attributes
        edge_attr = self.edge_encoder(self.graph.edge_attr)
        edge_attr = einops.repeat(edge_attr, "e f -> (repeat e) f", repeat=start_features.shape[0])

        # Construct the edge index
        edge_index = torch.cat(
            [
                self.graph.edge_index + i * torch.max(self.graph.edge_index) + i
                for i in range(start_features.shape[0])
            ],
            dim=1,
        )

        # Move latlon_nodes to device
        self.latlon_nodes = self.latlon_nodes.to(processor_features.device)

        # Rearrange and concatenate features
        features = einops.rearrange(processor_features, "(b n) f -> b n f", b=start_features.shape[0])
        features = torch.cat(
            [features, einops.repeat(self.latlon_nodes, "n f -> b n f", b=start_features.shape[0])],
            dim=1
        )
        features = einops.rearrange(features, "b n f -> (b n) f")

        # Perform message passing and decoding
        out, _ = self.graph_processor(features, edge_index, edge_attr)
        out = self.node_decoder(out)
        out = einops.rearrange(out, "(b n) f -> b n f", b=start_features.shape[0])

        # Split the output and add to the start features for residual connection
        _, out = torch.split(out, [self.num_h3, self.num_latlons], dim=1)
        return out + start_features

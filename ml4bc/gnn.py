from typing import Optional, Tuple
import torch
from torch import cat, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum

# Multi-Layer Perceptron (MLP) for graph processing
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: Optional[str] = "LayerNorm",
        use_checkpointing: bool = False,
    ):
        super(MLP, self).__init__()
        self.use_checkpointing = use_checkpointing

        # Building the MLP layers
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        # Adding normalization if specified
        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using gradient checkpointing if specified
        if self.use_checkpointing:
            out = checkpoint(self.model, x, use_reentrant=False)
        else:
            out = self.model(x)
        return out

# Edge Processor class
class EdgeProcessor(nn.Module):
    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
    ):
        super(EdgeProcessor, self).__init__()
        self.edge_mlp = MLP(
            2 * in_dim_node + in_dim_edge, in_dim_edge, hidden_dim, hidden_layers, norm_type
        )

    def forward(
        self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor, u=None, batch=None
    ) -> torch.Tensor:
        # Concatenating source node, destination node, and edge embeddings
        out = cat([src, dest, edge_attr], -1)
        out = self.edge_mlp(out)
        out += edge_attr  # Applying residual connection
        return out

# Node Processor class
class NodeProcessor(nn.Module):
    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
    ):
        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u=None, batch=None
    ) -> torch.Tensor:
        row, col = edge_index
        scatter_dim = 0
        output_size = x.size(scatter_dim)
        # Aggregating edge message by target
        out = scatter_sum(edge_attr, col, dim=scatter_dim, dim_size=output_size)
        out = cat([x, out], dim=-1)
        out = self.node_mlp(out)
        out += x  # Applying residual connection
        return out

# Function to build the Graph Net Block using MetaLayer
def build_graph_processor_block(
    in_dim_node: int = 128,
    in_dim_edge: int = 128,
    hidden_dim_node: int = 128,
    hidden_dim_edge: int = 128,
    hidden_layers_node: int = 2,
    hidden_layers_edge: int = 2,
    norm_type: str = "LayerNorm",
) -> torch.nn.Module:
    return MetaLayer(
        edge_model=EdgeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type
        ),
        node_model=NodeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type
        ),
    )

# Graph Processor class
class GraphProcessor(nn.Module):
    def __init__(
        self,
        mp_iterations: int = 15,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim_node: int = 128,
        hidden_dim_edge: int = 128,
        hidden_layers_node: int = 2,
        hidden_layers_edge: int = 2,
        norm_type: str = "LayerNorm",
    ):
        super(GraphProcessor, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(
                build_graph_processor_block(
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    norm_type,
                )
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Iterating through the blocks for message passing
        for block in self.blocks:
            x, edge_attr, _ = block(x, edge_index, edge_attr)

        return x, edge_attr

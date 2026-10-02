"""Flow Matching model for plan-level normalized floorplan generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_utils import (
    AU_BIT_DIM,
    AU_COORD_DIM,
    build_edge_au_features,
)


CLIP_DIM = 768
TOPO_FEAT_DIM = 61
NODE_INPUT_DIM = TOPO_FEAT_DIM
AREA_FEATURE_DIM = 3
CORNER_GEOMETRY_DIM = 9
MAX_CORNERS_PER_ROOM = 28


class InputProjection(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.SiLU(),
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
        )

    def forward(self, inputs):
        return self.net(inputs)


class FeedForward(nn.Module):
    def __init__(self, dimension, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)


class MaskedSelfAttention(nn.Module):
    def __init__(
        self,
        dimension,
        num_heads=8,
        head_dimension=64,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.query_projection = nn.Linear(
            dimension,
            num_heads * head_dimension,
        )
        self.key_projection = nn.Linear(
            dimension,
            num_heads * head_dimension,
        )
        self.value_projection = nn.Linear(
            dimension,
            num_heads * head_dimension,
        )
        self.output_projection = nn.Linear(
            num_heads * head_dimension,
            dimension,
        )
        self.query_norm = nn.RMSNorm(head_dimension)
        self.key_norm = nn.RMSNorm(head_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.query_projection(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        )
        key = self.key_projection(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        )
        value = self.value_projection(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        )
        query = self.query_norm(query).transpose(1, 2)
        key = self.key_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)

        if attention_mask is not None:
            diagonal = torch.eye(
                sequence_length,
                dtype=torch.bool,
                device=attention_mask.device,
            )
            attention_mask = attention_mask | diagonal
            mask = attention_mask[:, None]
            float_mask = torch.zeros(
                mask.shape,
                dtype=query.dtype,
                device=query.device,
            ).masked_fill(~mask, float("-inf"))
        else:
            float_mask = None

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=float_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.num_heads * self.head_dimension,
        )
        return self.dropout(self.output_projection(output))


class AdaLayerNorm(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            dimension,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.projection = nn.Linear(dimension, 6 * dimension)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states, condition):
        projection = self.projection(
            F.silu(condition)
        )
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = projection.chunk(6, dim=-1)
        shift_attention = shift_attention[:, None]
        scale_attention = scale_attention[:, None]
        gate_attention = gate_attention[:, None]
        shift_mlp = shift_mlp[:, None]
        scale_mlp = scale_mlp[:, None]
        gate_mlp = gate_mlp[:, None]
        normalized = self.layer_norm(hidden_states)
        conditioned = normalized * (1 + scale_attention) + shift_attention
        return (
            conditioned,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, output_dimension, frequencies=256):
        super().__init__()
        self.frequencies = frequencies
        self.projection = InputProjection(
            frequencies,
            output_dimension,
        )

    def forward(self, sigma):
        half = self.frequencies // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half,
                device=sigma.device,
                dtype=torch.float32,
            )
            / half
        )
        angle = sigma.float()[:, None] * frequencies
        embedding = torch.cat(
            [torch.sin(angle), torch.cos(angle)],
            dim=-1,
        )
        return self.projection(embedding)


class CornerPositionalEncoding(nn.Module):
    def __init__(self, dimension, max_length=28, dropout=0.0):
        super().__init__()
        self.max_length = max_length
        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, corner_ids):
        return self.dropout(
            hidden_states
            + self.encoding[
                corner_ids.clamp(min=0, max=self.max_length - 1)
            ]
        )


class NodeTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension,
        num_heads,
        head_dimension,
        dropout,
    ):
        super().__init__()
        self.connection_attention = MaskedSelfAttention(
            dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.global_attention = MaskedSelfAttention(
            dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.connection_norm = nn.LayerNorm(dimension)
        self.global_norm = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(dimension)

    def forward(
        self,
        hidden_states,
        connection_mask,
        global_mask,
    ):
        hidden_states = hidden_states + self.connection_attention(
            self.connection_norm(hidden_states),
            attention_mask=connection_mask,
        )
        hidden_states = hidden_states + self.global_attention(
            self.global_norm(hidden_states),
            attention_mask=global_mask,
        )
        hidden_states = hidden_states + self.feed_forward(
            self.ffn_norm(hidden_states)
        )
        return hidden_states


class CornerTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension,
        num_heads,
        head_dimension,
        dropout,
    ):
        super().__init__()
        self.ada_norm = AdaLayerNorm(dimension)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.room_attention = MaskedSelfAttention(
            dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.connection_attention = MaskedSelfAttention(
            dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.global_attention = MaskedSelfAttention(
            dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.feed_forward = FeedForward(dimension, dropout=dropout)

    def forward(
        self,
        hidden_states,
        time_clip_condition,
        room_condition,
        room_mask,
        connection_mask,
        global_mask,
    ):
        conditioned_input = hidden_states + room_condition
        (
            conditioned,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_norm(conditioned_input, time_clip_condition)
        hidden_states = hidden_states + gate_attention * self.room_attention(
            conditioned,
            attention_mask=room_mask,
        )

        conditioned_input = hidden_states + room_condition
        (
            conditioned,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_norm(conditioned_input, time_clip_condition)
        hidden_states = hidden_states + (
            gate_attention
            * self.connection_attention(
                conditioned,
                attention_mask=connection_mask,
            )
        )

        conditioned_input = hidden_states + room_condition
        (
            conditioned,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_norm(conditioned_input, time_clip_condition)
        hidden_states = hidden_states + gate_attention * self.global_attention(
            conditioned,
            attention_mask=global_mask,
        )

        ffn_input = self.ffn_norm(hidden_states)
        ffn_input = ffn_input * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + (
            gate_mlp * self.feed_forward(ffn_input)
        )
        return hidden_states


def quantize_au_bits(au_features):
    clamped = au_features.clamp(-1.0, 1.0)
    scaled = (
        (clamped / 2.0 + 0.5) * 256.0
    ).round().int().clamp(0, 255)
    bits = 2 ** torch.arange(
        7,
        -1,
        -1,
        device=scaled.device,
    )
    binary = (
        scaled.unsqueeze(-1).bitwise_and(bits).ne(0)
    ).to(au_features.dtype)
    return (
        binary.reshape(
            scaled.shape[0],
            scaled.shape[1],
            AU_BIT_DIM,
        )
        * 2.0
        - 1.0
    )


def quantize_corner_bits(xy):
    clamped = xy.clamp(-1.0, 1.0)
    scaled = (
        (clamped / 2.0 + 0.5) * 256.0
    ).round().int().clamp(0, 255)
    bits = 2 ** torch.arange(
        7,
        -1,
        -1,
        device=scaled.device,
    )
    binary = (
        scaled.unsqueeze(-1).bitwise_and(bits).ne(0)
    ).to(xy.dtype)
    return (
        binary.reshape(
            scaled.shape[0],
            scaled.shape[1],
            16,
        )
        * 2.0
        - 1.0
    )


def discrete_logits_to_coords(
    logits,
    bits=8,
):
    binary = (logits > 0.0).to(logits.dtype)
    binary_xy = binary.reshape(
        binary.shape[0],
        binary.shape[1],
        2,
        bits,
    )
    weights = 2 ** torch.arange(
        bits - 1,
        -1,
        -1,
        device=logits.device,
        dtype=logits.dtype,
    )
    decoded = (binary_xy * weights).sum(dim=-1)
    return decoded / 256.0 * 2.0 - 1.0


def soft_discrete_logits_to_coords(
    logits,
    bits=8,
):
    probabilities = ((logits.clamp(-1.0, 1.0) + 1.0) / 2.0).reshape(
        logits.shape[0],
        logits.shape[1],
        2,
        bits,
    )
    weights = 2 ** torch.arange(
        bits - 1,
        -1,
        -1,
        device=logits.device,
        dtype=logits.dtype,
    )
    decoded = (probabilities * weights).sum(dim=-1)
    return decoded / 256.0 * 2.0 - 1.0


def flow_x0_with_discrete_refinement(
    corner_features,
    velocity,
    sigma,
    discrete_logits,
    threshold=0.06,
    soft=True,
    hard_at_zero=True,
):
    """Use the bit head as a low-sigma x0 refiner without changing x_t."""
    continuous_x0 = (
        corner_features[:, :, :2]
        - sigma[:, None, None] * velocity
    )
    hard_x0 = discrete_logits_to_coords(discrete_logits)
    if soft:
        soft_x0 = soft_discrete_logits_to_coords(discrete_logits)
        if hard_at_zero:
            at_zero = (sigma <= 1e-8).view(-1, 1, 1)
            discrete_x0 = torch.where(
                at_zero,
                hard_x0,
                soft_x0,
            )
        else:
            discrete_x0 = soft_x0
    else:
        discrete_x0 = hard_x0
    use_discrete = (sigma < threshold).view(-1, 1, 1)
    return torch.where(
        use_discrete,
        discrete_x0,
        continuous_x0,
    )


def velocity_from_x0(
    corner_features,
    x0_prediction,
    sigma,
    eps=1e-5,
):
    """Convert an x0 estimate at sigma into the corresponding FM velocity."""
    safe_sigma = sigma.clamp(min=eps).view(-1, 1, 1)
    velocity = (
        corner_features[:, :, :2] - x0_prediction
    ) / safe_sigma
    active = (sigma > eps).view(-1, 1, 1)
    return torch.where(
        active,
        velocity,
        torch.zeros_like(velocity),
    )


class DiscreteTransformerBlock(nn.Module):
    """Masked room -> connection -> global encoder for the bit head."""

    def __init__(
        self,
        dimension,
        num_heads,
        head_dimension,
        dropout,
    ):
        super().__init__()
        self.room_norm = nn.LayerNorm(dimension)
        self.connection_norm = nn.LayerNorm(dimension)
        self.global_norm = nn.LayerNorm(dimension)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.room_attention = MaskedSelfAttention(
            dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.connection_attention = MaskedSelfAttention(
            dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.global_attention = MaskedSelfAttention(
            dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.feed_forward = FeedForward(dimension, dropout=dropout)

    def forward(
        self,
        hidden_states,
        room_mask,
        connection_mask,
        global_mask,
    ):
        hidden_states = hidden_states + self.room_attention(
            self.room_norm(hidden_states),
            attention_mask=room_mask,
        )
        hidden_states = hidden_states + self.connection_attention(
            self.connection_norm(hidden_states),
            attention_mask=connection_mask,
        )
        hidden_states = hidden_states + self.global_attention(
            self.global_norm(hidden_states),
            attention_mask=global_mask,
        )
        hidden_states = hidden_states + self.feed_forward(
            self.ffn_norm(hidden_states)
        )
        return hidden_states


class DiscreteCoordinateRefinementHead(nn.Module):
    def __init__(
        self,
        dimension,
        num_heads,
        head_dimension,
        num_layers=2,
        dropout=0.0,
        detach_from_velocity=True,
        base_logits_weight=0.0,
    ):
        super().__init__()
        self.detach_from_velocity = bool(detach_from_velocity)
        self.base_logits_weight = float(base_logits_weight)
        if not 0.0 <= self.base_logits_weight <= 1.0:
            raise ValueError("base_logits_weight must be in [0, 1]")
        self.input_projection = nn.Linear(
            AU_COORD_DIM + AU_BIT_DIM,
            dimension,
        )
        self.layers = nn.ModuleList(
            [
                DiscreteTransformerBlock(
                    dimension,
                    num_heads=num_heads,
                    head_dimension=head_dimension,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.logit_projection = nn.Linear(dimension, 16)

    def forward(
        self,
        room_condition,
        au_features,
        base_logits,
        room_mask,
        connection_mask,
        global_mask,
    ):
        if self.detach_from_velocity:
            room_condition = room_condition.detach()
            au_features = au_features.detach()
            base_logits = base_logits.detach()
        discrete_input = torch.cat(
            [au_features, quantize_au_bits(au_features.detach())],
            dim=-1,
        )
        hidden = (
            F.silu(self.input_projection(discrete_input))
            + room_condition
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                room_mask,
                connection_mask,
                global_mask,
            )
        direct_logits = self.logit_projection(hidden)
        return (
            self.base_logits_weight * base_logits
            + (1.0 - self.base_logits_weight) * direct_logits
        )


class FlowMatchingPlanTransformer(nn.Module):
    def __init__(
        self,
        hidden_dimension=512,
        num_heads=8,
        head_dimension=64,
        num_node_layers=2,
        num_corner_layers=4,
        dropout=0.0,
        detach_bit_from_velocity=True,
        base_logits_weight=0.0,
    ):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.node_input_projection = InputProjection(
            NODE_INPUT_DIM,
            hidden_dimension,
        )
        self.area_projection = InputProjection(
            AREA_FEATURE_DIM,
            hidden_dimension,
        )
        self.corner_input_projection = InputProjection(
            CORNER_GEOMETRY_DIM,
            hidden_dimension,
        )
        self.clip_projection = InputProjection(
            CLIP_DIM,
            hidden_dimension,
        )
        self.time_embedding = TimestepEmbedding(hidden_dimension)
        self.corner_position_encoding = CornerPositionalEncoding(
            hidden_dimension
        )
        self.node_layers = nn.ModuleList(
            [
                NodeTransformerBlock(
                    hidden_dimension,
                    num_heads=num_heads,
                    head_dimension=head_dimension,
                    dropout=dropout,
                )
                for _ in range(num_node_layers)
            ]
        )
        self.corner_layers = nn.ModuleList(
            [
                CornerTransformerBlock(
                    hidden_dimension,
                    num_heads=num_heads,
                    head_dimension=head_dimension,
                    dropout=dropout,
                )
                for _ in range(num_corner_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dimension)
        self.velocity_projection = nn.Linear(hidden_dimension, 2)
        self.discrete_head = DiscreteCoordinateRefinementHead(
            hidden_dimension,
            num_heads=num_heads,
            head_dimension=head_dimension,
            dropout=dropout,
            detach_from_velocity=detach_bit_from_velocity,
            base_logits_weight=base_logits_weight,
        )

    def _corner_inputs(self, corner_features, corner_layout):
        valid = corner_layout["corner_valid"]
        room_ids = corner_layout["room_ids"].long()
        corner_ids = corner_layout["corner_ids"].long()
        geometry = torch.cat(
            [
                corner_features[:, :, :4],
                corner_layout["ring_phase"],
                corner_layout["corner_id_norm"],
            ],
            dim=-1,
        )
        return (
            geometry,
            valid,
            room_ids,
            corner_ids,
        )

    def forward(
        self,
        corner_features,
        corner_layout,
        node_features,
        area_condition,
        clip_embedding,
        sigma,
        corner_room_mask,
        corner_connection_mask,
        corner_global_mask,
        corner_next_index,
        corner_edge_valid,
        node_connection_mask,
        node_global_mask,
        return_discrete=False,
    ):
        (
            geometry,
            corner_valid,
            room_ids,
            corner_ids,
        ) = self._corner_inputs(corner_features, corner_layout)
        room_hidden = self.node_input_projection(node_features)
        time_clip_condition = (
            self.time_embedding(sigma * 1000.0)
            + self.clip_projection(clip_embedding)
        )

        for layer in self.node_layers:
            room_hidden = layer(
                room_hidden,
                node_connection_mask,
                node_global_mask,
            )
        if area_condition is not None:
            room_hidden = room_hidden + self.area_projection(
                area_condition
            )

        corner_hidden = self.corner_input_projection(geometry)
        corner_hidden = self.corner_position_encoding(
            corner_hidden,
            corner_ids,
        )
        room_condition = torch.gather(
            room_hidden,
            1,
            room_ids.unsqueeze(-1).expand(
                -1,
                -1,
                self.hidden_dimension,
            ),
        )
        for layer in self.corner_layers:
            corner_hidden = layer(
                corner_hidden,
                time_clip_condition,
                room_condition,
                corner_room_mask,
                corner_connection_mask,
                corner_global_mask,
            )
        corner_hidden = self.final_norm(corner_hidden)
        velocity = self.velocity_projection(corner_hidden)
        velocity = velocity * corner_valid.unsqueeze(-1).to(
            velocity.dtype
        )
        if not return_discrete:
            return velocity

        x0_prediction = (
            corner_features[:, :, :2]
            - sigma[:, None, None] * velocity
        )
        x0_au = build_edge_au_features(
            x0_prediction,
            corner_next_index,
            corner_edge_valid,
        )
        base_logits = quantize_corner_bits(x0_prediction)
        discrete_logits = self.discrete_head(
            room_condition,
            x0_au,
            base_logits,
            corner_room_mask,
            corner_connection_mask,
            corner_global_mask,
        )
        return velocity, discrete_logits


MainFlowTransformer = FlowMatchingPlanTransformer

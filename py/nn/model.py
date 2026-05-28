from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.fx.experimental.symbolic_shapes import expect_true as _torch_expect_true

from search.config import ModelConfig
from .encoding import BOARD_FEATURE_INDEX


POPULATED_BOARD_CHANNELS = frozenset(BOARD_FEATURE_INDEX.values())


@dataclass
class ModelOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    pooled_state: torch.Tensor
    debug: dict[str, torch.Tensor] | None = None


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _make_spatial_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _make_spatial_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _make_spatial_norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class HybridPolicyValueNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.empty_board_channel_indices = [
            channel for channel in range(cfg.board_channels) if channel not in POPULATED_BOARD_CHANNELS
        ]
        board_layers: list[nn.Module] = [
            nn.Conv2d(cfg.board_channels, cfg.cnn_channels, kernel_size=3, padding=1),
            _make_spatial_norm(cfg.cnn_channels),
            nn.GELU(),
        ]
        board_layers.extend(ResidualConvBlock(cfg.cnn_channels) for _ in range(max(1, int(getattr(cfg, "board_res_blocks", 2)))))
        board_layers.extend(
            [
                nn.Conv2d(cfg.cnn_channels, cfg.d_model, kernel_size=1),
                nn.GELU(),
            ]
        )
        self.board_encoder = nn.Sequential(*board_layers)
        self.unit_proj = nn.Linear(getattr(cfg, "unit_feature_dim", cfg.entity_feature_dim), cfg.d_model)
        self.city_proj = nn.Linear(getattr(cfg, "city_feature_dim", cfg.entity_feature_dim), cfg.d_model)
        self.action_proj = nn.Linear(cfg.action_feature_dim, cfg.d_model)
        self.scalar_value_proj = nn.Linear(1, cfg.d_model)
        self.scalar_index_embeddings = nn.Embedding(cfg.scalar_dim, cfg.d_model)
        self.scalar_summary_proj = nn.Sequential(
            nn.Linear(cfg.scalar_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.use_cls_token else None
        self.token_type_embeddings = nn.Embedding(7, cfg.d_model)
        self.token_dropout = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.core = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.action_attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.value_action_attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.policy_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
        self.value_head = nn.Sequential(nn.Linear(cfg.d_model * 4, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(
        self,
        encoded,
        *,
        return_debug: bool = False,
    ) -> ModelOutput:
        board_tokens = self._add_token_type(self._encode_board(encoded.board), 3)
        unit_tokens = self._add_token_type(self.unit_proj(encoded.unit_features), 4)
        city_tokens = self._add_token_type(self.city_proj(encoded.city_features), 5)
        scalar_tokens = self._add_token_type(self._encode_scalars(encoded.scalar_features), 0)
        scalar_summary_token = self._add_token_type(self.scalar_summary_proj(encoded.scalar_features).unsqueeze(1), 1)
        batch_size = encoded.board.shape[0]

        pieces = []
        cls_len = 0
        if self.cls_token is not None:
            pieces.append(self._add_token_type(self.cls_token.expand(batch_size, -1, -1), 6))
            cls_len = 1
        pieces.extend([scalar_summary_token, scalar_tokens, board_tokens, unit_tokens, city_tokens])
        tokens = self.token_dropout(torch.cat(pieces, dim=1))

        scalar_len = encoded.scalar_features.shape[1]
        board_start = cls_len + 1 + scalar_len
        unit_start = board_start + board_tokens.shape[1]
        city_start = unit_start + encoded.unit_features.shape[1]
        key_padding_mask = torch.zeros(batch_size, tokens.shape[1], dtype=torch.bool, device=tokens.device)
        key_padding_mask[:, unit_start:city_start] = ~encoded.unit_mask
        key_padding_mask[:, city_start:] = ~encoded.city_mask

        latent = self.core(tokens, src_key_padding_mask=key_padding_mask)
        if self.cls_token is not None:
            pooled = latent[:, 0, :]
        else:
            valid = (~key_padding_mask).unsqueeze(-1).to(latent.dtype)
            pooled = (latent * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        action_queries = self.action_proj(encoded.action_features)
        attended, attention_weights = self.action_attention(
            action_queries,
            latent,
            latent,
            key_padding_mask=key_padding_mask,
            need_weights=return_debug,
            average_attn_weights=False,
        )
        logits = self.policy_head(attended).squeeze(-1).masked_fill(~encoded.action_mask, -1e9)
        action_context = self._pool_actions_for_value(pooled, attended, encoded.action_mask)
        value_input = torch.cat([pooled, action_context], dim=-1)
        value = torch.tanh(self.value_head(value_input).squeeze(-1))
        debug = None
        if return_debug:
            debug = {
                "tokens": tokens,
                "latent": latent,
                "pooled_state": pooled,
                "action_attended": attended,
                "value_action_context": action_context,
            }
            if attention_weights is not None:
                debug["action_attention_weights"] = attention_weights
        return ModelOutput(
            policy_logits=logits,
            value=value,
            pooled_state=pooled,
            debug=debug,
        )

    def _encode_board(self, board: torch.Tensor) -> torch.Tensor:
        board = self._with_empty_board_priors(board)
        spatial = self.board_encoder(board)
        batch, channels, height, width = spatial.shape
        return spatial.view(batch, channels, height * width).transpose(1, 2)

    def _with_empty_board_priors(self, board: torch.Tensor) -> torch.Tensor:
        if (
            not self.empty_board_channel_indices
            or not getattr(self.cfg, "use_empty_board_coordinate_priors", True)
        ):
            return board
        priors = self._coordinate_prior_planes(board)
        board = board.clone()
        board[:, self.empty_board_channel_indices, :, :] = priors
        return board

    def _coordinate_prior_planes(self, board: torch.Tensor) -> torch.Tensor:
        _, _, height, width = board.shape
        dtype = board.dtype
        device = board.device
        y = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device).view(1, height, 1).expand(1, height, width)
        x = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device).view(1, 1, width).expand(1, height, width)
        centered_x = x * 2.0 - 1.0
        centered_y = y * 2.0 - 1.0
        edge_distance = torch.minimum(torch.minimum(x, 1.0 - x), torch.minimum(y, 1.0 - y)) * 2.0
        center_distance = torch.sqrt(centered_x.square() + centered_y.square()).clamp(max=1.0)
        aspect = torch.full_like(x, float(width) / max(1.0, float(height)))
        area = torch.full_like(x, float(height * width) / float(max(1, self.cfg.board_size * self.cfg.board_size)))

        base_planes = [
            x,
            y,
            centered_x,
            centered_y,
            edge_distance,
            center_distance,
            1.0 - center_distance,
            aspect,
            area,
        ]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            base_planes.extend(
                [
                    torch.sin(x * frequency * torch.pi),
                    torch.cos(x * frequency * torch.pi),
                    torch.sin(y * frequency * torch.pi),
                    torch.cos(y * frequency * torch.pi),
                ]
            )
        priors = torch.cat(base_planes, dim=0).unsqueeze(0)
        needed = len(self.empty_board_channel_indices)
        if priors.shape[1] < needed:
            repeat_count = (needed + priors.shape[1] - 1) // priors.shape[1]
            priors = priors.repeat(1, repeat_count, 1, 1)
        priors = priors[:, :needed, :, :]
        return priors.expand(board.shape[0], -1, -1, -1)

    def _encode_scalars(self, scalar_features: torch.Tensor) -> torch.Tensor:
        scalar_values = scalar_features.unsqueeze(-1)
        scalar_tokens = self.scalar_value_proj(scalar_values)
        scalar_ids = torch.arange(
            scalar_features.shape[1],
            dtype=torch.long,
            device=scalar_features.device,
        )
        return scalar_tokens + self.scalar_index_embeddings(scalar_ids).unsqueeze(0)

    def _add_token_type(self, tokens: torch.Tensor, token_type: int) -> torch.Tensor:
        type_ids = torch.full((tokens.shape[0], tokens.shape[1]), token_type, dtype=torch.long, device=tokens.device)
        return tokens + self.token_type_embeddings(type_ids)

    def _pool_actions_for_value(
        self,
        pooled: torch.Tensor,
        action_tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if action_tokens.shape[1] == 0:
            return action_tokens.new_zeros((action_tokens.shape[0], action_tokens.shape[2] * 3))
        has_action = action_mask.any(dim=1, keepdim=True)
        valid = action_mask.unsqueeze(-1).to(action_tokens.dtype)
        count = valid.sum(dim=1).clamp_min(1.0)
        mean_pool = (action_tokens * valid).sum(dim=1) / count

        masked_actions = action_tokens.masked_fill(~action_mask.unsqueeze(-1), torch.finfo(action_tokens.dtype).min)
        max_pool = masked_actions.max(dim=1).values
        max_pool = torch.where(has_action, max_pool, torch.zeros_like(max_pool))

        safe_action_mask = action_mask.clone()
        safe_action_mask[~has_action.squeeze(1), 0] = True
        safe_action_tokens = action_tokens.masked_fill(~safe_action_mask.unsqueeze(-1), 0.0)
        attn_pool, _ = self.value_action_attention(
            pooled.unsqueeze(1),
            safe_action_tokens,
            safe_action_tokens,
            key_padding_mask=~safe_action_mask,
            need_weights=False,
        )
        attn_pool = attn_pool.squeeze(1)
        attn_pool = torch.where(has_action, attn_pool, torch.zeros_like(attn_pool))
        return torch.cat([mean_pool, max_pool, attn_pool], dim=-1)

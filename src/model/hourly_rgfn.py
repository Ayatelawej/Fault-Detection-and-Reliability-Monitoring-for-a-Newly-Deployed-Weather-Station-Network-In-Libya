from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn

from src.model.feature_spec import CONTINUOUS_FEATURES, STATIC_FEATURES, rule_evidence_feature_names
from src.model.hourly_detection import MASK_MODE_PER_FEATURE, MASK_MODE_PER_HOUR, MASK_MODES


SHORT_WINDOW_HOURS = 7
SENSOR_HIDDEN_SIZE = 48
EVIDENCE_HIDDEN_SIZE = 32
EVIDENCE_EMBED_SIZE = 16
FUSION_HIDDEN_SIZE = 32
CONV_FIRST_CHANNELS = 32
CONV_SECOND_CHANNELS = 64
DEFAULT_DROPOUT = 0.3
ENCODER_GRU = "gru"
ENCODER_CONV = "conv"
ENCODER_MLP = "mlp"
ENCODER_NAMES = (ENCODER_GRU, ENCODER_CONV, ENCODER_MLP)


@dataclass(frozen=True)
class HourlyRgfnConfig:
    n_continuous: int = len(CONTINUOUS_FEATURES)
    n_static: int = len(STATIC_FEATURES)
    n_rule_evidence: int = len(rule_evidence_feature_names())
    window_hours: int = SHORT_WINDOW_HOURS
    sensor_hidden_size: int = SENSOR_HIDDEN_SIZE
    evidence_hidden_size: int = EVIDENCE_HIDDEN_SIZE
    evidence_embed_size: int = EVIDENCE_EMBED_SIZE
    fusion_hidden_size: int = FUSION_HIDDEN_SIZE
    dropout: float = DEFAULT_DROPOUT
    mask_mode: str = MASK_MODE_PER_HOUR


def set_hourly_rgfn_seed(seed: int) -> None:
    resolved = int(seed)
    random.seed(resolved)
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HourlyReliabilityAwareGatedFusionNetwork(nn.Module):
    def __init__(
        self,
        encoder: str = ENCODER_GRU,
        config: HourlyRgfnConfig | None = None,
        n_continuous: int | None = None,
        n_static: int | None = None,
        n_rule_evidence: int | None = None,
        window_hours: int | None = None,
        dropout: float | None = None,
        mask_mode: str | None = None,
        output_dim: int = 1,
        mechanism_count: int | None = None,
    ) -> None:
        super().__init__()
        base = config or HourlyRgfnConfig()
        self.encoder_name = str(encoder).lower()
        if self.encoder_name not in ENCODER_NAMES:
            raise KeyError(f"unknown hourly RGFN encoder: {encoder}")
        self.n_continuous = int(base.n_continuous if n_continuous is None else n_continuous)
        self.n_static = int(base.n_static if n_static is None else n_static)
        self.n_rule_evidence = int(base.n_rule_evidence if n_rule_evidence is None else n_rule_evidence)
        self.window_hours = int(base.window_hours if window_hours is None else window_hours)
        self.sensor_hidden_size = int(base.sensor_hidden_size)
        self.evidence_embed_size = int(base.evidence_embed_size)
        self.output_dim = int(output_dim)
        self.mechanism_count = None if mechanism_count is None else int(mechanism_count)
        resolved_dropout = float(base.dropout if dropout is None else dropout)
        self.mask_mode = str(base.mask_mode if mask_mode is None else mask_mode)
        if self.n_continuous < 1 or self.n_static < 0 or self.n_rule_evidence < 1:
            raise ValueError("hourly RGFN feature widths must be valid")
        if self.output_dim < 1:
            raise ValueError("hourly RGFN output_dim must be positive")
        if self.output_dim == 1 and self.mechanism_count is not None:
            raise ValueError("a binary hourly RGFN cannot define mechanism_count")
        if self.output_dim > 1 and (
            self.mechanism_count is None
            or self.mechanism_count < 1
            or self.mechanism_count >= self.output_dim
        ):
            raise ValueError("a multi-label hourly RGFN needs a mechanism_count within its output width")
        if self.window_hours < 1:
            raise ValueError("hourly RGFN window_hours must be positive")
        if self.encoder_name == ENCODER_MLP and self.window_hours != 1:
            raise ValueError("the RGFN MLP encoder requires a one-hour input")
        if not 0.0 <= resolved_dropout < 1.0:
            raise ValueError("hourly RGFN dropout must be in [0, 1)")
        if self.mask_mode not in MASK_MODES:
            raise ValueError(f"unknown hourly RGFN mask mode: {self.mask_mode}")
        self.input_width = (
            self.n_continuous + 2
            if self.mask_mode == MASK_MODE_PER_HOUR
            else self.n_continuous * 2 + 1
        )
        self.temporal_dropout = nn.Dropout(resolved_dropout)
        if self.encoder_name == ENCODER_GRU:
            self.sensor_encoder = nn.GRU(
                input_size=self.input_width,
                hidden_size=self.sensor_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.sensor_projection: nn.Module = nn.Identity()
        elif self.encoder_name == ENCODER_CONV:
            self.sensor_encoder = nn.Sequential(
                nn.Conv1d(self.input_width, CONV_FIRST_CHANNELS, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(resolved_dropout),
                nn.Conv1d(CONV_FIRST_CHANNELS, CONV_SECOND_CHANNELS, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.sensor_projection = nn.Linear(CONV_SECOND_CHANNELS, self.sensor_hidden_size)
        else:
            self.sensor_encoder = nn.Sequential(
                nn.Linear(self.input_width, self.sensor_hidden_size),
                nn.ReLU(),
                nn.Dropout(resolved_dropout),
                nn.Linear(self.sensor_hidden_size, self.sensor_hidden_size),
                nn.ReLU(),
            )
            self.sensor_projection = nn.Identity()
        self.temporal_head = nn.Linear(self.sensor_hidden_size, self.output_dim)
        self.rule_net = nn.Sequential(
            nn.Linear(self.n_rule_evidence, int(base.evidence_hidden_size)),
            nn.ReLU(),
            nn.Dropout(resolved_dropout),
            nn.Linear(int(base.evidence_hidden_size), self.evidence_embed_size),
            nn.ReLU(),
            nn.Dropout(resolved_dropout),
        )
        self.rule_head = nn.Linear(self.evidence_embed_size, self.output_dim)
        fusion_width = self.sensor_hidden_size + self.evidence_embed_size + self.n_static
        self.gate = nn.Sequential(
            nn.Linear(fusion_width, int(base.fusion_hidden_size)),
            nn.ReLU(),
            nn.Linear(int(base.fusion_hidden_size), self.output_dim),
        )

    @property
    def evidence_gate(self) -> nn.Module:
        return self.gate

    @property
    def evidence_net(self) -> nn.Module:
        return self.rule_net

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def normalized_time_since_last(self, time_since_last: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(time_since_last, min=0.0, max=float(self.window_hours))
        return torch.log1p(clipped) / math.log1p(float(self.window_hours))

    def _validate_inputs(
        self,
        x_cont: torch.Tensor,
        mask: torch.Tensor,
        time_since_last: torch.Tensor,
        static: torch.Tensor,
        rule_evidence: torch.Tensor,
    ) -> None:
        if x_cont.ndim != 3:
            raise ValueError("X_cont must have shape [examples, hours, continuous_features]")
        expected_mask_width = 1 if self.mask_mode == MASK_MODE_PER_HOUR else self.n_continuous
        expected_mask_sequence = (x_cont.shape[0], self.window_hours, expected_mask_width)
        expected_time_sequence = (x_cont.shape[0], self.window_hours, 1)
        if x_cont.shape[1] != self.window_hours or x_cont.shape[2] != self.n_continuous:
            raise ValueError("X_cont does not match the configured hourly RGFN shape")
        if tuple(mask.shape) != expected_mask_sequence:
            raise ValueError("mask does not match the configured hourly RGFN mask mode")
        if tuple(time_since_last.shape) != expected_time_sequence:
            raise ValueError("time_since_last does not match the configured hourly RGFN shape")
        if tuple(static.shape) != (x_cont.shape[0], self.n_static):
            raise ValueError("static does not match the configured hourly RGFN shape")
        if tuple(rule_evidence.shape) != (x_cont.shape[0], self.n_rule_evidence):
            raise ValueError("rule_evidence does not match the configured hourly RGFN shape")

    def temporal_input(
        self,
        x_cont: torch.Tensor,
        mask: torch.Tensor,
        time_since_last: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.nan_to_num(x_cont, nan=0.0, posinf=0.0, neginf=0.0)
        present = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        elapsed = torch.nan_to_num(time_since_last, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([values, present, self.normalized_time_since_last(elapsed)], dim=-1)

    def sensor_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.encoder_name == ENCODER_GRU:
            _, hidden = self.sensor_encoder(sequence)
            return self.temporal_dropout(hidden[-1])
        if self.encoder_name == ENCODER_MLP:
            return self.temporal_dropout(self.sensor_encoder(sequence[:, 0, :]))
        encoded = self.sensor_encoder(sequence.transpose(1, 2))
        pooled = encoded.mean(dim=-1)
        return self.temporal_dropout(torch.relu(self.sensor_projection(pooled)))

    def forward(
        self,
        x_cont: torch.Tensor,
        mask: torch.Tensor,
        time_since_last: torch.Tensor,
        static: torch.Tensor,
        rule_evidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(x_cont, mask, time_since_last, static, rule_evidence)
        sequence = self.temporal_dropout(self.temporal_input(x_cont, mask, time_since_last))
        temporal_embedding = self.sensor_embedding(sequence)
        evidence_values = torch.nan_to_num(rule_evidence, nan=0.0, posinf=0.0, neginf=0.0)
        static_values = torch.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
        evidence_embedding = self.rule_net(evidence_values)
        fusion = torch.cat([temporal_embedding, evidence_embedding, static_values], dim=-1)
        if self.output_dim == 1:
            temporal_logit = self.temporal_head(temporal_embedding).squeeze(-1)
            rule_logit = self.rule_head(evidence_embedding).squeeze(-1)
            alpha = torch.sigmoid(self.gate(fusion).squeeze(-1))
            final_fault_logit = alpha * temporal_logit + (1.0 - alpha) * rule_logit
            probability = torch.sigmoid(final_fault_logit)
            return {
                "final_fault_logit": final_fault_logit,
                "fault_logit": final_fault_logit,
                "binary_prob": probability,
                "temporal_logit": temporal_logit,
                "sensor_logit": temporal_logit,
                "rule_logit": rule_logit,
                "evidence_logit": rule_logit,
                "alpha": alpha,
            }
        temporal_logits = self.temporal_head(temporal_embedding)
        rule_logits = self.rule_head(evidence_embedding)
        alpha = torch.sigmoid(self.gate(fusion))
        reason_code_logits = alpha * temporal_logits + (1.0 - alpha) * rule_logits
        probabilities = torch.sigmoid(reason_code_logits)
        mechanism_count = int(self.mechanism_count)
        return {
            "reason_code_logits": reason_code_logits,
            "reason_code_probabilities": probabilities,
            "mechanism_logits": reason_code_logits[:, :mechanism_count],
            "component_logits": reason_code_logits[:, mechanism_count:],
            "mechanism_probabilities": probabilities[:, :mechanism_count],
            "component_probabilities": probabilities[:, mechanism_count:],
            "temporal_logits": temporal_logits,
            "rule_logits": rule_logits,
            "alpha": alpha,
        }

    def predict_proba(
        self,
        x_cont: torch.Tensor,
        mask: torch.Tensor,
        time_since_last: torch.Tensor,
        static: torch.Tensor,
        rule_evidence: torch.Tensor,
    ) -> torch.Tensor:
        output = self(x_cont, mask, time_since_last, static, rule_evidence)
        if self.output_dim == 1:
            return output["binary_prob"]
        return output["reason_code_probabilities"]


class HourlyRgfnGru(HourlyReliabilityAwareGatedFusionNetwork):
    def __init__(self, config: HourlyRgfnConfig | None = None, **kwargs: object) -> None:
        super().__init__(encoder=ENCODER_GRU, config=config, **kwargs)


class HourlyRgfnConv(HourlyReliabilityAwareGatedFusionNetwork):
    def __init__(self, config: HourlyRgfnConfig | None = None, **kwargs: object) -> None:
        super().__init__(encoder=ENCODER_CONV, config=config, **kwargs)


class HourlyRgfnMlp(HourlyReliabilityAwareGatedFusionNetwork):
    def __init__(self, config: HourlyRgfnConfig | None = None, **kwargs: object) -> None:
        super().__init__(encoder=ENCODER_MLP, config=config, **kwargs)


HourlyRGFNGRU = HourlyRgfnGru
HourlyRGFNConv = HourlyRgfnConv
HourlyRGFNMLP = HourlyRgfnMlp
HourlyRuleGatedFusionNetwork = HourlyReliabilityAwareGatedFusionNetwork


def build_hourly_rgfn(
    encoder: str,
    config: HourlyRgfnConfig | None = None,
    **kwargs: object,
) -> HourlyReliabilityAwareGatedFusionNetwork:
    resolved = str(encoder).lower()
    if resolved == ENCODER_GRU:
        return HourlyRgfnGru(config=config, **kwargs)
    if resolved == ENCODER_CONV:
        return HourlyRgfnConv(config=config, **kwargs)
    if resolved == ENCODER_MLP:
        return HourlyRgfnMlp(config=config, **kwargs)
    raise KeyError(f"unknown hourly RGFN encoder: {encoder}")

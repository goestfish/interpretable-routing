import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_num_experts(base_block: nn.Module) -> int:
    if hasattr(base_block, "gate") and hasattr(base_block.gate, "num_experts"):
        return base_block.gate.num_experts
    if hasattr(base_block, "num_experts"):
        return base_block.num_experts
    if hasattr(base_block, "experts") and hasattr(base_block.experts, "num_experts"):
        return base_block.experts.num_experts
    if hasattr(base_block, "experts"):
        return len(base_block.experts)
    if hasattr(base_block, "gate") and hasattr(base_block.gate, "weight"):
        return base_block.gate.weight.shape[0]
    raise AttributeError("Could not infer number of experts from OLMoE MoE block.")


def infer_top_k(base_block: nn.Module) -> int:
    if hasattr(base_block, "gate") and hasattr(base_block.gate, "top_k"):
        return base_block.gate.top_k
    if hasattr(base_block, "top_k"):
        return base_block.top_k
    if hasattr(base_block, "topk"):
        return base_block.topk
    if hasattr(base_block, "num_experts_per_tok"):
        return base_block.num_experts_per_tok
    if hasattr(base_block, "config") and hasattr(base_block.config, "num_experts_per_tok"):
        return base_block.config.num_experts_per_tok
    return 2


def infer_norm_topk_prob(base_block: nn.Module) -> bool:
    if hasattr(base_block, "gate") and hasattr(base_block.gate, "norm_topk_prob"):
        return base_block.gate.norm_topk_prob
    if hasattr(base_block, "norm_topk_prob"):
        return base_block.norm_topk_prob
    if hasattr(base_block, "norm_topk"):
        return base_block.norm_topk
    if hasattr(base_block, "config") and hasattr(base_block.config, "norm_topk_prob"):
        return base_block.config.norm_topk_prob
    return False


class ZRouterBias(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, num_z: int, mlp_hidden_size: int | None = None):
        super().__init__()
        mlp_hidden_size = mlp_hidden_size or hidden_size // 4
        self.num_z = num_z
        self.num_experts = num_experts
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, num_z),
        )
        self.U = nn.Parameter(torch.zeros(num_z, num_experts))

    def forward(self, hidden_states: torch.Tensor, tau: float = 1.0, hard: bool = True):
        z_logits = self.predictor(hidden_states)
        z = F.gumbel_softmax(z_logits.float(), tau=tau, hard=hard, dim=-1).to(hidden_states.dtype)
        router_bias = z @ self.U.to(z.dtype)
        return router_bias, z_logits, z


class SharedZPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_z: int, mlp_hidden_size: int | None = None):
        super().__init__()
        mlp_hidden_size = mlp_hidden_size or hidden_size // 4
        self.num_z = num_z
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, num_z),
        )
        self.cached_z_logits = None
        self.cached_z = None
        self.last_balance_loss = None

    def reset_cache(self):
        self.cached_z_logits = None
        self.cached_z = None
        self.last_balance_loss = None

    def compute(self, hidden_states: torch.Tensor, tau: float = 1.0, hard: bool = True):
        z_logits = self.predictor(hidden_states)
        z = F.gumbel_softmax(z_logits.float(), tau=tau, hard=hard, dim=-1).to(hidden_states.dtype)
        self.cached_z_logits = z_logits
        self.cached_z = z
        mean_z = z.float().mean(dim=0).clamp_min(1e-8)
        self.last_balance_loss = (mean_z * (mean_z * self.num_z).log()).sum()
        return z_logits, z

    def get_cached(self):
        if self.cached_z is None or self.cached_z_logits is None:
            raise RuntimeError("Shared z was requested before the block-entry layer computed it.")
        return self.cached_z_logits, self.cached_z


class ZAugmentedOlmoeSparseMoeBlock(nn.Module):
    """Wrap an OLMoE sparse MoE block and add a learned z-dependent router bias."""

    def __init__(
        self,
        base_block: nn.Module,
        hidden_size: int,
        num_z: int,
        z_mlp_hidden_size: int | None = None,
        tau: float = 1.0,
        hard: bool = True,
    ):
        super().__init__()
        self.base_block = base_block
        self.num_experts = infer_num_experts(base_block)
        self.top_k = infer_top_k(base_block)
        self.norm_topk_prob = infer_norm_topk_prob(base_block)
        self.tau = tau
        self.hard = hard
        self.last_z_summary = None
        self.z_router = ZRouterBias(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
            num_z=num_z,
            mlp_hidden_size=z_mlp_hidden_size,
        )

    @property
    def gate(self):
        return self.base_block.gate

    @property
    def experts(self):
        return self.base_block.experts

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(flat_hidden, self.gate.weight)
        router_bias, z_logits, z = self.z_router(flat_hidden, tau=self.tau, hard=self.hard)
        if not self.training:
            self.last_z_summary = {
                "z_logits": z_logits.detach().float().cpu(),
                "z": z.detach().float().cpu(),
            }
        router_logits = router_logits + router_bias.to(router_logits.dtype)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

        final_hidden_states = self.experts(flat_hidden, selected_experts, routing_weights).reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class CrossLayerSharedZOlmoeSparseMoeBlock(nn.Module):
    """OLMoE MoE block wrapper using one cross-layer shared z and per-layer router bias."""

    def __init__(
        self,
        base_block: nn.Module,
        shared_predictor: SharedZPredictor,
        num_z: int,
        compute_z: bool,
        shared_U: nn.Parameter | None = None,
        tau: float = 1.0,
        hard: bool = True,
        alpha_init: float = 1e-3,
    ):
        super().__init__()
        self.base_block = base_block
        self.shared_predictor = shared_predictor
        self.num_experts = infer_num_experts(base_block)
        self.top_k = infer_top_k(base_block)
        self.norm_topk_prob = infer_norm_topk_prob(base_block)
        self.num_z = num_z
        self.compute_z = compute_z
        self.tau = tau
        self.hard = hard
        self.last_z_summary = None
        self.last_perturb_loss = None
        if shared_U is None:
            self.U = nn.Parameter(torch.zeros(num_z, self.num_experts))
        else:
            self.register_parameter("U", shared_U)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def gate(self):
        return self.base_block.gate

    @property
    def experts(self):
        return self.base_block.experts

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        if self.compute_z:
            z_logits, z = self.shared_predictor.compute(flat_hidden, tau=self.tau, hard=self.hard)
        else:
            z_logits, z = self.shared_predictor.get_cached()

        router_logits = F.linear(flat_hidden, self.gate.weight)
        router_bias = z @ self.U.to(z.dtype)
        scaled_bias = self.alpha.to(router_bias.dtype) * router_bias
        self.last_perturb_loss = scaled_bias.float().pow(2).mean()
        if not self.training:
            self.last_z_summary = {
                "z_logits": z_logits.detach().float().cpu(),
                "z": z.detach().float().cpu(),
            }
        router_logits = router_logits + scaled_bias.to(router_logits.dtype)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

        final_hidden_states = self.experts(flat_hidden, selected_experts, routing_weights).reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


def freeze_all_parameters(model: nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def get_olmoe_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Expected an OLMoE-style model with model.layers.")


def install_z_router_blocks(
    model: nn.Module,
    start_layer: int,
    block_size: int,
    num_z: int,
    z_mlp_hidden_size: int | None = None,
    tau: float = 1.0,
    hard: bool = True,
    allow_router_update: bool = False,
    sharing: str = "independent",
    u_sharing: str = "per_layer",
    sharing_group_size: int | None = None,
    alpha_init: float = 1e-3,
):
    layers = get_olmoe_layers(model)
    hidden_size = model.config.hidden_size
    installed = []

    if sharing not in {"independent", "cross_layer_shared"}:
        raise ValueError(f"Unsupported sharing mode: {sharing}")
    if u_sharing not in {"per_layer", "shared"}:
        raise ValueError(f"Unsupported U sharing mode: {u_sharing}")

    if sharing == "cross_layer_shared":
        sharing_group_size = sharing_group_size or block_size
        if sharing_group_size <= 0:
            raise ValueError("sharing_group_size must be positive.")
        if block_size % sharing_group_size != 0:
            raise ValueError("block_size must be divisible by sharing_group_size.")

        for group_start in range(start_layer, start_layer + block_size, sharing_group_size):
            first_block = layers[group_start].mlp
            reference_parameter = next(first_block.parameters())
            shared_predictor = SharedZPredictor(
                hidden_size=hidden_size,
                num_z=num_z,
                mlp_hidden_size=z_mlp_hidden_size,
            )
            shared_predictor.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
            shared_U = None
            if u_sharing == "shared":
                shared_U = nn.Parameter(
                    torch.zeros(
                        num_z,
                        infer_num_experts(first_block),
                        device=reference_parameter.device,
                        dtype=reference_parameter.dtype,
                    )
                )

            for layer_idx in range(group_start, group_start + sharing_group_size):
                base_block = layers[layer_idx].mlp
                reference_parameter = next(base_block.parameters())
                wrapped = CrossLayerSharedZOlmoeSparseMoeBlock(
                    base_block=base_block,
                    shared_predictor=shared_predictor,
                    num_z=num_z,
                    compute_z=layer_idx == group_start,
                    shared_U=shared_U,
                    tau=tau,
                    hard=hard,
                    alpha_init=alpha_init,
                )
                wrapped.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
                layers[layer_idx].mlp = wrapped
                installed.append(layer_idx)
                for parameter in wrapped.shared_predictor.parameters():
                    parameter.requires_grad_(True)
                wrapped.U.requires_grad_(True)
                wrapped.alpha.requires_grad_(True)
                if allow_router_update:
                    wrapped.gate.weight.requires_grad_(True)
        return installed

    for layer_idx in range(start_layer, start_layer + block_size):
        base_block = layers[layer_idx].mlp
        reference_parameter = next(base_block.parameters())
        wrapped = ZAugmentedOlmoeSparseMoeBlock(
            base_block=base_block,
            hidden_size=hidden_size,
            num_z=num_z,
            z_mlp_hidden_size=z_mlp_hidden_size,
            tau=tau,
            hard=hard,
        )
        wrapped.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        layers[layer_idx].mlp = wrapped
        installed.append(layer_idx)

        for parameter in wrapped.z_router.parameters():
            parameter.requires_grad_(True)
        if allow_router_update:
            wrapped.gate.weight.requires_grad_(True)

    return installed


def collect_z_router_losses(model: nn.Module):
    balance_losses = []
    perturb_losses = []
    seen_predictors = set()
    for module in model.modules():
        if isinstance(module, SharedZPredictor) and module.last_balance_loss is not None:
            module_id = id(module)
            if module_id not in seen_predictors:
                balance_losses.append(module.last_balance_loss)
                seen_predictors.add(module_id)
        if hasattr(module, "last_perturb_loss") and module.last_perturb_loss is not None:
            perturb_losses.append(module.last_perturb_loss)
    device = next(model.parameters()).device
    zero = torch.zeros((), device=device)
    balance_loss = torch.stack(balance_losses).mean() if balance_losses else zero
    perturb_loss = torch.stack(perturb_losses).mean() if perturb_losses else zero
    return balance_loss, perturb_loss


def save_trainable_state(model: nn.Module, output_dir: str | Path, metadata: dict):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(trainable_state, output_dir / "z_router_trainable_state.pt")
    (output_dir / "z_router_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

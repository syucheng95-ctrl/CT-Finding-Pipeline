"""DoRA / LoRA modules, experiment configs, and VoxTell model builder."""

import copy
import json
import pydoc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── vendor path ──
VOXTELL_VENDOR = Path(__file__).resolve().parent.parent / "workspace" / "voxtell"
if str(VOXTELL_VENDOR) not in sys.path:
    sys.path.insert(0, str(VOXTELL_VENDOR))


# ======================================================================
# DoraLinear, DoraMHA (same as stage1_workspace)
# ======================================================================

class DoraLinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.register_buffer("weight", linear.weight.detach().clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        self.magnitude = nn.Parameter(self.weight.data.norm(p=2, dim=1).clone())

    def _dora_weight(self):
        delta = self.lora_B @ self.lora_A
        combined = self.weight + self.scaling * delta
        norm = combined.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        return self.magnitude.unsqueeze(1) * (combined / norm)

    def forward(self, x):
        return F.linear(self.dropout(x), self._dora_weight(), self.bias)


class LoraLinear(nn.Module):
    """Pure LoRA wrapper (no DoRA magnitude) for experiment F."""
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.register_buffer("weight", linear.weight.detach().clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        delta = self.lora_B @ self.lora_A
        return F.linear(self.dropout(x), self.weight + self.scaling * delta, self.bias)


class DoraMultiheadAttention(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention, r: int = 8, alpha: int = 16, dropout: float = 0.0, use_dora: bool = True):
        super().__init__()
        embed_dim = mha.embed_dim
        self.embed_dim = embed_dim
        self.num_heads = mha.num_heads
        self.dropout = mha.dropout

        in_w = mha.in_proj_weight.detach()
        in_b = mha.in_proj_bias.detach() if mha.in_proj_bias is not None else None
        self.register_buffer("in_proj_weight", in_w.clone())
        if in_b is not None:
            self.register_buffer("in_proj_bias", in_b.clone())
        else:
            self.in_proj_bias = None

        self.scaling = alpha / r

        # Q/K/V LoRA adapters
        for pfx in ("q", "k", "v"):
            setattr(self, f"{pfx}_lora_A", nn.Parameter(torch.zeros(r, embed_dim)))
            setattr(self, f"{pfx}_lora_B", nn.Parameter(torch.zeros(embed_dim, r)))
            nn.init.kaiming_uniform_(getattr(self, f"{pfx}_lora_A"), a=5 ** 0.5)
            nn.init.zeros_(getattr(self, f"{pfx}_lora_B"))

        qw, kw, vw = in_w[:embed_dim].clone(), in_w[embed_dim:2*embed_dim].clone(), in_w[2*embed_dim:].clone()
        self.q_magnitude = nn.Parameter(qw.norm(p=2, dim=1).clone())
        self.k_magnitude = nn.Parameter(kw.norm(p=2, dim=1).clone())
        self.v_magnitude = nn.Parameter(vw.norm(p=2, dim=1).clone())
        if not use_dora:
            self.q_magnitude.requires_grad_(False)
            self.k_magnitude.requires_grad_(False)
            self.v_magnitude.requires_grad_(False)

        lin_cls = DoraLinear if use_dora else LoraLinear
        self.out_proj = lin_cls(mha.out_proj, r=r, alpha=alpha, dropout=dropout)

        for attr in ("bias_k", "bias_v"):
            val = getattr(mha, attr, None)
            self.register_buffer(attr, val.detach().clone() if val is not None else torch.tensor([]))

        self._use_dora = use_dora

    def _build_in_proj(self):
        def _apply(w, A, B, mag):
            delta = B @ A
            combined = w + self.scaling * delta
            if self._use_dora:
                norm = combined.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                return mag.unsqueeze(1) * (combined / norm)
            return combined

        e = self.embed_dim
        return torch.cat([
            _apply(self.in_proj_weight[:e], self.q_lora_A, self.q_lora_B, self.q_magnitude),
            _apply(self.in_proj_weight[e:2*e], self.k_lora_A, self.k_lora_B, self.k_magnitude),
            _apply(self.in_proj_weight[2*e:], self.v_lora_A, self.v_lora_B, self.v_magnitude),
        ], dim=0)

    def forward(self, query, key, value, **kwargs):
        in_proj = self._build_in_proj()
        bias = self.in_proj_bias
        e = self.embed_dim
        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]
        head_dim = e // self.num_heads

        w_q, w_k, w_v = in_proj.chunk(3)
        b_q = b_k = b_v = None
        if bias is not None:
            b_q, b_k, b_v = bias.chunk(3)

        q = F.linear(query, w_q, b_q).view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = F.linear(key, w_k, b_k).view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = F.linear(value, w_v, b_v).view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)

        if self.bias_k.numel():
            k = k + self.bias_k.repeat(1, bsz, 1)
        if self.bias_v.numel():
            v = v + self.bias_v.repeat(1, bsz, 1)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(0, 1).contiguous().view(tgt_len, bsz, e)
        return self.out_proj(attn_out), None


# ======================================================================
# Model builder
# ======================================================================

def _find_mha(module, prefix=""):
    results = []
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.MultiheadAttention):
            results.append((full, child))
        results.extend(_find_mha(child, full))
    return results


def _find_linears(module, prefix=""):
    results = []
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            results.append((full, child))
        results.extend(_find_linears(child, full))
    return results


def _resolve(model, path):
    obj = model
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _set_mod(model, path, new):
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


def build_voxtell_from_checkpoint(model_dir, device=torch.device("cpu")):
    model_dir = Path(model_dir)
    from voxtell.model.voxtell_model import VoxTellModel

    with open(model_dir / "plans.json", "r", encoding="utf-8") as f:
        plans = json.load(f)
    arch = dict(**plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"])
    for k in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
        if arch[k] is not None:
            arch[k] = pydoc.locate(arch[k])

    net = VoxTellModel(input_channels=1, **arch, decoder_layer=4,
                       text_embedding_dim=2560, num_maskformer_stages=5,
                       num_heads=32, query_dim=2048, project_to_decoder_hidden_dim=2048,
                       deep_supervision=False).to(device)

    ckpt = torch.load(model_dir / "fold_0" / "checkpoint_final.pth", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["network_weights"])
    net.eval()
    return net


# ======================================================================
# Per-experiment apply function
# ======================================================================

def apply_for_exp(model, exp: str, r: int, alpha: int = None):
    """
    Apply DoRA / LoRA to VoxTell according to experiment config.

    exp a: no DoRA/LoRA, only decoder+seg trainable
    exp b: DoRA on cross-attn + FFN + projections (with self-attn)
    exp c: same as b but skip self-attn DoRA
    exp d: same as c with different r
    exp e: same as c with different lr (handled in train.py)
    exp f: same as c but pure LoRA
    exp g: same as c + unfreeze encoder stage 4-5
    """
    if alpha is None:
        alpha = r * 2

    use_dora = (exp != "f")  # experiment F uses pure LoRA
    do_self_attn = exp.startswith("b")  # B and variants (b-r16, b-enc) wrap self-attn
    do_cross_attn = exp not in ("a",)
    do_ffn = exp not in ("a",)
    do_proj = exp not in ("a",)
    unfreeze_enc = (exp == "g" or exp == "b-enc")

    model = copy.deepcopy(model)

    # 1. Freeze everything first. Each experiment then explicitly enables only
    # the intended adapters/modules, keeping the ablation semantics clean.
    for param in model.parameters():
        param.requires_grad_(False)

    # 2. Transformer decoder
    if do_cross_attn:
        for i, layer in enumerate(model.transformer_decoder.layers):
            # Cross-attention (always)
            mha = getattr(layer, "multihead_attn")
            setattr(layer, "multihead_attn",
                    DoraMultiheadAttention(mha, r=r, alpha=alpha, use_dora=use_dora))
            print(f"  [cross-attn] layer.{i}")

            # Self-attention (only experiment B)
            if do_self_attn:
                mha_self = getattr(layer, "self_attn")
                setattr(layer, "self_attn",
                        DoraMultiheadAttention(mha_self, r=r, alpha=alpha, use_dora=use_dora))
                print(f"  [self-attn]  layer.{i}")

    if do_ffn:
        lin_cls = DoraLinear if use_dora else LoraLinear
        for i, layer in enumerate(model.transformer_decoder.layers):
            for ffn_name in ("linear1", "linear2"):
                lin = getattr(layer, ffn_name)
                if isinstance(lin, nn.Linear):
                    setattr(layer, ffn_name, lin_cls(lin, r=r, alpha=alpha))
                    print(f"  [FFN] layer.{i}.{ffn_name}")

    # 3. Projections
    if do_proj:
        lin_cls = DoraLinear if use_dora else LoraLinear
        for prefix in ["project_bottleneck_embed", "project_text_embed", "project_to_decoder_channels"]:
            target = _resolve(model, prefix)
            for path, lin in _find_linears(target):
                _set_mod(target, path, lin_cls(lin, r=r, alpha=alpha))
                print(f"  [proj] {prefix}.{path}")

    # 4. Decoder + seg_layers: full fine-tuning (skip nested encoder ref)
    for name, param in model.decoder.named_parameters():
        param.requires_grad_(not name.startswith("encoder."))

    # 5. decoder_norm frozen
    if hasattr(model, "decoder_norm") and model.decoder_norm is not None:
        for param in model.decoder_norm.parameters():
            param.requires_grad_(False)

    # 6. Unfreeze encoder last 2 stages (experiment G)
    if unfreeze_enc:
        encoder_stages = model.encoder.stages
        for stage_idx in [4, 5]:  # stages 4 and 5
            for param in encoder_stages[stage_idx].parameters():
                param.requires_grad_(True)
        print("  [encoder] unfroze stages 4-5")

    return model


def count_params(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {t:,}  Frozen: {f:,}  Total: {t+f:,}")
    return t, f


# ======================================================================
# Experiment configs
# ======================================================================

EXP_CONFIGS = {
    "a": {"r": 0, "lr": 1e-4, "lr_unfreeze_enc": False},
    "b": {"r": 8, "lr": 1e-4, "lr_unfreeze_enc": False},
    "c": {"r": 8, "lr": 1e-4, "lr_unfreeze_enc": False},
    "d": {"r": 16, "lr": 1e-4, "lr_unfreeze_enc": False},
    "e": {"r": 8, "lr": 5e-4, "lr_unfreeze_enc": False},
    "f": {"r": 8, "lr": 1e-4, "lr_unfreeze_enc": False},
    "g": {"r": 8, "lr": 1e-4, "lr_unfreeze_enc": True},
    # V2: B-based experiments with negative prompt training
    "b-r16": {"r": 16, "lr": 1e-4, "lr_unfreeze_enc": False},
    "b-enc": {"r": 8,  "lr": 1e-4, "lr_unfreeze_enc": True},
}

TRAIN_CFG = {
    "patch_size": (192, 192, 192),
    "batch_size": 1,
    "grad_accum": 4,
    "lr": 1e-4,
    "lr_encoder": 1e-5,  # for experiment G
    "weight_decay": 1e-5,
    "warmup_epochs": 2,    # shorter for 20-epoch runs
    "max_epochs": 10,
    "early_stop_patience": 10,
    "val_interval": 2,
    "prob_fg": 0.5,
    "dice_weight": 1.0,
    "bce_weight": 1.0,
    "amp_dtype": "bfloat16",
    "neg_ratio": 0.1,  # light negative prompt ratio — pretrained model already knows suppression
}

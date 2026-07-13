# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert the NVIDIA PersonaPlex checkpoint (dep_q=16) to HF Personaplex format.

Source : nvidia/personaplex-7b-v1  (original moshi-package layout, model.safetensors)
Target : a local folder loadable by
             PersonaplexForConditionalGeneration.from_pretrained(<out_dir>)

Unlike the stock Moshi conversion, the depth decoder (Depformer) keeps all 16 slices
(agent stream 0..7 + user stream 8..15). Inference with the default generation config
only ever queries slices 0..7 (the depth decoder generates `num_codebooks + 1 = 9`
tokens per frame), which is numerically identical to stock prompted inference; the
user-stream slices stay available for future full-duplex work
(`generate(depth_decoder_max_length=17)`).

A released file may store some Depformer tensors with only 8 slices (the original
PersonaPlex loader expands them at load time, see loaders.py Patch 1/2 in the
reference implementation); `expand_depformer_slices` mirrors that expansion.

PersonaPlex ships a byte-identical Mimi and the same text tokenizer as stock Moshi,
so both are grafted from the `kmhf/hf-moshiko` base checkpoint.

Access requirements: nvidia/personaplex-7b-v1 is a gated repo — accept the license on
the model page and authenticate (`hf auth login` or HF_TOKEN).
Resource notes: ~17 GB download, ~35 GB peak CPU RAM.
"""

from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path

import torch


PERSONAPLEX_REPO = "nvidia/personaplex-7b-v1"
BASE_CHECKPOINT = "kmhf/hf-moshiko"
PERSONAPLEX_DEP_Q = 16  # Depformer slices (agent 8 + user 8)
STOCK_DEP_Q = 8  # stock Moshi Depformer slices (agent only)

# LMGen sampling defaults, inherited unchanged by PersonaPlex (reference lm.py).
TEXT_TEMPERATURE, TEXT_TOP_K = 0.7, 25
AUDIO_TEMPERATURE, AUDIO_TOP_K = 0.8, 250

_RE_SELF_ATTN = re.compile(r"^depformer\.layers\.\d+\.self_attn\.(in_proj_weight|out_proj\.weight)$")
# indexed per-slice groups: (key prefix, number of slices needed for dep_q slices)
_INDEXED_GROUPS = [
    ("depformer_in", PERSONAPLEX_DEP_Q),
    ("linears", PERSONAPLEX_DEP_Q),
    ("depformer_emb", PERSONAPLEX_DEP_Q - 1),
]
_RE_GATING = re.compile(r"^(depformer\.layers\.\d+\.gating)\.(\d+)\.(.+)$")


# ---------------------------------------------------------------------------
# Step 1: Depformer slice expansion (mirror of the reference loaders.py Patch 1/2)
# ---------------------------------------------------------------------------
def expand_depformer_slices(
    state_dict: dict,
    *,
    depformer_dim: int,
    src_slices: int = STOCK_DEP_Q,
    dst_slices: int = PERSONAPLEX_DEP_Q,
    verbose: bool = True,
) -> tuple[dict, list[str]]:
    """Expands a partially-stored Depformer to dst_slices.

    Patch 1: packed self-attention tensors stored with src_slices row-chunks are
    expanded by repeating the agent chunks (reference `copy_missing_weights=True`).
    Patch 2: missing indexed keys `{group}.{i}` for i in [src_slices, dst_slices) are
    filled by copying `{group}.{i - src_slices}`.

    Returns the expanded state dict and the list of keys the reference loader would
    also leave to the model's random init (e.g. `depformer_emb.{src_slices - 1}`,
    which has no 8-slice counterpart).
    """
    expanded, copied = [], []
    for name, tensor in list(state_dict.items()):
        m = _RE_SELF_ATTN.match(name)
        if m is None:
            continue
        rows_per_slice = 3 * depformer_dim if m.group(1) == "in_proj_weight" else depformer_dim
        n_slices, rem = divmod(tensor.shape[0], rows_per_slice)
        if rem != 0 or n_slices not in (src_slices, dst_slices):
            raise ValueError(
                f"{name}: unexpected shape {tuple(tensor.shape)} for depformer_dim={depformer_dim} "
                f"(expected {src_slices} or {dst_slices} slices of {rows_per_slice} rows)"
            )
        if n_slices == src_slices:
            state_dict[name] = torch.cat([tensor, tensor], dim=0)
            expanded.append(name)

    # Collect existing indices per indexed group (incl. per-layer gating groups).
    groups: dict[str, dict[int, str]] = {}
    for name in list(state_dict.keys()):
        m = _RE_GATING.match(name)
        if m is not None:
            groups.setdefault(f"{m.group(1)}|{m.group(3)}", {})[int(m.group(2))] = name
            continue
        for prefix, _ in _INDEXED_GROUPS:
            m = re.match(rf"^{prefix}\.(\d+)\.(.+)$", name)
            if m is not None:
                groups.setdefault(f"{prefix}|{m.group(2)}", {})[int(m.group(1))] = name

    left_missing = []
    for group_key, by_index in groups.items():
        prefix, suffix = group_key.split("|")
        needed = PERSONAPLEX_DEP_Q - 1 if prefix == "depformer_emb" else PERSONAPLEX_DEP_Q
        for i in range(needed):
            if i in by_index:
                continue
            src = i - src_slices
            new_name = f"{prefix}.{i}.{suffix}"
            if src in by_index:
                state_dict[new_name] = state_dict[by_index[src]].clone()
                copied.append(new_name)
            else:
                left_missing.append(new_name)

    if verbose:
        print(
            f"[expand] row-expanded {len(expanded)} packed attention tensors, "
            f"copied {len(copied)} slice tensors, left to random init: {left_missing or 'none'}"
        )
        if not expanded and not copied:
            print(f"[expand] checkpoint already has {dst_slices} Depformer slices; nothing to do")
    return state_dict, left_missing


# ---------------------------------------------------------------------------
# Step 2: original-layout -> HF-layout conversion
# Vendored from the official Moshi conversion (Apache-2.0, The HuggingFace Inc.
# team), with the Depformer loops driven by the *depth decoder's* num_codebooks
# (16) instead of the main config's (8). Do not "clean up": the replacement list
# is order-sensitive and intentionally cascades.
# ---------------------------------------------------------------------------

convert_list = [
    # GENERAL
    ("out_norm", "decoder.model.norm"),
    ("depformer_emb", "depth_decoder.emb"),
    ("depformer_text_emb", "depth_decoder.text_emb"),
    ("text_emb", "decoder.model.emb"),
    ("emb", "embed_tokens"),
    ("text_linear", "decoder.lm_head"),
    ("depformer", "depth_decoder"),
    ("transformer", "decoder.model"),
    # TRANSFORMERS PART
    ("gating.linear_in", "mlp.fc1"),
    ("gating.linear_out", "mlp.fc2"),
    ("self_attn.out_proj", "self_attn.o_proj.linear"),
    ("norm1", "input_layernorm"),
    ("norm2", "post_attention_layernorm"),
    ("layer_scale_1", "self_attn_layer_scale"),
    ("layer_scale_2", "mlp_layer_scale"),
    ("alpha", "weight"),
]


def _preprocess_state_dict(state_dict, config):
    # Moshi original weights are using a gating mechanism
    dep_q = config.depth_decoder_config.num_codebooks

    for layer_idx in range(config.depth_decoder_config.num_hidden_layers):
        linear_layers_in = [
            state_dict.pop(f"depformer.layers.{layer_idx}.gating.{i}.linear_in.weight") for i in range(dep_q)
        ]
        linear_layers_out = [
            state_dict.pop(f"depformer.layers.{layer_idx}.gating.{i}.linear_out.weight") for i in range(dep_q)
        ]

        state_dict[f"depth_decoder.layers.{layer_idx}.mlp.fc1.weight"] = torch.stack(linear_layers_in)
        state_dict[f"depth_decoder.layers.{layer_idx}.mlp.fc2.weight"] = torch.stack(linear_layers_out)

    input_projections = []
    lm_heads = []
    for codebook_idx in range(dep_q):
        input_projections.append(state_dict.pop(f"depformer_in.{codebook_idx}.weight"))
        lm_heads.append(state_dict.pop(f"linears.{codebook_idx}.weight"))

    state_dict["depth_decoder.input_projections.weight"] = torch.stack(input_projections, dim=0)
    state_dict["depth_decoder.lm_heads.weight"] = torch.stack(lm_heads, dim=0)

    return state_dict


def _convert_model(
    state_dict,
    hf_model,
    convert_list,
    device,
    config,
    unwanted_prefix=None,
    allowed_missing=(),
):
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_heads = int(config.hidden_size // config.head_dim)
    num_key_value_heads = config.num_key_value_heads
    key_value_head_dim = config.num_key_value_heads * head_dim
    dep_q = config.depth_decoder_config.num_codebooks

    state_dict = _preprocess_state_dict(state_dict, config)

    # permute for sliced rotary
    def permute(w, n_heads, dim1=hidden_size, dim2=hidden_size):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for k, v in list(state_dict.items()):
        if "audio_encoder" not in k:
            new_k = k if unwanted_prefix is None else k[len(unwanted_prefix) :]
            for old_layer_name, new_layer_name in convert_list:
                if old_layer_name in new_k:
                    new_k = new_k.replace(old_layer_name, new_layer_name)

            if "alpha" in k:
                state_dict[k] = state_dict[k].squeeze()

            if "in_proj_weight" in new_k:
                # split qkv into query key and value
                mixed_qkv = state_dict.pop(k)
                if "depth_decoder" in new_k:
                    mixed_qkv = mixed_qkv.view(dep_q, -1, mixed_qkv.shape[-1])

                    qkv_dim = mixed_qkv.size(1) // 3

                    query_layer = mixed_qkv[:, :qkv_dim]
                    key_layer = mixed_qkv[:, qkv_dim : qkv_dim * 2]
                    value_layer = mixed_qkv[:, qkv_dim * 2 :]
                    state_dict[new_k.replace("in_proj_weight", "q_proj.linear.weight")] = query_layer
                    state_dict[new_k.replace("in_proj_weight", "k_proj.linear.weight")] = key_layer

                else:
                    qkv_dim = mixed_qkv.size(0) // 3

                    query_layer = mixed_qkv[:qkv_dim]
                    key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
                    value_layer = mixed_qkv[qkv_dim * 2 :]
                    state_dict[new_k.replace("in_proj_weight", "q_proj.linear.weight")] = permute(
                        query_layer, num_heads, hidden_size, hidden_size
                    )
                    state_dict[new_k.replace("in_proj_weight", "k_proj.linear.weight")] = permute(
                        key_layer, num_key_value_heads, key_value_head_dim, hidden_size
                    )

                state_dict[new_k.replace("in_proj_weight", "v_proj.linear.weight")] = value_layer
            elif "o_proj" in new_k and "depth_decoder" in new_k:
                output_layer = state_dict.pop(k)
                state_dict[new_k] = output_layer.view(dep_q, -1, output_layer.shape[-1])
            else:
                state_dict[new_k] = state_dict.pop(k)

    # Do the last one by hand
    state_dict["depth_decoder.text_embed_tokens.weight"] = state_dict.pop(
        "depth_decoder.decoder.model.embed_tokens.weight"
    )

    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if missing_keys - set(allowed_missing):
        raise ValueError(f"missing keys: {missing_keys - set(allowed_missing)}")
    hf_model.load_state_dict(state_dict, strict=not missing_keys)

    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


# ---------------------------------------------------------------------------
# Step 3: end-to-end conversion
# ---------------------------------------------------------------------------


def _download_from_hub(repo: str, filename: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo, filename, token=token))
    except Exception as exc:  # gated repo, missing token, network, ...
        raise SystemExit(
            f"Could not download {filename} from {repo}: {exc}\n"
            f"This repo is gated. Fix:\n"
            f"  1) accept the license at https://huggingface.co/{repo}\n"
            f"  2) authenticate: `hf auth login` (or pass --token / set HF_TOKEN)"
        ) from exc


def _load_personaplex_state_dict(path: Path) -> dict:
    from safetensors import safe_open

    state_dict = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    # A training snapshot may nest the weights (mirrors the official script).
    if "best_state" in state_dict:
        state_dict = state_dict["best_state"]
    return state_dict


def _hf_missing_keys(allowed_original_missing: list[str], convert_list) -> list[str]:
    """Maps original-layout key names the loader leaves to random init (e.g.
    `depformer_emb.7.weight`) onto their HF names so the strict check can allow them."""
    hf_names = []
    for name in allowed_original_missing:
        for old, new in convert_list:
            if old in name:
                name = name.replace(old, new)
        hf_names.append(name)
    return hf_names


def convert(
    personaplex_file: Path,
    out_dir: Path,
    base_checkpoint: str = BASE_CHECKPOINT,
    dry_run: bool = False,
) -> None:
    from transformers import AutoFeatureExtractor, AutoTokenizer, MoshiForConditionalGeneration
    from transformers.models.personaplex.configuration_personaplex import PersonaplexConfig
    from transformers.models.personaplex.modeling_personaplex import PersonaplexForConditionalGeneration

    print(f"[1/6] reading PersonaPlex weights: {personaplex_file}")
    state_dict = _load_personaplex_state_dict(personaplex_file)
    print(f"      {len(state_dict)} tensors")

    # PersonaPlex keeps moshiko's architecture; the PersonaplexConfig defaults *are*
    # the moshiko 7B hyperparameters, with the depth decoder at dep_q=16.
    config = PersonaplexConfig()
    assert config.depth_decoder_config.num_codebooks == PERSONAPLEX_DEP_Q

    print("[2/6] expanding partially-stored Depformer slices (no-op on a full checkpoint)")
    state_dict, left_missing = expand_depformer_slices(
        state_dict, depformer_dim=config.depth_decoder_config.hidden_size
    )
    if dry_run:
        print("[dry-run] stopping before model materialization")
        return

    print(f"[3/6] building Personaplex skeleton + grafting Mimi from {base_checkpoint}")
    try:
        base = MoshiForConditionalGeneration.from_pretrained(base_checkpoint, dtype=torch.bfloat16)
    except TypeError:  # older transformers uses torch_dtype=
        base = MoshiForConditionalGeneration.from_pretrained(base_checkpoint, torch_dtype=torch.bfloat16)
    model = PersonaplexForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
    # PersonaPlex ships the byte-identical Mimi file as stock Moshi (same sha256).
    state_dict.update({f"audio_encoder.{k}": v for k, v in base.audio_encoder.state_dict().items()})
    del base

    print("[4/6] converting layout (official HF mapping, strict key check)")
    with torch.no_grad():
        _convert_model(
            state_dict,
            model,
            convert_list,
            "cpu",
            config,
            allowed_missing=_hf_missing_keys(left_missing, convert_list),
        )

    print("[5/6] writing generation defaults")
    model.generation_config.do_sample = True
    model.generation_config.temperature = TEXT_TEMPERATURE
    model.generation_config.top_k = TEXT_TOP_K
    # The depth decoder must keep generating `num_codebooks + 1 = 9` tokens per frame
    # (agent slices only); these mirror the defaults `generate()` would otherwise build.
    model.generation_config.depth_decoder_config = {
        "min_length": config.num_codebooks + 1,
        "max_length": config.num_codebooks + 1,
        "cache_implementation": "static",
        "do_sample": True,
        "temperature": AUDIO_TEMPERATURE,
        "top_k": AUDIO_TOP_K,
    }

    print(f"[6/6] saving to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(base_checkpoint).save_pretrained(str(out_dir))
    try:  # optional
        AutoFeatureExtractor.from_pretrained(base_checkpoint).save_pretrained(str(out_dir))
    except Exception as exc:
        print(f"feature extractor not saved (optional): {exc}")

    print(
        "done. Use it with:\n"
        "  model = PersonaplexForConditionalGeneration.from_pretrained(out_dir)\n"
        "  inputs = model.build_persona_prompt(voice_input_values=..., persona_input_ids=...)\n"
        "  model.generate(**inputs, max_new_tokens=125)"
    )


def _download_voices(out_dir: Path, token: str | None) -> None:
    """Fetch the official PersonaPlex voice-prompt library (voices.tgz)."""
    tgz = _download_from_hub(PERSONAPLEX_REPO, "voices.tgz", token)
    dest = out_dir / "voices"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz) as tar:
        try:
            tar.extractall(dest, filter="data")  # Python >= 3.12 safe filter
        except TypeError:
            tar.extractall(dest)
    print(f"voice prompts extracted to {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--personaplex-file",
        default=None,
        help="local path to PersonaPlex model.safetensors "
        "(default: download from the hub, requires accepted license + token)",
    )
    p.add_argument("--repo", default=PERSONAPLEX_REPO)
    p.add_argument("--base", default=BASE_CHECKPOINT)
    p.add_argument("--out", default="personaplex-hf")
    p.add_argument("--token", default=None, help="HF access token (else cached login / HF_TOKEN)")
    p.add_argument(
        "--download-voices",
        action="store_true",
        help="also download and extract the official voices.tgz into <out>/voices",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="stop after the expansion report (no conversion, no save)",
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    src = (
        Path(args.personaplex_file)
        if args.personaplex_file
        else _download_from_hub(args.repo, "model.safetensors", args.token)
    )
    convert(src, out_dir, base_checkpoint=args.base, dry_run=args.dry_run)
    if args.download_voices and not args.dry_run:
        _download_voices(out_dir, args.token)


if __name__ == "__main__":
    main()

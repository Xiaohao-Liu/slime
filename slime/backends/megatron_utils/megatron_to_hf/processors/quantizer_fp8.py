import re

import torch

from slime.backends.megatron_utils.kernels.fp8_kernel import blockwise_cast_to_fp8_triton

from ...sglang import quant_weight_ue8m0, should_deepgemm_weight_requant_ue8m0, transform_scale_ue8m0


def quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "fp8"
    fmt = quantization_config.get("fmt", "e4m3")
    assert fmt == "e4m3", f"Unsupported FP8 format: {fmt}"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)
    # SGLang's FP8 loader honors both keys; mirror that here so the on-the-fly
    # quantizer keeps the same module set as the static checkpoint.
    modules_to_not_convert = (
        quantization_config.get("modules_to_not_convert")
        or quantization_config.get("ignored_layers")
        or []
    )

    # Accept the Qwen3-VL prefix (`language_model.`) in addition to the plain
    # decoder prefix so the LM weights of multimodal models are quantized too.
    decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
        match = re.match(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            return _quantize_named_params(
                converted_named_params,
                weight_block_size,
                modules_to_not_convert,
                drop_existing_scales=True,
            )

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            return _quantize_named_params(
                converted_named_params,
                weight_block_size,
                modules_to_not_convert,
            )

    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        # indexer
        "self_attention.wq_b.weight",
        "self_attention.wk.weight",
        # linear attention
        "self_attention.linear_attn.in_proj_qkv.weight",
        "self_attention.linear_attn.in_proj_z.weight",
        "self_attention.linear_attn.out_proj.weight",
    ]:
        return _quantize_named_params(
            converted_named_params,
            weight_block_size,
            modules_to_not_convert,
        )

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_named_params(
    converted_named_params,
    weight_block_size,
    modules_to_not_convert,
    *,
    drop_existing_scales=False,
):
    out = []
    for converted_name, param in converted_named_params:
        # skip bf16 weight_scale / input_scale that some converters emit;
        # we will regenerate the scale during quantization below.
        if drop_existing_scales and converted_name.endswith("_scale"):
            continue
        if _is_module_skipped(converted_name, modules_to_not_convert):
            # Module is in the FP8 checkpoint's not-convert list; pass it
            # through as BF16 so the dtype matches what SGLang expects.
            out.append((converted_name, param))
            continue
        out.extend(_quantize_param(converted_name, param, weight_block_size))
    return out


def _is_module_skipped(weight_name, modules_to_not_convert):
    if not modules_to_not_convert:
        return False
    base = weight_name
    for suffix in (".weight_scale_inv", ".weight_scale", ".weight", ".bias"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for entry in modules_to_not_convert:
        if base == entry or base.startswith(entry + "."):
            return True
    return False


def _quantize_param(name, weight, weight_block_size):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        if should_deepgemm_weight_requant_ue8m0 and should_deepgemm_weight_requant_ue8m0(
            weight_block_size=weight_block_size
        ):
            qweight, scale = quant_weight_ue8m0(weight, weight_block_size=weight_block_size)
            scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
        else:
            qweight, scale = blockwise_cast_to_fp8_triton(weight, weight_block_size)
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # per tensor quant
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]

# Qwen3.5 (`qwen3_5`)

TPU model directory for the Qwen3.5 MoE checkpoints, starting with
[`Qwen/Qwen3.5-397B-A17B-FP8`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
(`Qwen3_5MoeForConditionalGeneration`), served prefill/decode disaggregated
for the InferenceX (IX) points.

- **Support tier**: `Experimental`
- **Owners**: Shrihari Murali (directory, settings, config hook, validation)
- **Tracking issue**: to be linked when this directory's first PR opens

The native model still lives in `models/vllm/` (`qwen3_5_patch.py`,
`qwen3_5_mtp.py`, and `qwen3_next_patch.py`, which Qwen3.5 shares with
Qwen3-Next).

## Directory Layout

- `settings.py`: model-scoped settings, read from
  `--additional-config '{"qwen3_5": {...}}'`. A setting is added by the change
  that implements it; unknown keys fail at engine start.
- `hooks.py`: the config hook registered in vLLM's `MODELS_CONFIG_MAP`. It
  keeps vLLM's own Qwen3.5 checks, validates the settings, and fills TPU
  serving defaults the user left unset (see its docstring).
- `validation/`: the serving entries of the IX points and the accuracy gates
  (see `validation/README.md`).
- `kernels/`: Qwen3.5-only kernels and model-owned copies of shared kernels
  (each copy records its source path and commit in its header). Added by later
  changes.
- `megakernel/`: reserved for the Qwen3.5 decode megakernel, which is
  developed in its own lane. Its layout and ownership are agreed with that
  lane before anything lands here; nothing in this directory imports or calls
  it.

## Settings

| Setting | Type | Default | What it does |
| --- | --- | --- | --- |

No setting exists yet. Each change that adds one adds its row here.

## Coordination

Changes that need hooks in shared core directories (`runner/`,
`platforms/`, `layers/`, `core/`, `offload/`, `distributed/`) land as
standalone shared changes; this directory holds only Qwen3.5-specific code.

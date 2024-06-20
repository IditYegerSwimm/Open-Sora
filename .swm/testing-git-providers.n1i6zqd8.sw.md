---
title: Testing git providers
---
<SwmSnippet path="/scripts/inference.py" line="37">

---

&nbsp;

&nbsp;

This code snippet sets the gradient calculation to be disabled. It then checks if CUDA is available and sets the device accordingly. It also checks the dtype and sets it to torch dtype. Finally, it allows tf32 for CUDA matrix multiplication and cudnn operations.

```python
def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

---

</SwmSnippet>

<SwmSnippet path="/scripts/inference.py" line="79">

---

This code snippet prepares the video size by checking if `image_size` is provided in the `cfg` dictionary. If not, it checks if `resolution` and `aspect_ratio` are provided, and if so, it calculates the `image_size` using the `get_image_size` function. Finally, it gets the number of frames using the `get_num_frames` function.

```python
    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)
```

---

</SwmSnippet>

<SwmSnippet path="/scripts/inference.py" line="93">

---

This code snippet builds a model by calling the `build_module` function with various arguments. The `cfg.model` argument specifies the model type, `MODELS` is a dictionary of available models. Other arguments include `latent_size`, `vae.out_channels`, <SwmToken path="/scripts/inference.py" pos="99:3:5" line-data="            caption_channels=text_encoder.output_dim,">`text_encoder.output_dim`</SwmToken>, `text_encoder.model_max_length`, and `enable_sequence_parallelism`.

```python
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
```

---

</SwmSnippet>

<SwmSnippet path="/scripts/inference.py" line="297">

---

This code snippet updates the value of <SwmToken path="/scripts/inference.py" pos="297:1:1" line-data="        start_idx += len(batch_prompts)">`start_idx`</SwmToken> by adding the length of <SwmToken path="/scripts/inference.py" pos="153:1:1" line-data="        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)">`batch_prompts`</SwmToken>. It then logs an informational message indicating that the inference has finished, and another message indicating the number of samples saved to a specific directory.

&nbsp;

```python
        start_idx += len(batch_prompts)
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)
```

---

</SwmSnippet>

<SwmToken path="/scripts/inference.py" pos="116:1:1" line-data="    start_idx = cfg.get(&quot;start_index&quot;, 0)">`start_idx`</SwmToken>

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBT3Blbi1Tb3JhJTNBJTNBSWRpdFllZ2VyU3dpbW0=" repo-name="Open-Sora"><sup>Powered by [Swimm](https://staging.swimm.cloud/)</sup></SwmMeta>

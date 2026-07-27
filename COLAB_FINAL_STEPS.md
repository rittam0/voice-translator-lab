# Archived Colab experiment

`notebooks/FINAL_GPU_ACCEPTANCE.ipynb` is retained as an experiment record, not
as a supported one-click notebook.

The combined environment is known to fail dependency resolution:

- `qwen-tts==0.0.5` requires `transformers==4.57.3`;
- `nemo_toolkit[asr]==2.2.1` requires
  `transformers>=4.48.0,<=4.48.3`.

The Odia/NeMo path therefore requires a separate process and virtual
environment. That isolation was designed in the source but was not completed
and accepted before the experiment stopped. Colab GPU quota and temporary
tunnel availability are additional external constraints.

For repository verification, use the lightweight commands in `README.md`.
They test orchestration without downloading model weights. Do not interpret
those tests as speech-quality evidence.

Anyone continuing the real-model work should:

1. split Qwen and NeMo into isolated services;
2. replace the temporary Cloudflare tunnel with authenticated infrastructure;
3. use only recordings from consenting speakers;
4. capture machine-readable artifacts for every claimed metric;
5. review every model license before deployment.

No personal recording, model checkpoint, generated audio, secret, or temporary
acceptance artifact is committed here.

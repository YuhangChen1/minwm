# Full Duplex Training Inspector

Start the local monitor from the repository root:

```bash
python -m Full_Duplex_Fix.debug.server --host 0.0.0.0 --port 8765
```

Opening `http://127.0.0.1:8765` starts one isolated debug run. The run is fixed to
`cuda:2`, five optimizer steps, the configured cached sample, no W&B, and no model
checkpoint serialization. Output is written below `Full_Duplex_Fix/debug/runs/`.

The page starts in auto mode with a 50 ms event delay. It exposes next, auto,
pause-at-next, stop, and reset controls; pause-at-next switches to interactive step
mode. Use `?autostart=0` to inspect the page without creating a training thread.

Events cover cached inputs, flow corruption, interleaved layout construction,
patch/time/text/camera encoders, RoPE, PRoPE, the shared attention mask, every
Transformer sub-stage, the output head, weighted flow loss, backward tensor hooks,
post-clipping gradients, and AdamW parameter updates. Tensor statistics are sampled;
shapes and the first two values are exact. Parameter tables report flattened first
two values before/after the optimizer update.

Normal training keeps debugging disabled. `train_overfit.py --debug-mode` enables
the same instrumentation in non-interactive auto mode and writes `debug_events.jsonl`
to the configured training output directory.

# TLM q/v LoRA

The built-in TLM consumer follows the implementation family in the upstream
TLM repository: PEFT LoRA is attached to every `q_proj` and `v_proj`, with
rank 8, alpha 16, dropout 0, and learning rate `5e-5` by default.

This is deliberately separate from fast weight. TLM performs a normal
full-model causal-LM forward/backward in a Hugging Face/PEFT side model and
trains persistent low-rank parameters. The result is an immutable LoRA
version registered through vLLM's native LoRA manager. A request pins the
latest version when it is admitted and keeps that version until completion;
an update is visible only to later requests.

Fast weight instead provides request-keyed, short-lived effective weights
inside an inference forward. Its update rule, tail state, and timing are owned
by the overlay provider. The reusable part is only module resolution,
capability admission, per-request context, and wrapper lifecycle. TLM does not
need that mechanism because vLLM already has a native LoRA execution path.

Install the optional trainer dependency with `pip install -e '.[tlm]'`, call
`bind_hf_model()` with the matching base causal LM, and construct vLLM with
native LoRA enabled and `max_lora_rank` at least the configured rank. The
current publication path is in-process; a multiprocess deployment must
transport immutable versions to each worker before admission.

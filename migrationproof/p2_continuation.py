from copy import deepcopy


def build_continuation_config(
    source,
    *,
    load,
    save,
    resume_iteration,
    additional_iterations,
    learning_rate,
    effective_batch_size,
):
    if resume_iteration < 0 or additional_iterations <= 0:
        raise ValueError("continuation iterations must be positive")
    if effective_batch_size <= 0 or effective_batch_size % 2:
        raise ValueError("effective batch size must be a positive multiple of 2")
    config = deepcopy(source)
    config["parallel"]["fsdp_plan"].update(
        {"num_to_forward_prefetch": 1, "num_to_backward_prefetch": 1}
    )
    config["features"]["recompute"] = False
    target_iteration = resume_iteration + additional_iterations
    config["training"].update(
        {
            "load": load,
            "save": save,
            "train_iters": target_iteration,
            "save_interval": target_iteration,
            "lr": learning_rate,
            "gradient_accumulation_steps": effective_batch_size // 2,
            "val_interval": 0,
        }
    )
    return config

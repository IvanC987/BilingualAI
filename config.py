config = {
    "en_ds_path":     "dataset/train_en_f10.txt",    # English dataset path
    "zh_ds_path":     "dataset/train_zh_f10.txt",    # Chinese dataset path
    "compile_model":  False,                         # Whether or not to use torch.compile
    "load_model":     False,                         # If to load in model from ckpt

    "batch_size":     384,                           # Training batch size
    "seq_len":        256,                           # Based on the training data (effective seq_len = 192)
    "n_embd":         512,                           # Embedding dimension
    "n_heads":        8,                             # Number of attention heads
    "n_layers":       4,                             # Number of transformer layers
    "dropout":        0.1,                           # Dropout rate

    "n_epochs":       5,                             # Number of training epochs
    "eval_interval":  256,                           # Evaluate the model at every x steps
    "warmup_steps":   1500,                          # Number of warmup steps to take before reaching base_lr

    "output_path":    "output_logs.txt",             # Filepath for logs
    "ckpt_dir":       "ckpt_dir",                    # Directory where model/optimizer params would be saved
    "ckpt_epoch":     1,                             # Save model checkpoint every 1 epoch, adjust as needed
}

import os
import torch
from transformer import Transformer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}")


pth_path = "ckpt_dir/ep_10_0.4110_0.3586.pth"
ckpt_dict = torch.load(pth_path, map_location=device, weights_only=True)


model_configs = ckpt_dict["config"]
model_state_dict = ckpt_dict["model_state_dict"]
str_to_int = ckpt_dict["str_to_int"]
int_to_str = {v: k for k, v in str_to_int.items()}


model = Transformer(vocab_size=len(str_to_int),
                    seq_len=model_configs["seq_len"],
                    n_embd=model_configs["n_embd"],
                    n_heads=model_configs["n_heads"],
                    n_layers=model_configs["n_layers"],
                    dropout=model_configs["dropout"],
                    device=device
                    ).to(device)

model.load_state_dict(model_state_dict)

seq_len = model_configs["seq_len"]
sos_token = "<SOS>"
eos_token = "<EOS>"
pad_idx = str_to_int["<PAD>"]


def batch_tokenize(batch_sentences: list[str], add_sos: bool, add_eos: bool, check_unk_chars=False):
    if check_unk_chars:  # This should only be set to true when tokenizing inputs for model inference
        for char in set("".join(batch_sentences)):
            # Checking for existence of all characters in provided batch
            if char not in str_to_int:
                return f"Invalid character found: '{char}'"

    tokens = []
    for sentence in batch_sentences:
        result = [str_to_int[c] for c in sentence]
        if add_sos:
            result.insert(0, str_to_int[sos_token])
        if add_eos:
            result.append(str_to_int[eos_token])
        if len(result) < seq_len:
            result.extend([pad_idx] * (seq_len - len(result)))

        result = result[:seq_len]
        assert len(result) == seq_len

        tokens.append(result)

    return torch.tensor(tokens, dtype=torch.long)


print()
print("This is Bilingual AI, feel free to test it out.")
print("Type 'exit()' to exit and 'clear' to clear screen")
print("-------------")
while True:
    user = input("\n>>> ")

    if user.lower() == "exit()":
        exit()
    elif user.lower() == "clear":
        os.system("clear" if os.name == "posix" else "cls")
        continue

    try:
        txt_tokens = batch_tokenize([user], True, False, True).to(device)
        prediction = model.translate(txt_tokens, sos_token=str_to_int[sos_token], eos_token=str_to_int[eos_token])
        translation = "".join([int_to_str[i] for i in prediction.tolist()[0]])

        translation = translation[5:]  # Trim <SOS>
        if translation.endswith("<EOS>"):  # Trim <EOS>
            translation = translation[:-5]

        print(f">>> {translation}\n")
    except Exception as e:
        print(f">>> Exception {e} has occurred. Likely invalid character(s) were submitted\n")


import jieba
import torch
import nltk
from nltk.translate.meteor_score import meteor_score
from transformer import Transformer

nltk.download('wordnet')


with open("dataset/test_en_f10.txt", "r", encoding="utf-8") as f:
    src_sentences = [line.strip() for line in f if line.strip()]

with open("dataset/test_zh_f10.txt", "r", encoding="utf-8") as f:
    tgt_sentences = [line.strip() for line in f if line.strip()]

assert len(src_sentences) == len(tgt_sentences), f"Mismatch: {len(src_sentences)} source vs {len(tgt_sentences)} target"

n = 10  # Take first n sentences (For a quick test, use whole test set for more thorough testing)
src_sentences = src_sentences[:n]
tgt_sentences = tgt_sentences[:n]

print(f"Length of testing dataset: {len(src_sentences)}")


device = "cuda" if torch.cuda.is_available() else "cpu"
pth_path = "ckpt_dir/ep_10_0.4110_0.3586.pth"
ckpt_dict = torch.load(pth_path, map_location=device, weights_only=True)


model_configs = ckpt_dict["config"]
model_state_dict = ckpt_dict["model_state_dict"]
str_to_int = ckpt_dict["str_to_int"]
int_to_str = {v: k for k, v in str_to_int.items()}


model = Transformer(
    vocab_size=len(str_to_int),
    seq_len=model_configs["seq_len"],
    n_embd=model_configs["n_embd"],
    n_heads=model_configs["n_heads"],
    n_layers=model_configs["n_layers"],
    dropout=model_configs["dropout"],
    device=device
).to(device)

model.load_state_dict(model_state_dict)
model.eval()


seq_len = model_configs["seq_len"]
sos_token = "<SOS>"
eos_token = "<EOS>"
pad_idx = str_to_int["<PAD>"]


def batch_tokenize(batch_sentences: list[str], add_sos: bool, add_eos: bool, check_unk_chars=False):
    if check_unk_chars:
        for char in set("".join(batch_sentences)):
            if char not in str_to_int:
                raise ValueError(f"Invalid character found: '{char}'")

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
        tokens.append(result)

    return torch.tensor(tokens, dtype=torch.long)



print("Starting evaluation...")
hypotheses = []
references = []

with torch.no_grad():
    for idx, (source, target) in enumerate(zip(src_sentences, tgt_sentences)):
        try:
            txt_tokens = batch_tokenize([source], add_sos=True, add_eos=False, check_unk_chars=True).to(device)
            prediction = model.translate(txt_tokens, sos_token=str_to_int[sos_token], eos_token=str_to_int[eos_token])
            translation = "".join([int_to_str[i] for i in prediction.tolist()[0]])

            if translation.startswith("<SOS>"):  # Should always be the case
                translation = translation[5:]
            if translation.endswith("<EOS>"):  # Should be the case unless len exceeded
                translation = translation[:-5]

            hypotheses.append(translation)
            references.append([target.strip()])  # wrap in list for METEOR format
        except Exception as e:
            print(f"\nException {e} occurred\n")

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(src_sentences)}")


# Tokenize for METEOR
tokenized_hypotheses = [list(jieba.cut(h)) for h in hypotheses]
tokenized_references = [[list(jieba.cut(r)) for r in ref_list] for ref_list in references]


# Compute METEOR
scores = [meteor_score(refs, hyp) for refs, hyp in zip(tokenized_references, tokenized_hypotheses)]
average_score = sum(scores) / len(scores)
print(f"\nAverage METEOR score: {average_score:.4f}")


print("\n--- Sample Translations ---")
for i in range(3):
    print(f"Source    : {src_sentences[i]}")
    print(f"Hypothesis: {hypotheses[i]}")
    print(f"Reference : {tgt_sentences[i]}")
    print()

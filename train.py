import torch
from torch import nn
import random
import math
import time
import os
from transformer import Transformer
from dataset_loader import DatasetLoader
from config import config


# Setting to 'high' uses TF32 rather than FP32, which makes the training process faster (varies on machines)
# Can set to 'medium' for even faster training, though will be loss in performance
# Check out the documentations https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}")

print("\nConfigurations:")
print("-" * 20)
for k, v in config.items():
    print(f"{k}: {v}")
print("-" * 20)
print("")


confirmation = input("Confirm above hyperparameters [Y/N]: ")
if confirmation.lower() != "y":
    exit()



en_ds_path    = config["en_ds_path"]
zh_ds_path    = config["zh_ds_path"]
compile_model = config["compile_model"]
load_model    = config["load_model"]

batch_size    = config["batch_size"]
seq_len       = config["seq_len"]
n_embd        = config["n_embd"]
n_heads       = config["n_heads"]
n_layers      = config["n_layers"]
dropout       = config["dropout"]

n_epochs      = config["n_epochs"]
eval_interval = config["eval_interval"]
warmup_steps  = config["warmup_steps"]

output_path   = config["output_path"]
ckpt_dir      = config["ckpt_dir"]
ckpt_epoch    = config["ckpt_epoch"]



os.makedirs(ckpt_dir, exist_ok=True)  # Create the checkpointing directory to hold intermediate model/optimizer state dict and config

if os.path.exists(output_path):
    os.remove(output_path)  # Remove old output file if exists


SOS_TOKEN = "<SOS>"
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"


dataset_loader = DatasetLoader(src_ds_path=en_ds_path,
                               tgt_ds_path=zh_ds_path,
                               sos_token=SOS_TOKEN,
                               eos_token=EOS_TOKEN,
                               pad_token=PAD_TOKEN,
                               batch_size=batch_size,
                               seq_len=seq_len,
                               train_split=0.995)


@torch.no_grad()
def eval_model():
    vl = []
    model.eval()

    val_epoch = dataset_loader.val_epoch
    while dataset_loader.val_epoch == val_epoch:
        x, y = dataset_loader.get_batch(train=False)
        x_train = dataset_loader.batch_tokenize(x, False, False).to(device)
        y_train = dataset_loader.batch_tokenize(y, True, True).to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            if compile_model:
                y_pred = compiled_model(x_train, y_train)
            else:
                y_pred = model(x_train, y_train)

            y_label = dataset_loader.batch_tokenize(y, False, True).to(device)
            B, T, C = y_pred.shape
            loss = criterion(y_pred.view(B*T, C), y_label.view(B*T))


        vl.append(loss.item())

    model.train()
    return vl



str_to_int = dataset_loader.str_to_int
int_to_str = dataset_loader.int_to_str

model = Transformer(vocab_size=len(str_to_int),
                    seq_len=seq_len,
                    n_embd=n_embd,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    dropout=dropout,
                    device=device
                    ).to(device)

model.train()




if compile_model:
    compiled_model = torch.compile(model)


# Print the number of parameters in the model
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

criterion = nn.CrossEntropyLoss(ignore_index=str_to_int[PAD_TOKEN])
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9)



start_epoch = 0
# Training data needs to match when resume training
if load_model:
    while True:
        model_path = input("Enter model path: ")
        if os.path.exists(model_path) and model_path.endswith(".pth"):
            break
        else:
            print(f"{model_path=} does not exist!\n")
    print("Now loading model...")

    ckpt_dict = torch.load(model_path, map_location=device, weights_only=True)
    model_state_dict = ckpt_dict["model_state_dict"]
    optimizer_state_dict = ckpt_dict["optimizer_state_dict"]

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    start_epoch = ckpt_dict["train_epoch"]
else:
    print("Training from scratch...")



steps = 0
start = time.time()
train_losses = []
val_losses = []
for epoch in range(start_epoch, start_epoch + n_epochs):
    print("*" * 20)
    print(f"Currently in epoch {epoch+1}")
    print("*" * 20)

    train_epoch = dataset_loader.train_epoch
    while dataset_loader.train_epoch == train_epoch:
        x, y = dataset_loader.get_batch(train=True)
        x_train = dataset_loader.batch_tokenize(x, False, False).to(device)
        y_train = dataset_loader.batch_tokenize(y, True, True).to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            if compile_model:
                y_pred = compiled_model(x_train, y_train)
            else:
                y_pred = model(x_train, y_train)

            y_label = dataset_loader.batch_tokenize(y, False, True).to(device)
            B, T, C = y_pred.shape
            loss = criterion(y_pred.view(B*T, C), y_label.view(B*T))

        # Update learning rate
        lr = (n_embd ** -0.5) * min((steps+1) ** -0.5, (steps+1) * (warmup_steps ** -1.5))
        lr *= 1.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if steps % eval_interval == 0:
            v_losses = eval_model()
            val_loss = sum(v_losses) / len(v_losses)
            val_losses.append(val_loss)

            n_avg = 50  # Taking last 50 to average
            train_loss = sum(train_losses[-n_avg:]) / len(train_losses[-n_avg:])
            n = int(dataset_loader.num_training_samples / batch_size)
            print(f"Steps: {steps}/{n} ({(steps / n) * 100:.2f}%)   |   "
                  f"TL: {train_loss:.4f}   |   "
                  f"VL: {val_loss:.4f}   |   "
                  f"Perplexity: {math.e ** train_loss:.2f}   |   "
                  f"Learning Rate: {lr}   |   "
                  f"Time: {int(time.time() - start)}s"
                  )

            # Update output file
            with open(output_path, "a") as f:
                f.write(f"{epoch+1},{steps/n},{train_loss},{val_loss},{int(time.time()-start)}\n")

            start = time.time()

            print("Logit stats:", y_pred.mean().item(), y_pred.std().item())

            # Generating prediction based on above sentence
            sentence = y_pred[0].squeeze(dim=0)  # Get the corresponding first batch
            sentence = torch.softmax(sentence, dim=-1)  # Softmax to convert to probabilities
            all_idx = []
            for j in range(sentence.shape[0]):
                current_idx = torch.multinomial(sentence[j], num_samples=1)
                if current_idx == str_to_int[EOS_TOKEN]:
                    break
                all_idx.append(current_idx.item())
            print(f"Src Sentence: {x[0]}")
            print(f"Prediction Sentence: {''.join([int_to_str[k] for k in all_idx])}")

            print("**********************************")

            # Testing out the current progress
            examples = ["What is your name?", "He went to school.", "This is my friend Chris.", "Where are you going?",
                        "How is your day today?", "Nice to meet you. My name is John.", "Are you sure?!?",
                        "She is going to the US"]
            print(f"Testing")
            for _ in range(3):
                txt = random.choice(examples)
                txt_tokens = dataset_loader.batch_tokenize([txt], True, False, True).to(device)
                prediction = model.translate(txt_tokens, sos_token=str_to_int[dataset_loader.sos_token], eos_token=str_to_int[dataset_loader.eos_token])
                translation = "".join([int_to_str[i] for i in prediction.tolist()[0]])
                print(f"Src: {txt}")
                print(f"Translation: {translation}\n")
            print("-------------------------------------")



        steps += 1


    if epoch % ckpt_epoch == 0:
        print(f"Now saving model checkpoint at epoch {epoch+1}")
        ckpt_dict = {"model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "config": config,
                     "str_to_int": str_to_int,
                     "train_epoch": dataset_loader.train_epoch + start_epoch,
                     }

        n_avg = 50  # Recalculating for explicitness
        train_loss = sum(train_losses[-n_avg:]) / len(train_losses[-n_avg:])
        torch.save(ckpt_dict, f"{ckpt_dir}/ep_{epoch+1}_{train_loss:.4f}_{val_losses[-1]:.4f}.pth")


print(f"End of training loop")
print(f"Saving final model")

ckpt_dict = {"model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "config": config,
             "str_to_int": str_to_int,
             "train_epoch": dataset_loader.train_epoch + start_epoch,
             }

n_avg = 50
train_loss = sum(train_losses[-n_avg:]) / len(train_losses[-n_avg:])
torch.save(ckpt_dict, f"{ckpt_dir}/final_model_{train_loss:.4f}_{val_losses[-1]:.4f}.pth")
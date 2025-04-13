import random
import torch


class DatasetLoader:
    def __init__(self,
                 src_ds_path: str,
                 tgt_ds_path: str,
                 sos_token: str,
                 eos_token: str,
                 pad_token: str,
                 batch_size: int,
                 seq_len: int,
                 train_split: float):

        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.batch_size = batch_size
        self.seq_len = seq_len

        with open(src_ds_path, "r", encoding="utf-8") as f:
            src_data_str = f.read()
            src_data = [i for i in src_data_str.split("\n") if len(i) > 0]

        with open(tgt_ds_path, "r", encoding="utf-8") as f:
            tgt_data_str = f.read()
            tgt_data = [i for i in tgt_data_str.split("\n") if len(i) > 0]


        unique_chars = set(src_data_str)
        unique_chars |= set(tgt_data_str)
        unique_chars = sorted(list(unique_chars)) + [sos_token, eos_token, pad_token]
        self.str_to_int = {c: i for i, c in enumerate(unique_chars)}  # Converting from character to integer
        self.int_to_str = {i: c for i, c in enumerate(unique_chars)}  # Converting from integer to character

        self.pad_idx = self.str_to_int[pad_token]  # Used in tokenization function below

        del src_data_str
        del tgt_data_str


        # Assert each text file have == number of examples
        assert len(src_data) == len(tgt_data), (f"Num sample mismatch:\nBoth files need to have same number of samples!"
                                                f"\nFile {src_ds_path} have {len(src_data)} samples"
                                                f"\nFile {tgt_ds_path} have {len(tgt_data)} samples")

        # Training pairs
        pairs = [(i, j) for i, j in zip(src_data, tgt_data)]

        n = int(len(pairs) * train_split)
        self.num_training_samples = n
        self.num_val_samples = len(pairs) - n

        self.train_ds = pairs[:n]
        self.val_ds = pairs[n:]

        # Randomly shuffle
        random.shuffle(self.train_ds)
        random.shuffle(self.val_ds)

        self.train_idx = 0
        self.val_idx = 0

        self.train_epoch = 0
        self.val_epoch = 0

        print("Dataset Loader Stats")
        print("===========================")
        print(f"Number of unique characters: {len(self.str_to_int)}")
        print(f"There are {self.num_training_samples} training samples in this dataset. ")
        print(f"Using {batch_size=}, there are {int(self.num_training_samples/batch_size)} steps per epoch for training")
        print(f"There are {self.num_val_samples} validation samples in this dataset. ")
        print(f"Using {batch_size=}, there are {int(self.num_val_samples/batch_size)} steps per epoch for validation")
        print("===========================")

    def get_batch(self, train: bool):
        if train:
            data = self.train_ds[self.train_idx: self.train_idx + self.batch_size]

            self.train_idx += self.batch_size
            if self.train_idx + self.batch_size > self.num_training_samples:
                random.shuffle(self.train_ds)
                self.train_idx = 0
                self.train_epoch += 1
        else:
            data = self.val_ds[self.val_idx: self.val_idx + self.batch_size]

            self.val_idx += self.batch_size
            if self.val_idx + self.batch_size > self.num_val_samples:
                random.shuffle(self.val_ds)
                self.val_idx = 0
                self.val_epoch += 1

        return [i[0] for i in data], [i[1] for i in data]

    def batch_tokenize(self, batch_sentences: list[str], add_sos: bool, add_eos: bool, check_unk_chars=False):
        if check_unk_chars:  # This should only be set to true when tokenizing inputs for model inference
            for char in set("".join(batch_sentences)):
                # Checking for existence of all characters in provided batch
                if char not in self.str_to_int:
                    return f"Invalid character found: '{char}'"

        tokens = []
        for sentence in batch_sentences:
            result = [self.str_to_int[c] for c in sentence]
            if add_sos:
                result.insert(0, self.str_to_int[self.sos_token])
            if add_eos:
                result.append(self.str_to_int[self.eos_token])
            if len(result) < self.seq_len:
                result.extend([self.pad_idx] * (self.seq_len - len(result)))

            result = result[:self.seq_len]
            assert len(result) == self.seq_len

            tokens.append(result)

        return torch.tensor(tokens, dtype=torch.long)

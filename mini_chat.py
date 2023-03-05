import gzip
import random
import tqdm
import numpy as np

import torch
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from palm_rlhf_pytorch import PaLM, RewardModel, RLHFTrainer
from accelerate import Accelerator


# ************** Language Model: PaLM **************
# constants
NUM_BATCHES = int(1e1)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 10 
PRIME_LENGTH = 128
GENERATE_EVERY = 10
GENERATE_LENGTH = 512
SEQ_LEN = 2000
NUM_TOKENS = 256

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# accelerator
accelerator = Accelerator()
device = accelerator.device

# instantiate palm
lang_model = PaLM(
    num_tokens=NUM_TOKENS,
    dim=512,
    depth=8
).to(device)

# prepare enwik8 data
with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer
optim = Lion(lang_model.palm_parameters(), lr = LEARNING_RATE)

lang_model, optim, train_loader, val_loader = accelerator.prepare(
    lang_model, optim, train_loader, val_loader
)

# training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    lang_model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = lang_model(next(train_loader), return_loss = True)
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    accelerator.print(f"training loss: {loss.item()}")
    accelerator.clip_grad_norm_(lang_model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if (i + 1) % VALIDATE_EVERY == 0:
        lang_model.eval()
        with torch.no_grad():
            loss = lang_model(next(val_loader), return_loss = True)
            accelerator.print(f"validation loss: {loss.item()}")

    if (i + 1) % GENERATE_EVERY == 0:
        lang_model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        accelerator.print(f"{'*' * 10}")
        accelerator.print(f"Input: {prime} \n")

        sample = lang_model.generate(GENERATE_LENGTH, inp[None, ...])
        output_str = decode_tokens(sample[0])
        accelerator.print(f"Output: {output_str} \n ")
        accelerator.print(f"{'*' * 10}")

# ************** Reward model **************
NUM_PROMPTS=5

reward_model = RewardModel(
    lang_model,
    num_binned_output = 5 # say rating from 1 to 5
).cuda()

# mock data
seq = torch.randint(0, NUM_TOKENS, (1, 1024)).cuda()
prompt_mask = torch.zeros(1, 1024).bool().cuda() # which part of the sequence is prompt, which part is response
labels = torch.randint(0, 5, (1,)).cuda()

# train
loss = reward_model(seq, prompt_mask = prompt_mask, labels = labels)
#loss.backward()

# after much training
reward = reward_model(seq, prompt_mask = prompt_mask)
print(f"Reward: {reward}")

# ************** RLHF **************
# ready your list of prompts for reinforcement learning
prompts = torch.randint(0, 256, (NUM_PROMPTS, 512)).cuda()
print(f"Prompts: {prompts}")

trainer = RLHFTrainer(
    palm = lang_model,
    reward_model = reward_model,
    prompt_token_ids = prompts
)

trainer.train(num_episodes = 1, max_timesteps= 1)

answer = trainer.generate(SEQ_LEN, prompt = prompts[0], num_samples = 10)
print(f"Answer: {answer}")

import functools
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append("picoGPT")

from gpt2 import gpt2, softmax
from utils import load_encoder_hparams_and_params

from datasets import load_dataset

import pandas as pd

logs = []
results = []

def max_fn(x):
    x_max = np.where(x > 0, x, 0)
    return x_max / np.sum(x_max)


def sample(p):
    return np.random.choice(np.arange(p.shape[-1]), p=p)


def autoregressive_sampling(x, model, N):
    n = len(x)
    T = len(x) + N
    with tqdm(total=N, desc="autoregressive sampling") as pbar:
        while n < T:
            x = np.append(x, sample(model(x)[-1]))
            n += 1
            pbar.update(1)

    return x


def speculative_sampling(x, draft_model, target_model, N, K):
    # NOTE: paper indexes arrays starting from 1, python indexes from 0, so
    # we have to add an extra -1 term when indexing using n, T, or t
    n = len(x)
    T = len(x) + N
    log = []

    with tqdm(total=N, desc="speculative sampling") as pbar:
        while n < T:
            prev_n = n

            # Step 1: auto-regressive decode K tokens from draft model and get final p
            x_draft = x
            for _ in range(K):
                p = draft_model(x_draft)
                x_draft = np.append(x_draft, sample(p[-1]))

            # Step 2: target model forward passes on x_draft
            q = target_model(x_draft)

            # Step 3: append draft tokens based on rejection criterion and resample
            # a token on rejection
            all_accepted = True
            for _ in range(K):
                i = n - 1
                j = x_draft[i + 1]
                if np.random.random() < min(1, q[i][j] / p[i][j]):  # accepted
                    x = np.append(x, j)
                    n += 1
                    log.append(0)
                else:  # rejected
                    x = np.append(x, sample(max_fn(q[i] - p[i])))  # resample
                    n += 1
                    all_accepted = False
                    log.append(1)
                    break

            # Step 4: if all draft tokens were accepted, sample a final token
            if all_accepted:
                x = np.append(x, sample(q[-1]))
                n += 1

            # just keeping my sanity
            pbar.update(n - prev_n)
            assert n == len(x), f"{n} {len(x)}"
    logs.append(log)
    return x


def create_model_fn(params, hparams, temperature, eps=1e-10):
    f = functools.partial(gpt2, **params, n_head=hparams["n_head"])

    def model_fn(inputs):
        logits = f(inputs)
        logits = logits / (temperature + eps)  # eps to avoid division by zero
        probs = softmax(logits)
        return probs

    return model_fn


def main(
    prompt: str = "Alan Turing theorized that computers would one day become",
    n_tokens_to_generate: int = 200,
    draft_model_size: str = "124M",
    target_model_size: str = "1558M",
    models_dir: str = "models",
    K: int = 10,
    temperature: float = 0.0,
    seed: int = 123,
):
    # seed numpy rng
    np.random.seed(seed)
    tot_auto = 0
    tot_sps = 0
    dataset = load_dataset("hellaswag")
    
    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, draft_hparams, draft_params = load_encoder_hparams_and_params(
        draft_model_size, models_dir
    )
    _, target_hparams, target_params = load_encoder_hparams_and_params(
        target_model_size, models_dir
    )
    draft_model = create_model_fn(draft_params, draft_hparams, temperature)
    target_model = create_model_fn(target_params, target_hparams, temperature)

    
    def run_sampling_fn(decode_fn, input_ids, **kwargs):
        start = time.perf_counter()
        output_ids = decode_fn(x=input_ids, **kwargs)
        text = encoder.decode(output_ids)
        elapsed_time = time.perf_counter() - start
        return text, elapsed_time
    
    for i, entry in enumerate(tqdm(dataset['train'])):
        if i != 0:
            max_length = max(map(len, logs))
            logs_tmp = [i + [-1] * (max_length - len(i)) for i in logs]
            df_logs = pd.DataFrame(logs_tmp)
            with open("throughput.txt", "w") as f:
                f.write("Autoregressive Decode\n")
                f.write("---------------------\n")
                f.write(f"Time = {tot_auto:.2f}s\n")
                f.write("\n")
                f.write("Speculative Decode\n")
                f.write("------------------\n")
                f.write(f"Time = {tot_sps:.2f}s\n")
            df_logs.to_csv("log.csv", index = False)
            df_results = pd.DataFrame(results)
            df_results.to_csv("results.csv", index = False)
            
        prompt = entry['ctx']

        input_ids = encoder.encode(prompt)
        
        autoregressive_text, autoregressive_time= run_sampling_fn(
            autoregressive_sampling,
            input_ids,
            model=target_model,
            N=n_tokens_to_generate,
        )
        
        speculative_text, speculative_time= run_sampling_fn(
            speculative_sampling,
            input_ids,
            target_model=target_model,
            draft_model=draft_model,
            N=n_tokens_to_generate,
            K=K,
        )
        tot_auto += autoregressive_time
        tot_sps += speculative_time
        results.append([autoregressive_text, speculative_text])
        # print results
        #print()
        #print("Autoregressive Decode")
        #print("---------------------")
        #print(f"Time = {autoregressive_time:.2f}s")
        #print(f"Text = {autoregressive_text}")
        #print()
        #print("Speculative Decode")
        #print("------------------")
        #print(f"Time = {speculative_time:.2f}s")
        #print(f"Text = {speculative_text}")
        

if __name__ == "__main__":
    import fire

    fire.Fire(main)

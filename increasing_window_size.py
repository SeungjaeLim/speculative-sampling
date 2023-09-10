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

def no_change(k):
    return k

def add_value(k, value=10):
    return k + value

def mult_value(k, value=2):
    return k * value

def speculative_sampling(x, draft_model, target_model, N, K, update_K_fn=no_change):
    n = len(x)
    T = len(x) + N
    log = []

    with tqdm(total=N, desc="speculative sampling") as pbar:
        while n < T:
            print(K)
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

            # Update K based on the function passed
            K = update_K_fn(K)
            
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
    n_tokens_to_generate: int = 20,
    draft_model_size: str = "124M",
    target_model_size: str = "1558M",
    models_dir: str = "models",
    K: int = 5,
    temperature: float = 0.0,
    seed: int = 123,
):
    # seed numpy rng
    np.random.seed(seed)
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

    results_data = []

    update_functions = [
        ('small k', no_change),
        ('large k', no_change),
        ('add k', functools.partial(add_value, value=10)),
        ('mult k', functools.partial(mult_value, value=2))
    ]
    
    def run_sampling_fn(decode_fn, input_ids, **kwargs):
        start = time.perf_counter()
        output_ids = decode_fn(x=input_ids, **kwargs)
        text = encoder.decode(output_ids)
        elapsed_time = time.perf_counter() - start
        return text, elapsed_time
    
    for i, entry in enumerate(tqdm(dataset['train'])):
        if i != 0:
            df_results = pd.DataFrame(results_data)
            df_results.to_csv("results_data.csv", index=False)
            
        prompt = entry['ctx']
        input_ids = encoder.encode(prompt)

        result_entry = {"prompt": prompt}
        for update_name, update_fn in update_functions:
            init_k = K
            if update_name == 'small k':
                init_k = 5
            elif update_name == 'large k':
                init_k = 50
            _, speculative_time = run_sampling_fn(
                speculative_sampling,
                input_ids,
                target_model=target_model,
                draft_model=draft_model,
                N=n_tokens_to_generate,
                K=init_k,
                update_K_fn=update_fn
            )
            result_entry[update_name] = speculative_time

        results_data.append(result_entry)
        

if __name__ == "__main__":
    import fire

    fire.Fire(main)

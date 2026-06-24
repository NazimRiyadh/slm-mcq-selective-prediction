"""Extract finite-choice features for AG News, TREC-6, and DBPedia-14.

This is the Kaggle-oriented extraction script used for the broad non-MCQ
extension. It writes one CSV with option-score features, one NPZ with final-token
hidden states, and one JSON metadata file per model--dataset condition.
"""
import os, json, time, gc, re
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = os.environ["MODEL_ID"]
MODEL_SHORT = os.environ["MODEL_SHORT"]
HF_TOKEN = os.environ.get("HF_TOKEN", None) or None
TASKS = [x.strip() for x in os.environ.get("TASKS", "agnews,trec6,dbpedia14").split(",") if x.strip()]
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working/features_nonmcq_finitechoice")
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "768"))
OPTION_BATCH_SIZE = int(os.environ.get("OPTION_BATCH_SIZE", "4"))
AGNEWS_PER_CLASS = int(os.environ.get("AGNEWS_PER_CLASS", "250"))
TREC_PER_CLASS = int(os.environ.get("TREC_PER_CLASS", "80"))
DBPEDIA_PER_CLASS = int(os.environ.get("DBPEDIA_PER_CLASS", "100"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

TASK_CONFIGS = {
    "agnews": {"display_name": "AG News", "labels": ["World", "Sports", "Business", "Sci/Tech"], "per_class": AGNEWS_PER_CLASS},
    "trec6": {"display_name": "TREC", "labels": ["Abbreviation", "Entity", "Description", "Human", "Location", "Numeric"], "per_class": TREC_PER_CLASS},
    "dbpedia14": {"display_name": "DBPedia", "labels": ["Company", "Educational institution", "Artist", "Athlete", "Office holder", "Mean of transportation", "Building", "Natural place", "Village", "Animal", "Plant", "Album", "Film", "Written work"], "per_class": DBPEDIA_PER_CLASS},
}

def safe_col_name(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:40]

def normalize_label_zero_based(df, n_classes):
    df = df.copy()
    mn, mx = int(df["label"].min()), int(df["label"].max())
    if mn == 1 and mx == n_classes:
        df["label"] = df["label"] - 1
    return df

def balanced_sample(df, n_classes, per_class, task_name):
    df = normalize_label_zero_based(df, n_classes)
    sampled = []
    for y in range(n_classes):
        sub = df[df["label"] == y].copy()
        if len(sub) == 0:
            raise ValueError(f"{task_name}: class {y} has 0 examples.")
        n = min(per_class, len(sub))
        if n < per_class:
            print(f"WARNING: {task_name} class {y} has only {len(sub)} examples; using {n}.")
        sampled.append(sub.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

def load_agnews():
    # Direct CSV avoids the bare-id Hugging Face URI issue encountered in some Kaggle images.
    url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    df = pd.read_csv(url, header=None, names=["label", "title", "description"])
    df["label"] = df["label"].astype(int) - 1
    df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    return df[["text", "label"]].dropna().reset_index(drop=True)

def load_trec6():
    direct_url = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
    coarse_map = {"ABBR": 0, "ENTY": 1, "DESC": 2, "HUM": 3, "LOC": 4, "NUM": 5}
    try:
        raw = pd.read_csv(direct_url, sep="\t", header=None, names=["line"], encoding="latin1")
        labels, texts = [], []
        for line in raw["line"].astype(str).tolist():
            if " " not in line or ":" not in line.split(" ", 1)[0]:
                continue
            tag, text = line.split(" ", 1)
            coarse = tag.split(":", 1)[0]
            if coarse in coarse_map:
                labels.append(coarse_map[coarse])
                texts.append(text.strip())
        df = pd.DataFrame({"text": texts, "label": labels})
        if len(df) >= 100:
            return df.dropna().reset_index(drop=True)
    except Exception as e:
        print("Direct TREC load failed, falling back to HF CogComp/trec:", repr(e))
    ds = load_dataset("CogComp/trec", split="train", trust_remote_code=True)
    return pd.DataFrame([{"text": str(ex.get("text", "")), "label": int(ex.get("coarse_label"))} for ex in ds]).dropna().reset_index(drop=True)

def load_dbpedia14():
    tried = []
    for repo in ["fancyzhx/dbpedia_14", "dbpedia_14"]:
        try:
            ds = load_dataset(repo, split="test")
            rows = []
            for ex in ds:
                label = int(ex["label"])
                if "title" in ex and "content" in ex:
                    text = str(ex["title"]) + ". " + str(ex["content"])
                elif "title" in ex and "text" in ex:
                    text = str(ex["title"]) + ". " + str(ex["text"])
                elif "content" in ex:
                    text = str(ex["content"])
                elif "text" in ex:
                    text = str(ex["text"])
                else:
                    text = ". ".join(str(v) for k, v in ex.items() if k != "label" and isinstance(v, str))
                rows.append({"text": text, "label": label})
            df = pd.DataFrame(rows).dropna().reset_index(drop=True)
            if len(df) > 100:
                return df
        except Exception as e:
            tried.append((repo, repr(e)))
    raise RuntimeError(f"DBPedia load failed for all repos: {tried}")

def load_balanced_task(task_name):
    cfg = TASK_CONFIGS[task_name]
    if task_name == "agnews":
        df = load_agnews()
    elif task_name == "trec6":
        df = load_trec6()
    elif task_name == "dbpedia14":
        df = load_dbpedia14()
    else:
        raise ValueError(f"Unknown task: {task_name}")
    out = balanced_sample(df, len(cfg["labels"]), int(cfg["per_class"]), task_name)
    print(f"\nLoaded {task_name}: {out.shape}")
    print(out["label"].value_counts().sort_index())
    return out, cfg

def make_prompt(tokenizer, text, labels):
    choice_lines = [f"{LETTERS[i]}. {lab}" for i, lab in enumerate(labels)]
    user_content = (
        "Classify the following text into exactly one category.\n\n"
        f"Text:\n{text}\n\n"
        "Choices:\n" + "\n".join(choice_lines) + "\n\n"
        f"Return only the letter {LETTERS[0]} to {LETTERS[len(labels)-1]}."
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_content + "\nAnswer:"

@torch.no_grad()
def score_options(model, tokenizer, prompt, letters):
    scores = []
    for start in range(0, len(letters), OPTION_BATCH_SIZE):
        batch_letters = letters[start:start + OPTION_BATCH_SIZE]
        texts = [prompt + " " + x for x in batch_letters]
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        prompt_len = len(prompt_ids)
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH, add_special_tokens=True).to(model.device)
        input_ids, attn = enc["input_ids"], enc["attention_mask"]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        labels[attn == 0] = -100
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        safe_labels = shift_labels.clone()
        safe_labels[~mask] = 0
        tok_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1) * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_logp = tok_logp.sum(dim=1) / lengths
        scores.extend(mean_logp.detach().float().cpu().numpy().tolist())
        del enc, input_ids, attn, labels, out, logits
        torch.cuda.empty_cache()
    return np.asarray(scores, dtype=np.float32)

@torch.no_grad()
def get_hidden(model, tokenizer, prompt):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, add_special_tokens=True).to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1]
    last_idx = enc["attention_mask"].sum(dim=1).item() - 1
    vec = h[0, last_idx, :].detach().float().cpu().numpy()
    del enc, out, h
    torch.cuda.empty_cache()
    return vec

def softmax_np(scores):
    scores = np.asarray(scores, dtype=np.float64)
    z = scores - np.max(scores)
    p = np.exp(z)
    return p / p.sum()

def entropy_np(p):
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(p * np.log(p + 1e-12)))

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"device_map": "auto", "trust_remote_code": True, "token": HF_TOKEN}
    if USE_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    return tokenizer, model

def run_task(task_name, tokenizer, model):
    cfg = TASK_CONFIGS[task_name]
    labels = cfg["labels"]
    letters = LETTERS[:len(labels)]
    feat_path = os.path.join(OUTPUT_DIR, f"{MODEL_SHORT}__{task_name}__features.csv")
    hid_path = os.path.join(OUTPUT_DIR, f"{MODEL_SHORT}__{task_name}__hidden.npz")
    meta_path = os.path.join(OUTPUT_DIR, f"{MODEL_SHORT}__{task_name}__meta.json")
    if os.path.exists(feat_path) and os.path.exists(hid_path) and os.path.exists(meta_path):
        print(f"\nSKIP existing output: {MODEL_SHORT} {task_name}")
        return
    df, cfg = load_balanced_task(task_name)
    rows, hidden_rows = [], []
    start_time = time.time()
    for i, r in tqdm(df.iterrows(), total=len(df), desc=f"{MODEL_SHORT} {task_name}"):
        text = str(r["text"])
        gold = int(r["label"])
        prompt = make_prompt(tokenizer, text, labels)
        scores = score_options(model, tokenizer, prompt, letters)
        probs = softmax_np(scores)
        pred = int(np.argmax(scores))
        sorted_scores = np.sort(scores)[::-1]
        rec = {
            "example_id": i,
            "model": MODEL_SHORT,
            "dataset": task_name,
            "gold_label": gold,
            "pred_label": pred,
            "correct": int(pred == gold),
            "max_option_score": float(np.max(scores)),
            "mean_option_score": float(np.mean(scores)),
            "std_option_score": float(np.std(scores)),
            "min_option_score": float(np.min(scores)),
            "option_score_margin": float(sorted_scores[0] - sorted_scores[1]),
            "option_score_entropy": entropy_np(probs),
            "top_prob_softmax": float(np.max(probs)),
            "n_options": int(len(labels)),
        }
        for j, lab in enumerate(labels):
            suffix = safe_col_name(f"{letters[j]}_{lab}")
            rec[f"score_{suffix}"] = float(scores[j])
            rec[f"prob_{suffix}"] = float(probs[j])
        rows.append(rec)
        hidden_rows.append(get_hidden(model, tokenizer, prompt))
    feat = pd.DataFrame(rows)
    H = np.vstack(hidden_rows).astype("float32")
    feat.to_csv(feat_path, index=False)
    np.savez_compressed(hid_path, hidden=H)
    meta = {"model_id": MODEL_ID, "model_short": MODEL_SHORT, "dataset": task_name, "display_name": cfg["display_name"], "n": int(len(feat)), "accuracy": float(feat["correct"].mean()), "hidden_shape": list(H.shape), "labels": labels, "choice_letters": letters, "balanced_sampling": True, "per_class": int(cfg["per_class"]), "runtime_seconds": float(time.time() - start_time)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("\nDONE:", MODEL_SHORT, task_name, "accuracy", meta["accuracy"], "hidden", H.shape)

def main():
    print("=" * 80)
    print("MODEL_ID:", MODEL_ID)
    print("MODEL_SHORT:", MODEL_SHORT)
    print("TASKS:", TASKS)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("MAX_LENGTH:", MAX_LENGTH)
    print("OPTION_BATCH_SIZE:", OPTION_BATCH_SIZE)
    print("USE_4BIT:", USE_4BIT)
    print("=" * 80)
    tokenizer, model = load_model_and_tokenizer()
    for task in TASKS:
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task}")
        run_task(task, tokenizer, model)
        gc.collect(); torch.cuda.empty_cache()
    del model
    gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

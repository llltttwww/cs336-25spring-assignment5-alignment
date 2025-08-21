import json, re
from pathlib import Path

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")

def last_boxed(sol: str):
    m = None
    for m in BOXED_RE.finditer(sol):
        pass
    return m.group(1).strip() if m else None

R1_ZERO_PREFIX = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)

def gather_split(root: Path, split: str):
    for f in (root / split).rglob("*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            ex = json.load(fh)
        yield {"problem": ex["problem"].strip(), "solution": ex["solution"].strip()}

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def make_sft_rows(rows):
    out = []
    for ex in rows:
        prompt = R1_ZERO_PREFIX.format(question=ex["problem"])
        ans = last_boxed(ex["solution"]) or ""
        response = ex["solution"] + " </think> <answer>" + ans + "</answer>"
        out.append({"prompt": prompt, "response": response})
    return out

if __name__ == "__main__":
    root = Path("math-dataset/MATH")
    outdir = Path("data/MATH")

    train_rows = list(gather_split(root, "train"))
    val_rows = list(gather_split(root, "test"))

    # write_jsonl(outdir/"train.jsonl", train_rows)
    # write_jsonl(outdir/"validation.jsonl", val_rows)
    # write_jsonl(outdir/"sft.jsonl", make_sft_rows(train_rows))
    write_jsonl(outdir/"sft_validation.jsonl", make_sft_rows(val_rows))


    print("✅ Done:",
          len(train_rows), "train,",
          len(val_rows), "validation")
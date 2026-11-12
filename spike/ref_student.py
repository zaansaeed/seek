"""M0 spike: PyTorch reference scores for the off-the-shelf student (ms-marco-MiniLM-L6-v2).

For each test pair it computes two scores in full fp32:
  A = the model as trained: BERT -> pooler (dense + tanh) -> classifier, real token types (0 = query, 1 = passage).
      The Berlin pairs should reproduce the model card: +8.607 and -4.320.
  B = what llama.cpp computes: no pooler (the converter drops it) and every token type = 0.
      This is the target the C++ spike must match.

Writes build/student_ref.txt, one line per pair:  <score_a> <score_b> <n_tokens> <id_1> ... <id_n>
spike/student_spike.cpp reads it and compares. The pair texts below must match the C++ file exactly;
the token-ID comparison will catch any difference.

Run from the repo root, in the `seek` conda env:
    python spike/ref_student.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/hf/ms-marco-MiniLM-L6-v2"
OUT_PATH = Path("build/student_ref.txt")

SNIPPET_A = """def upload_with_retry(client, path, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return client.put(path)
        except ConnectionError:
            time.sleep(2 ** attempt)
    raise UploadFailed(path)"""

SNIPPET_B = """def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("timeout", 30)
    return cfg"""

GRAVITY_DOC = (
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
    "objects and is responsible for the movement of planets around the sun."
)

# (label, query, document)
PAIRS = [
    ("Berlin / population", "How many people live in Berlin?",
     "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
    ("Berlin / museums", "How many people live in Berlin?", "Berlin is well known for its museums."),
    ("China / Beijing", "What is the capital of China?", "The capital of China is Beijing."),
    ("China / gravity", "What is the capital of China?", GRAVITY_DOC),
    ("retry / snippet A", "where do we retry failed uploads?", SNIPPET_A),
    ("retry / snippet B", "where do we retry failed uploads?", SNIPPET_B),
]
MODEL_CARD = {0: 8.607138, 1: -4.320078}  # expected score A for the Berlin pairs


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.eval()  # turns off dropout

    lines = []
    print(f"{'pair':<20} {'A (as trained)':>15} {'B (llama.cpp)':>15} {'B - A':>8}  tokens")
    with torch.no_grad():
        for i, (label, query, doc) in enumerate(PAIRS):
            # Pair encoding: [CLS] query [SEP] doc [SEP], token_type_ids 0 for the query part, 1 for the doc.
            enc = tokenizer(query, doc, return_tensors="pt")

            # A: the normal forward pass.
            score_a = model(**enc).logits[0, 0].item()

            # B: same encoder, but every token type = 0, and the classifier applied directly to the
            # [CLS] hidden state, skipping the pooler (model.bert.pooler is never called).
            hidden = model.bert(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=torch.zeros_like(enc["token_type_ids"]),
            ).last_hidden_state
            score_b = model.classifier(hidden[:, 0]).squeeze().item()

            ids = enc["input_ids"][0].tolist()
            lines.append(f"{score_a:.6f} {score_b:.6f} {len(ids)} " + " ".join(map(str, ids)))
            print(f"{label:<20} {score_a:>+15.4f} {score_b:>+15.4f} {score_b - score_a:>+8.3f}  {len(ids)}")

    print("\nModel card check (score A):")
    for i, expected in MODEL_CARD.items():
        got = float(lines[i].split()[0])
        print(f"  {PAIRS[i][0]:<20} {got:+.4f}  (card {expected:+.4f})  diff {abs(got - expected):.4f}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()

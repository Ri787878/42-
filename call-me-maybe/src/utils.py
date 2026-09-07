from llm_sdk import Small_LLM_Model
import numpy as np
import json
import sys


def print_to_stderr(*a):
    """Print a message to stderr"""
    print(*a, file=sys.stderr)


def softmax(x, temperature=1.0):
    values = np.asarray(x, dtype=float)

    if not np.isfinite(values).any():
        raise ValueError(
            "Cannot calculate softmax: no finite logits were provided."
        )

    if temperature <= 1e-5:
        probabilities = np.zeros_like(values)
        probabilities[np.nanargmax(values)] = 1.0
        return probabilities

    finite_values = np.where(
        np.isfinite(values),
        values,
        -np.inf,
    )

    maximum = np.max(finite_values)
    shifted = finite_values - maximum
    exponentials = np.exp(shifted / temperature)
    total = exponentials.sum()

    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "Cannot calculate softmax: invalid probability total."
        )

    return exponentials / total


def decoding_strategy(probs: np.ndarray | None) -> int:
    return int(np.argmax(probs))


def get_token_id(llm_model: Small_LLM_Model) -> dict[str, int]:
    # --- Build token_to_id from vocab file ---
    vocab_path = llm_model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as vf:
        vocab_obj = json.load(vf)

    # Support common vocab layouts:
    # 1) {"token": id, ...}
    # 2) [{"token": "...", "id": 123}, ...]
    # 3) ["token0", "token1", ...]  -> index is id
    if isinstance(vocab_obj, dict):
        token_to_id = {str(tok): int(tid) for tok, tid in vocab_obj.items()}
    elif (
         isinstance(vocab_obj, list)
         and len(vocab_obj) > 0
         and isinstance(vocab_obj[0], dict)
         ):
        token_to_id = {str(x["token"]): int(x["id"]) for x in vocab_obj}
    elif isinstance(vocab_obj, list):
        token_to_id = {str(tok): i for i, tok in enumerate(vocab_obj)}
    else:
        raise ValueError(f"Unsupported vocab format in {vocab_path}")
    return token_to_id


def extract_last_position_if_needed(logits) -> np.ndarray:
    """
    Normalize model logits to a 1D next-token logits vector [vocab_size].

    Accepts:
      - [vocab]
      - [seq_len, vocab]
      - [batch, seq_len, vocab]  (uses batch index 0)
    """
    logits_arr = np.asarray(logits)

    if logits_arr.ndim == 1:
        # already [vocab]
        return logits_arr.astype(float, copy=False)

    if logits_arr.ndim == 2:
        # [seq_len, vocab] -> last position
        return logits_arr[-1].astype(float, copy=False)

    if logits_arr.ndim == 3:
        # [batch, seq_len, vocab] -> first batch, last position
        return logits_arr[0, -1].astype(float, copy=False)

    raise ValueError(f"Unsupported logits shape: {logits_arr.shape}")

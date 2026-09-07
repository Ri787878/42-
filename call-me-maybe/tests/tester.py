import json
import sys
from pathlib import Path


def compare_json_test_results(expected_path: str, generated_path: str):
    """Compares expected ground truth JSON against model-generated JSON."""
    p_exp, p_gen = Path(expected_path), Path(generated_path)

    if not p_exp.exists() or not p_gen.exists():
        print(f"Error: One or both files do not exist: {p_exp}, {p_gen}")
        sys.exit(1)

    with open(p_exp, "r", encoding="utf-8") as f:
        expected_data = json.load(f)

    with open(p_gen, "r", encoding="utf-8") as f:
        generated_data = json.load(f)

    # Index items by prompt for fast lookup
    exp_map = {item.get("prompt"): item for item in expected_data if "prompt" in item}
    gen_map = {item.get("prompt"): item for item in generated_data if "prompt" in item}

    all_prompts = list(dict.fromkeys(list(exp_map.keys()) + list(gen_map.keys())))

    passed = 0
    failed = 0
    missing = 0

    print("=" * 70)
    print(" JSON DECODING TEST VERIFICATION REPORT")
    print("=" * 70)

    for idx, prompt in enumerate(all_prompts, start=1):
        print(f"\n[Test {idx}] Prompt: {prompt!r}")

        exp_item = exp_map.get(prompt)
        gen_item = gen_map.get(prompt)

        if not exp_item:
            print("  ❌ FAILURE: Found in generated output but missing from expected file.")
            failed += 1
            continue

        if not gen_item:
            print("  ❌ FAILURE: Prompt missing from generated output.")
            missing += 1
            continue

        # Check for matching function name and parameters
        exp_name = exp_item.get("name")
        gen_name = gen_item.get("name")

        exp_params = exp_item.get("parameters", {})
        gen_params = gen_item.get("parameters", {})

        diffs = []
        if exp_name != gen_name:
            diffs.append(f"Function Name mismatch -> Expected: {exp_name!r}, Got: {gen_name!r}")

        if exp_params != gen_params:
            diffs.append(f"Parameters mismatch -> Expected: {exp_params}, Got: {gen_params}")

        if not diffs:
            print("  ✅ PASS: Output matches expected schema.")
            passed += 1
        else:
            print("  ❌ MISMATCH DETECTED:")
            for d in diffs:
                print(f"     - {d}")
            failed += 1

    print("\n" + "=" * 70)
    print(
        f" SUMMARY: Total: {len(all_prompts)} | Passed: {passed} "
        f"| Mismatched: {failed} | Missing: {missing}")
    print("=" * 70)


if __name__ == "__main__":
    # Adjust paths as needed for your test environment
    EXPECTED_FILE = "tests/test_2/solution_test_2.json"
    GENERATED_FILE = "tests/test_2/output_test_2.json"

    compare_json_test_results(EXPECTED_FILE, GENERATED_FILE)



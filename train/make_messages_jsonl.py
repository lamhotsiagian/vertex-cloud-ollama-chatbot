import argparse
import json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_jsonl", required=True, help="Each line: {"prompt":..., "response":...}")
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--system", default="You are concise.")
    args = p.parse_args()

    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        with open(args.pairs_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj["prompt"]
                resp = obj["response"]
                rec = {
                    "messages": [
                        {"role": "system", "content": args.system},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                    ],
                    "metadata": obj.get("metadata", {}),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Wrote:", args.out_jsonl)

if __name__ == "__main__":
    main()

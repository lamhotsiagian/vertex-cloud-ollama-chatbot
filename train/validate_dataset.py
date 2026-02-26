import json
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    args = p.parse_args()

    n = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "messages" in obj, "Missing 'messages'"
            assert isinstance(obj["messages"], list), "'messages' must be a list"
            for m in obj["messages"]:
                assert isinstance(m, dict), "Each message must be an object"
                assert m.get("role") in ("system", "user", "assistant"), f"Invalid role: {m.get('role')}"
                assert isinstance(m.get("content"), str), "content must be a string"
            n += 1
    print(f"OK: validated {n} records from {args.jsonl}")

if __name__ == "__main__":
    main()

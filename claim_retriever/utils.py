import json


def load_queries(filename="queries.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            queries = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        return queries
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please make sure it exists.")
        return []


def save_claims_to_file(claims, output_filename, encoding="utf-8"):
    try:
        with open(output_filename, "w", encoding=encoding) as fout:
            json.dump(claims, fout, ensure_ascii=False, indent=4)
        print(f"✅ Successfully saved {len(claims)} claims to '{output_filename}'")
    except Exception as e:
        print(f"Error saving claims to file: {e}")

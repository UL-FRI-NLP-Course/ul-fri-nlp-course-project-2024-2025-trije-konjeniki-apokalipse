import json

def convert_to_jsonl(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry in data:
            text_entry = {"text": entry.get("text", "")}
            json_line = json.dumps(text_entry, ensure_ascii=False)
            outfile.write(json_line + "\n")

if __name__ == "__main__":
    convert_to_jsonl("rag_road_chunks.json", "docs.jsonl")
    print("Done: docs.jsonl created.")
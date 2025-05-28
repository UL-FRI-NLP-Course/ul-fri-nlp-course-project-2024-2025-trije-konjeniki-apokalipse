import os, glob, re, json
import pandas as pd
import streamlit as st

# ─── CONFIG ────────────────────────────────────────────────────────────────────
N = 30  # first N examples for testing

BASE_DIR      = os.path.dirname(__file__)
RAW_DIR       = os.path.normpath(os.path.join(BASE_DIR, "../results"))
PROGRESS_DIR  = os.path.normpath(os.path.join(BASE_DIR, "../progress"))
FINAL_CSV_DIR = RAW_DIR

EVALUATORS = ["Janez", "Zan", "Matic"]
os.makedirs(PROGRESS_DIR, exist_ok=True)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def find_scenarios():
    return sorted(glob.glob(os.path.join(RAW_DIR, "*_eval.txt")))

def clean_block(text: str) -> str:
    """Remove any ‘BASIC:’, separator lines, and Assistant markers from a block."""
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        # skip blank
        if not stripped:
            out.append("")
            continue
        # skip "BASIC:" line
        if stripped.upper() == "BASIC:":
            continue
        # skip pure-==== lines
        if re.fullmatch(r"=+", stripped):
            continue
        # skip any "### Assistant:" or similar headings
        if re.match(r"^#+\s*Assistant\s*:", stripped, re.I):
            continue
        out.append(line)
    # collapse multiple blank lines
    cleaned = []
    prev_blank = False
    for l in out:
        is_blank = (l.strip() == "")
        if is_blank and prev_blank:
            continue
        cleaned.append(l)
        prev_blank = is_blank
    return "\n".join(cleaned).strip()

def parse_file(path, limit=N):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"===\s*Example\s*(\d+)\s*/\s*\d+\s*===", text, flags=re.I)
    entries = []
    for i in range(1, len(parts), 2):
        exnum = int(parts[i])
        body  = parts[i+1]

        inp_m = re.search(r"INPUT:\s*(.*?)\n\s*GROUND-TRUTH:", body, re.S)
        gt_m  = re.search(r"GROUND-TRUTH:\s*(.*?)\n\s*MODEL-OUTPUT:", body, re.S)
        mo_m  = re.search(r"MODEL-OUTPUT:\s*(.*)",               body, re.S)

        inp = clean_block(inp_m.group(1)) if inp_m else ""
        gt  = clean_block(gt_m.group(1))  if gt_m  else ""
        mo  = clean_block(mo_m.group(1))  if mo_m  else ""

        entries.append({
            "example":      exnum,
            "input":        inp,
            "ground_truth": gt,
            "model_output": mo,
        })
        if len(entries) >= limit:
            break

    return entries

def load_all_data():
    data = {}
    for path in find_scenarios():
        key = os.path.splitext(os.path.basename(path))[0]
        parsed = parse_file(path, N)
        if len(parsed) >= N:
            data[key] = parsed
    return data

def prog_file(ev): return os.path.join(PROGRESS_DIR, f"{ev}_progress.json")

def load_progress(ev):
    p = prog_file(ev)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return {"scenario_idx": {k:0 for k in all_data}, "entries": []}

def save_progress(ev, prog):
    json.dump(prog, open(prog_file(ev), "w"), indent=2, ensure_ascii=False)

def save_final_csv(ev, prog):
    df = pd.DataFrame(prog["entries"])
    df.to_csv(os.path.join(FINAL_CSV_DIR, f"{ev}.csv"), index=False)

def load_all_entries():
    rows = []
    for f in glob.glob(os.path.join(FINAL_CSV_DIR, "*.csv")):
        rows += pd.read_csv(f).to_dict("records")
    for ev in EVALUATORS:
        p = prog_file(ev)
        if os.path.exists(p):
            rows += json.load(open(p, encoding="utf-8"))["entries"]
    return pd.DataFrame(rows)

# ─── STREAMLIT APP ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🚦 Traffic Evaluation")

# load data & progress
all_data = load_all_data()
if not all_data:
    st.error(f"No scenario files with ≥{N} examples in `{RAW_DIR}`.")
    st.stop()

evaluator = st.sidebar.selectbox("Evaluator", EVALUATORS)
prog      = load_progress(evaluator)
scenarios = list(all_data.keys())
scenario  = st.sidebar.selectbox("Scenario", scenarios)

idx     = prog["scenario_idx"].get(scenario, 0)
records = all_data[scenario]
max_idx = len(records)

# sidebar: global & personal averages
st.sidebar.markdown("### 📊 Averages")
df_all = load_all_entries()
if not df_all.empty:
    ga = df_all.groupby("scenario").agg(avg_rating=("rating","mean"), pct_better=("better","mean")).round(3)
    st.sidebar.dataframe(ga, height=200)
df_me = pd.DataFrame(prog["entries"])
if {"evaluator","scenario"}.issubset(df_me.columns):
    me = df_me[(df_me["evaluator"]==evaluator)&(df_me["scenario"]==scenario)]
    if not me.empty:
        st.sidebar.write(f"**Your avg ({scenario}):**")
        st.sidebar.write(f"- Rating: {me.rating.mean():.2f}")
        st.sidebar.write(f"- Better%: {me.better.mean()*100:.0f}%")

# main
st.write(f"### Scenario **{scenario}** — Example {idx+1} of {max_idx}")
if idx >= max_idx:
    st.success("✅ Finished this scenario!")
    if st.button("Save final CSV"):
        save_final_csv(evaluator, prog)
        st.info(f"Saved `results/{evaluator}.csv`")
    st.stop()

rec = records[idx]

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("INPUT")
    st.write(rec["input"])
with col2:
    st.subheader("GROUND-TRUTH")
    st.write(rec["ground_truth"])
with col3:
    st.subheader("MODEL-OUTPUT")
    st.write(rec["model_output"])

st.markdown("---")
rating = st.slider("Your rating (1–5)", 1, 5, 3, key="rating")
better = st.checkbox("Better than ground-truth?", key="better")

def submit_cb():
    prog["entries"].append({
        "evaluator": evaluator,
        "scenario":  scenario,
        "example":   rec["example"],
        "rating":    st.session_state["rating"],
        "better":    int(st.session_state["better"])
    })
    prog["scenario_idx"][scenario] = idx + 1
    save_progress(evaluator, prog)

st.button("Submit & Next", on_click=submit_cb)
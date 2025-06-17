#!/usr/bin/env python3
import logging
import argparse
from src.loader    import load_all_texts
from src.filterer  import filter_by_prompt
from src.embeddings  import compute_embeddings
from src.indexer   import build_faiss_index, save_index
from src.matcher   import match_resumes_to_jobs

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def read_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

def main():
    parser = argparse.ArgumentParser(description="CV→Job matcher with prompt filter from file")
    parser.add_argument("--prompt-file", "-p",
                        help="Path to prompt.txt file",
                        default="prompt.txt")
    args = parser.parse_args()

    print("\n🚀 Starting CV-to-Job Matching Pipeline\n")

    # 1. Load filter-prompt from file
    prompt = read_prompt(args.prompt_file)
    print(f"🔍 Loaded filter prompt from {args.prompt_file}:")
    print(f"   “{prompt}”\n")

    # 2. Load resumes & jobs
    resumes = load_all_texts("resumes/")
    jobs    = load_all_texts("jobs/")
    print(f"1️⃣ Loaded {len(resumes)} resumes and {len(jobs)} jobs\n")

    # 3. Filter resumes via prompt
    print("2️⃣ Filtering resumes by prompt (this will invoke the Chat API)...")
    resumes = filter_by_prompt(resumes, prompt)
    print(f"   ✔️ {len(resumes)} resumes remain after filtering\n")
    if not resumes:
        print("❗️ No resumes passed the filter. Exiting.")
        return

    # 4. Compute embeddings, build index, match & export
    print("3️⃣ Computing embeddings for resumes...")
    resume_vecs = compute_embeddings(resumes)
    print("4️⃣ Computing embeddings for jobs...")
    job_vecs    = compute_embeddings(jobs)

    print("5️⃣ Building FAISS index...")
    index = build_faiss_index(resume_vecs)
    save_index(index, "faiss.index")
    print("   ✔️ Index built\n")

    print("6️⃣ Matching resumes to jobs...")
    matches_df = match_resumes_to_jobs(index, job_vecs, top_k=5)
    print(f"   ✔️ Found {len(matches_df)} matches\n")

    print("7️⃣ Exporting to matches.csv...")
    matches_df.to_csv("matches.csv", index=False, encoding="utf-8")
    print("🎉 Done! Review matches.csv\n")

if __name__ == "__main__":
    main()

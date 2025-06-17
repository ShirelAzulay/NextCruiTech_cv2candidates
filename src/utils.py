from src.loader import load_all_texts
from src.embeddings import compute_embeddings
from src.indexer import build_faiss_index, save_index
from src.matcher import match_resumes_to_jobs

def main():
    resumes = load_all_texts("resumes/")
    jobs    = load_all_texts("jobs/")

    resume_vecs = compute_embeddings(resumes)
    job_vecs    = compute_embeddings(jobs)

    index = build_faiss_index(resume_vecs)
    save_index(index, "faiss.index")

    matches_df = match_resumes_to_jobs(index, job_vecs, top_k=5)
    matches_df.to_csv("matches.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    main()

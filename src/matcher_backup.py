import numpy as np


def get_topk_matches(index, embeddings, ids, query_emb, top_k=5):
    vec = np.array(query_emb, dtype="float32").reshape(1, -1)
    D, I = index.search(vec, top_k)
    matches = []
    for dist, idx in zip(D[0], I[0]):
        # Convert L2 distance to similarity score
        score = 1 / (1 + dist)  # Better normalization for similarity
        resume_id = ids[idx]
        matches.append({
            "resume_id": resume_id,
            "score": score,
            "resume_text": embeddings[resume_id].get("text", "")
        })
    return matches


def gpt_filter_topk(client, candidates, filter_prompt):
    """
    Filter candidates based on GPT evaluation.
    Returns only candidates that match the filter criteria.
    """
    results = []
    for item in candidates:
        resume_text = item["resume_text"][:3000]  # Use more text for better evaluation

        # Create a more detailed prompt
        full_prompt = f"""
        Evaluate if this resume matches the following criteria: {filter_prompt}

        Resume text:
        {resume_text}

        Answer with ONLY 'yes' or 'no'. Answer 'yes' if the candidate meets the criteria, 'no' otherwise.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip().lower()

            # Only include candidates that pass the filter
            if answer.startswith("yes"):
                results.append(item)
        except Exception as e:
            print(f"Error in GPT filter: {e}")
            # On error, include the candidate to avoid losing potential matches
            results.append(item)

    return results
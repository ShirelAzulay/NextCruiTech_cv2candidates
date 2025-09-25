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
    UPGRADED: Filter candidates using GPT-4 for better accuracy
    Returns only candidates that match the filter criteria.
    """
    results = []

    for item in candidates:
        resume_text = item["resume_text"][:3000]  # More text for GPT-4
        resume_name = item["resume_id"].split('|')[1] if '|' in item["resume_id"] else item["resume_id"]

        # UPGRADED: Use GPT-4 with more sophisticated prompt
        full_prompt = f"""
אתה מומחה גיוס עם ניסיון של 20 שנה. נתח את קורות החיים הבאים והחלט אם המועמד מתאים לקריטריון.

קריטריון: {filter_prompt}

קורות חיים:
{resume_text}

בדוק בקפדנות ותשובה:
- "כן" אם המועמד מתאים בבירור לקריטריון
- "לא" אם המועמד לא מתאים או אם יש ספק

תשובה:
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4",  # UPGRADED to GPT-4
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip()

            # Debug output
            print(f"🔍 GPT-4 DEBUG: {resume_name} - answered: '{answer}' for filter: '{filter_prompt[:30]}...'")

            # Check for positive responses
            if answer == "כן" or answer.lower() == "yes":
                results.append(item)
                print(f"✅ PASSED: {resume_name}")
            else:
                print(f"❌ FILTERED OUT: {resume_name}")

        except Exception as e:
            print(f"❌ ERROR filtering {resume_name}: {e}")
            # On error, exclude candidate to be safe

    print(f"🎯 GPT-4 FILTER SUMMARY: {len(results)}/{len(candidates)} candidates passed filter: '{filter_prompt}'")
    return results
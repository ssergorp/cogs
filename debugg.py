import numpy as np

# pick first chunk
vec = np.array(all_chunks[0]['embedding'], dtype=float)
q_vec = np.array(q_embed, dtype=float)

print("vec[:10]=", vec[:10])
print("q_vec[:10]=", q_vec[:10])
print("cosine=", np.dot(vec, q_vec)/(np.linalg.norm(vec)*np.linalg.norm(q_vec)))


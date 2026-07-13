import torch

data = torch.load("data/embeddings/clip.pt", weights_only=False)

print("ref_path - type:", type(data["ref_path"]), "content: ", data["ref_path"])
print("dist_path - type:", type(data["dist_path"]), "content: ", data["dist_path"])
print("ref_path[0] - type:", type(data["ref_path"][0]), "content: ", data["ref_path"][0])
print("dist_path[0] - type:", type(data["dist_path"][0]), "content: ", data["dist_path"][0])
import torch

data = torch.load("data/embeddings/clip.pt", weights_only=False)

print("ref_path - type:", type(data["ref_path"]), "len: ", len(data["ref_path"]))
print("dist_path - type:", type(data["dist_path"]), "len: ", len(data["dist_path"]))
print("ref_path[0] - type:", type(data["ref_path"][0]), "len: ", len(data["ref_path"][0]))
print("dist_path[0] - type:", type(data["dist_path"][0]), "len: ", len(data["dist_path"][0]))

print
import h5py

path = "data/raw/biological_data_pfp/train/train_embeddings.h5"
with h5py.File(path, "r") as h5:
    print("Top-level keys:", list(h5.keys())[:50])

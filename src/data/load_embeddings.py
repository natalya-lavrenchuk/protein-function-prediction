import h5py
import numpy as np

def load_embeddings_h5(h5_path: str, protein_ids: list[str], key: str | None = None):
    with h5py.File(h5_path, "r") as h5:
        sample = protein_ids[: min(20, len(protein_ids))]
        if sample and all(pid in h5 for pid in sample):
            X = [np.array(h5[pid]) for pid in protein_ids]
            return np.vstack([x if x.ndim > 1 else x.reshape(1, -1) for x in X])

        if key and key in h5:
            data = np.array(h5[key])
            ids = [i.decode() if isinstance(i, (bytes, bytearray)) else str(i) for i in h5["ids"][:]]
            idx = {pid: i for i, pid in enumerate(ids)}
            return np.vstack([data[idx[pid]] for pid in protein_ids])

        raise RuntimeError(f"Unknown H5 format: {list(h5.keys())[:20]}")

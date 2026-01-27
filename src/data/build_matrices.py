from scipy import sparse

def build_label_space(labels_by_protein: dict[str, set[str]]):
    terms = sorted({go for gos in labels_by_protein.values() for go in gos})
    term_to_idx = {t: i for i, t in enumerate(terms)}
    return terms, term_to_idx

def build_Y(protein_ids: list[str], labels_by_protein: dict[str, set[str]], term_to_idx: dict[str, int]):
    rows, cols = [], []
    for r, pid in enumerate(protein_ids):
        gos = labels_by_protein.get(pid, set())
        for go in gos:
            c = term_to_idx.get(go)
            if c is not None:
                rows.append(r)
                cols.append(c)
    return sparse.csr_matrix(
        ([1]*len(rows), (rows, cols)),
        shape=(len(protein_ids), len(term_to_idx)),
        dtype="int8"
    )

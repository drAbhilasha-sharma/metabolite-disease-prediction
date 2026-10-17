"""Physicochemical descriptors for every compound in the candidate universe.

Computes the RDKit descriptors from SMILES for all compounds, annotated and
unannotated, so that the annotation-status audit compares the two pools on
values computed in the same way.

Writes rebuild/descriptors_all.csv
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

FUNCS = {
    "mw": Descriptors.MolWt,
    "logp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "hbd": rdMolDescriptors.CalcNumHBD,
    "hba": rdMolDescriptors.CalcNumHBA,
    "rotb": rdMolDescriptors.CalcNumRotatableBonds,
    "rings": rdMolDescriptors.CalcNumRings,
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "frac_csp3": rdMolDescriptors.CalcFractionCSP3,
    "n_hetero": lambda m: sum(1 for a in m.GetAtoms()
                              if a.GetSymbol() not in ("C", "H")),
    "formal_charge": Chem.GetFormalCharge,
}


def main():
    M = pd.read_csv("rebuild/metabolites_all.csv")[["hmdb_id", "smiles"]].dropna()
    FPI = pd.read_csv("rebuild/fp_index.csv")
    M = M[M.hmdb_id.isin(set(FPI.hmdb_id))].drop_duplicates("hmdb_id")
    rows = []
    bad = 0
    for hid, smi in zip(M.hmdb_id, M.smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad += 1
            continue
        r = {"hmdb_id": hid}
        for k, f in FUNCS.items():
            try:
                r[k] = float(f(mol))
            except Exception:
                r[k] = np.nan
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv("rebuild/descriptors_all.csv", index=False)
    print(f"descriptors for {len(D)} compounds; {bad} SMILES unparseable")


if __name__ == "__main__":
    main()

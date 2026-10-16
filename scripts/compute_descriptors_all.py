"""Compute physicochemical descriptors for the ENTIRE candidate pool.

The dataset-assembly stage computed descriptors only for annotated
metabolites. Comparing annotated against unannotated compounds on those
descriptors would therefore compare present values against absent ones and
recover missingness, not chemistry. This script computes the same descriptors
from SMILES for every compound in the pool so the comparison is real.

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

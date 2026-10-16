"""
Corrected HMDB v5.0 bulk-XML parser.

Rebuilds the metabolite-disease association table from primary source.

DESIGN NOTE (this is the fix for the audit's F1 bug):
The original parser accumulated results in module-level containers while
iterating, and its disease-name variable was not reliably reset per record,
producing a 20,017-row contiguous block in which one disease name was carried
forward across unrelated metabolites. Here, every <metabolite> record is parsed
by a PURE function (parse_one_record) that receives only that record's element
and returns a fresh dict. No state whatsoever survives between records, so the
carry-forward failure mode is structurally impossible rather than merely fixed.

We additionally record, per association, the PubMed references HMDB attaches to
each disease link, which the original parse discarded. This gives an evidence
count usable for filtering/weighting.
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import lxml.etree as ET

XML_PATH = Path("raw/data_raw/HMDB/hmdb_metabolites.xml")
OUT_DIR = Path("rebuild")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) and "}" in tag else tag


def parse_one_record(elem):
    """PURE per-record parse. Returns a dict built from `elem` alone.

    No default/carry-over values: every field starts as None/empty inside this
    call, so nothing from a previous record can leak in.
    """
    rec = {
        "accession": None,
        "name": None,
        "smiles": None,
        "inchikey": None,
        "kegg_id": None,
        "chebi_id": None,
        "super_class": None,
        "class": None,
        "diseases": [],   # list of (disease_name, n_pubmed_refs)
    }

    for child in elem:
        tag = local(child.tag)
        text = (child.text or "").strip()

        if tag == "accession" and rec["accession"] is None:
            rec["accession"] = text
        elif tag == "name" and rec["name"] is None:
            rec["name"] = text
        elif tag == "smiles" and rec["smiles"] is None:
            rec["smiles"] = text or None
        elif tag == "inchikey" and rec["inchikey"] is None:
            rec["inchikey"] = text or None
        elif tag == "kegg_id" and rec["kegg_id"] is None:
            rec["kegg_id"] = text or None
        elif tag == "chebi_id" and rec["chebi_id"] is None:
            rec["chebi_id"] = text or None
        elif tag == "taxonomy":
            for t in child:
                tt = local(t.tag)
                if tt == "super_class" and rec["super_class"] is None:
                    rec["super_class"] = (t.text or "").strip() or None
                elif tt == "class" and rec["class"] is None:
                    rec["class"] = (t.text or "").strip() or None
        elif tag == "diseases":
            for dis_elem in child:
                if local(dis_elem.tag) != "disease":
                    continue
                # dname is scoped to THIS disease element only
                dname = None
                n_refs = 0
                for d_child in dis_elem:
                    dt = local(d_child.tag)
                    if dt == "name" and dname is None:
                        dname = (d_child.text or "").strip() or None
                    elif dt == "references":
                        n_refs = sum(
                            1 for r in d_child
                            if local(r.tag) == "reference"
                            and any(local(x.tag) == "pubmed_id" and (x.text or "").strip()
                                    for x in r)
                        )
                if dname:
                    rec["diseases"].append((dname, n_refs))

    return rec


def main():
    if not XML_PATH.exists():
        raise SystemExit(f"missing {XML_PATH}")

    met_f = open(OUT_DIR / "metabolites_all.csv", "w", newline="")
    met_w = csv.writer(met_f)
    met_w.writerow(["hmdb_id", "name", "smiles", "inchikey", "kegg_id",
                    "chebi_id", "super_class", "chem_class", "n_diseases"])

    assoc_f = open(OUT_DIR / "associations_all.csv", "w", newline="")
    assoc_w = csv.writer(assoc_f)
    assoc_w.writerow(["hmdb_id", "metabolite_name", "disease_name_raw", "n_pubmed_refs"])

    disease_counter = Counter()
    n_met = 0
    n_assoc = 0
    n_with_smiles = 0

    # Track consecutive-identical-disease runs to positively demonstrate the
    # absence of the carry-forward artifact in this parse.
    last_single_disease = None
    run_len = 0
    max_run = 0
    max_run_name = None

    context = ET.iterparse(str(XML_PATH), events=("end",), huge_tree=True,
                           tag="{http://www.hmdb.ca}metabolite")

    for _, elem in context:
        rec = parse_one_record(elem)

        if rec["accession"]:
            n_met += 1
            if rec["smiles"]:
                n_with_smiles += 1
            met_w.writerow([rec["accession"], rec["name"], rec["smiles"],
                            rec["inchikey"], rec["kegg_id"], rec["chebi_id"],
                            rec["super_class"], rec["class"], len(rec["diseases"])])

            for dname, nrefs in rec["diseases"]:
                assoc_w.writerow([rec["accession"], rec["name"], dname, nrefs])
                disease_counter[dname] += 1
                n_assoc += 1

            # run-length audit
            names = {d for d, _ in rec["diseases"]}
            cur = next(iter(names)) if len(names) == 1 else None
            if cur is not None and cur == last_single_disease:
                run_len += 1
            else:
                run_len = 1
                last_single_disease = cur
            if cur is not None and run_len > max_run:
                max_run, max_run_name = run_len, cur

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n_met % 20000 == 0:
            print(f"  ...{n_met} metabolites, {n_assoc} associations", flush=True)

    met_f.close()
    assoc_f.close()

    with open(OUT_DIR / "parse_summary.txt", "w") as f:
        f.write(f"metabolites_parsed\t{n_met}\n")
        f.write(f"metabolites_with_smiles\t{n_with_smiles}\n")
        f.write(f"associations_parsed\t{n_assoc}\n")
        f.write(f"unique_disease_strings\t{len(disease_counter)}\n")
        f.write(f"max_consecutive_same_single_disease_run\t{max_run}\t{max_run_name}\n")
        f.write("\ntop_50_diseases_by_metabolite_count\n")
        for d, c in disease_counter.most_common(50):
            f.write(f"{c}\t{d}\n")

    print(f"DONE metabolites={n_met} assoc={n_assoc} diseases={len(disease_counter)}")
    print(f"max consecutive same-single-disease run: {max_run} ({max_run_name})")


if __name__ == "__main__":
    main()

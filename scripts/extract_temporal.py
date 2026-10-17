"""Extract per-association publication years from the source release.

Each <disease> block inside a metabolite record carries a <references> list
whose <reference_text> entries are formatted citations ending in a journal
name and a year. The earliest year across a disease block's references is the
year that association entered the literature; the latest is the most recent
supporting evidence. Both are recorded so that a prospective (temporal)
holdout can be constructed: train on associations whose evidence predates a
cutoff, test on those first reported after it.

Writes rebuild/assoc_years.csv with one row per (metabolite, disease) pair.
"""
import csv
import os
import re

import xml.etree.ElementTree as ET

SRC = "raw/data_raw/HMDB/hmdb_metabolites.xml"
OUT = "rebuild/assoc_years.csv"
NS = "{http://www.hmdb.ca}"

YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def strip(tag):
    return tag.replace(NS, "")


def main():
    os.makedirs("rebuild", exist_ok=True)
    n_met = n_row = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hmdb_id", "disease", "n_refs", "year_min", "year_max"])
        ctx = ET.iterparse(SRC, events=("end",))
        for _, el in ctx:
            if strip(el.tag) != "metabolite":
                continue
            n_met += 1
            acc = el.findtext(f"{NS}accession")
            dis_root = el.find(f"{NS}diseases")
            if acc and dis_root is not None:
                for d in dis_root.findall(f"{NS}disease"):
                    name = (d.findtext(f"{NS}name") or "").strip()
                    if not name:
                        continue
                    years = []
                    refs = d.find(f"{NS}references")
                    n_ref = 0
                    if refs is not None:
                        for r in refs.findall(f"{NS}reference"):
                            n_ref += 1
                            txt = r.findtext(f"{NS}reference_text") or ""
                            ys = [int(y) for y in YEAR.findall(txt)]
                            if ys:
                                years.append(max(ys))
                    w.writerow([acc, name, n_ref,
                                min(years) if years else "",
                                max(years) if years else ""])
                    n_row += 1
            el.clear()
            if n_met % 20000 == 0:
                print(f"  {n_met} metabolites, {n_row} disease blocks", flush=True)
    print(f"done: {n_met} metabolites, {n_row} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

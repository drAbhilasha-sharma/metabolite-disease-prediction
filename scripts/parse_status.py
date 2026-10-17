import lxml.etree as ET, csv
NS="{http://www.hmdb.ca}"
out=open("rebuild/metabolite_status.csv","w",newline=""); w=csv.writer(out)
w.writerow(["hmdb_id","status","n_biospecimen","has_normal_conc","has_abnormal_conc"])
ctx=ET.iterparse("raw/data_raw/HMDB/hmdb_metabolites.xml",events=("end",),huge_tree=True,tag=NS+"metabolite")
n=0
for _,e in ctx:
    acc=None;status=None;bios=set();nc=0;ac=0
    for c in e:
        t=c.tag.rsplit('}',1)[-1]
        if t=="accession" and acc is None: acc=(c.text or "").strip()
        elif t=="status" and status is None: status=(c.text or "").strip()
        elif t=="biological_properties":
            for b in c:
                if b.tag.rsplit('}',1)[-1]=="biospecimen_locations":
                    for x in b: bios.add((x.text or "").strip())
        elif t=="normal_concentrations": nc=len(list(c))
        elif t=="abnormal_concentrations": ac=len(list(c))
    if acc: w.writerow([acc,status,len(bios),nc,ac])
    n+=1; e.clear()
    while e.getprevious() is not None: del e.getparent()[0]
out.close(); print("done",n)

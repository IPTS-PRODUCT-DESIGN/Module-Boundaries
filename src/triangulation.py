"""
TRIANGULATION - Is the notion of a module unambiguous?

Two INDEPENDENT module assignments of the same product:
A) CAD assembly hierarchy (design structure)
B) assembly manual (assembly sequence, 263 pages, 16 chapters)

Both come from the same development team but serve different purposes.
If they diverge strongly, "module" is not uniquely determined -
and that holds independently of any algorithm.

Additionally: how strongly do the four clustering methods agree AMONG
themselves? That too tests uniqueness - this time within the geometric
notion of a module.

Note: the name matching between CAD and manual is deliberately weak and
counts as a failed attempt (see README, section "Known limitations").
"""
import json
import os
import re
import sys
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import SpectralClustering

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_real import klassifiziere, kontaktgraph

SEED = 42
DATA = os.environ.get("MODULGRENZEN_DATA", "data")
PDF = os.environ.get(
    "MODULGRENZEN_MANUAL",
    os.path.join(DATA, "voron", "Manual", "Assembly_Manual_2.4r2.pdf"))

# table of contents of the manual (page 3), start pages
KAPITEL = [
    ("Introduction", 4), ("Hardware", 7), ("Frame", 12),
    ("Z Drives and Idlers", 22), ("Build Plate", 52),
    ("A/B Drives and Idlers", 62), ("Gantry", 82), ("Z Axis", 108),
    ("A/B Belts", 124), ("Stealthburner", 146), ("Electronics", 148),
    ("Controller", 174), ("Wiring", 180), ("Skirts", 212),
    ("Panels", 240), ("Next Steps", 260),
]

def kapitel_von_seite(s):
    letzte = KAPITEL[0][0]
    for name, start in KAPITEL:
        if s >= start:
            letzte = name
        else:
            break
    return letzte

def seiten_text():
    from pypdf import PdfReader
    r = PdfReader(PDF)
    return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(r.pages)]

def normalisiere(n):
    """simplify part names for the text matching"""
    n = re.sub(r"[:(]\s*\d+\s*\)?$", "", n).strip()   # drop instance numbers
    n = re.sub(r"\s+v\d+$", "", n)                     # drop version suffix
    n = re.sub(r"[_\-]+", " ", n)
    return n.strip().lower()

if __name__ == "__main__":
    teile = json.load(open(os.path.join(DATA, "voron_step", "teile.json")))
    for t in teile:
        t["art"] = klassifiziere(t["name"])
        b = t["bbox"]
        t["volumen"] = max((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), 1e-9)

    print("=" * 76)
    print("TRIANGULATION - Voron 2.4r2")
    print("=" * 76)

    print("\nReading assembly manual (263 pages) ...")
    seiten = seiten_text()
    print(f" Chapters per table of contents: {len(KAPITEL)}")

    # --- assign parts to chapters via name occurrence ---
    txt_pro_seite = {s: t.lower() for s, t in seiten}
    treffer = defaultdict(Counter)
    namen = {}
    for i, t in enumerate(teile):
        namen[i] = normalisiere(t["name"])

    kandidaten = {i: n for i, n in namen.items() if len(n) >= 5}
    print(f" Part names of usable length: {len(kandidaten)} "
          f"of {len(teile)}")

    for s, txt in txt_pro_seite.items():
        if s < 7:
            continue
        kap = kapitel_von_seite(s)
        for i, n in kandidaten.items():
            if n in txt:
                treffer[i][kap] += 1

    zugeordnet = {i: c.most_common(1)[0][0] for i, c in treffer.items() if c}
    print(f" Parts uniquely assigned to a chapter: {len(zugeordnet)}")

    if len(zugeordnet) < 40:
        print("\n Too few matches for a reliable evaluation.")
        print(" -> name matching between CAD and manual too weak.")
    else:
        idx = sorted(zugeordnet)
        cad = [teile[i]["modul"].replace(":1", "") for i in idx]
        man = [zugeordnet[i] for i in idx]

        print("\n" + "=" * 76)
        print("A) CAD HIERARCHY versus ASSEMBLY MANUAL")
        print("=" * 76)
        print(f" parts compared: {len(idx)}")
        print(f" groups CAD: {len(set(cad))} | groups manual: {len(set(man))}")
        wert = ari(cad, man)
        print(f"\n agreement ARI = {wert:+.3f}")

        print("\n breakdown per CAD module:")
        vert = defaultdict(Counter)
        for c, m in zip(cad, man):
            vert[c][m] += 1
        for c, cnt in sorted(vert.items(), key=lambda x: -sum(x[1].values())):
            ges = sum(cnt.values())
            if ges < 3:
                continue
            top, n = cnt.most_common(1)[0]
            print(f"   {c[:26]:28s} {ges:4d} parts -> {len(cnt)} chapters, "
                  f"largest: {top[:22]:24s} {100*n/ges:4.1f} %")

    # ---------------------------------------------------------------
    # B) Do the clustering methods agree among each other?
    # ---------------------------------------------------------------
    print("\n" + "=" * 76)
    print("B) UNIQUENESS WITHIN THE GEOMETRIC NOTION OF A MODULE")
    print("   How strongly do the four methods agree AMONG themselves?")
    print("=" * 76)

    G = kontaktgraph(teile)
    H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    kn = list(H.nodes())
    k = len(set(H.nodes[n]["modul"] for n in kn))

    part = {}
    c = nx.community.louvain_communities(H, weight="weight",
                                         resolution=1.0, seed=SEED)
    part["Louvain"] = [{n: i for i, cc in enumerate(c) for n in cc}[n] for n in kn]
    c = nx.community.greedy_modularity_communities(H, weight="weight")
    part["Greedy"] = [{n: i for i, cc in enumerate(c) for n in cc}[n] for n in kn]
    c = list(nx.community.asyn_lpa_communities(H, weight="weight", seed=SEED))
    part["LabelProp"] = [{n: i for i, cc in enumerate(c) for n in cc}[n] for n in kn]
    A = nx.to_numpy_array(H, nodelist=kn, weight="weight")
    part["Spektral"] = list(SpectralClustering(
        n_clusters=k, affinity="precomputed", random_state=SEED).fit_predict(A))

    algs = list(part)
    print(f"\n {'':14s}" + "".join(f"{a:>13s}" for a in algs))
    werte = []
    for a in algs:
        z = f" {a:14s}"
        for b in algs:
            v = ari(part[a], part[b])
            z += f"{v:13.3f}"
            if a != b:
                werte.append(v)
        print(z)
    print(f"\n Mean agreement of the methods among themselves: "
          f"{np.mean(werte):.3f}")
    print(f" Range: {min(werte):.3f} to {max(werte):.3f}")

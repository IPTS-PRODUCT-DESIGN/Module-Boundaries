"""
SUBSTANTIATION OF THE FINDING

Three counter-checks to substantiate the finding:

A) NULL HYPOTHESIS - What do naive methods achieve WITHOUT a contact graph?
   Without this baseline the measured value is not interpretable.
   - random assignment (trivial lower bound)
   - k-Means on centroids (position only, no contact knowledge)
   - spatial grid (position only, no learning)
   - volume/size (control: should fail)

B) METHOD COMPARISON - Is the finding a statement about geometry
   or only about Louvain?
   Louvain / Greedy Modularity / Label Propagation / Spectral

C) MULTI-LEVEL COMPARISON - The CAD hierarchy has several levels.
   So far only the top one was compared.
"""
import json
import os
import sys
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_real import klassifiziere, kontaktgraph

SEED = 42
RES = [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0]

DATA = os.environ.get("MODULGRENZEN_DATA", "data")
GERAETE = [
    ("Voron 2.4r2",      os.path.join(DATA, "voron_step",      "teile.json")),
    ("Voron Trident",    os.path.join(DATA, "trident_step",    "teile.json")),
    ("Voron Switchwire", os.path.join(DATA, "switchwire_step", "teile.json")),
]

def lade(pfad):
    teile = json.load(open(pfad))
    for t in teile:
        t["art"] = klassifiziere(t["name"])
        b = t["bbox"]
        t["volumen"] = max((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), 1e-9)
        t["zentrum"] = [(b[0]+b[3])/2, (b[1]+b[4])/2, (b[2]+b[5])/2]
        # hierarchy levels: path alternates reference/target -> every 2nd step
        p = t["pfad"]
        t["ebene1"] = p[2] if len(p) > 2 else "ROOT"
        t["ebene2"] = p[4] if len(p) > 4 else t["ebene1"]
        t["ebene3"] = p[6] if len(p) > 6 else t["ebene2"]
    return teile

# ===================================================================
# A) NULL HYPOTHESES
# ===================================================================

def nullmodelle(teile, idx, k, truth, rng):
    """naive methods WITHOUT any contact information"""
    Z = np.array([teile[i]["zentrum"] for i in idx])
    V = np.array([[np.log10(teile[i]["volumen"])] for i in idx])
    out = {}

    # 1. random - trivial lower bound
    werte = [ari(truth, rng.integers(0, k, len(truth))) for _ in range(20)]
    out["Zufall"] = float(np.mean(werte))

    # 2. k-Means on part centroids (position only)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    out["k-Means Lage"] = ari(truth, km.labels_)

    # 3. spatial grid (position without learning)
    n_pro_achse = max(int(round(k ** (1/3))), 1)
    lab = np.zeros(len(Z), dtype=int)
    for d in range(3):
        kanten = np.quantile(Z[:, d], np.linspace(0, 1, n_pro_achse + 1)[1:-1])
        lab = lab * n_pro_achse + np.digitize(Z[:, d], kanten)
    out["Raumgitter"] = ari(truth, lab)

    # 4. part size (control - should fail)
    km2 = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(V)
    out["Bauteilgroesse"] = ari(truth, km2.labels_)

    return out

# ===================================================================
# B) METHOD COMPARISON on the same contact graph
# ===================================================================

def verfahren(H, knoten, truth, k):
    out = {}

    best = -1
    for r in RES:
        c = nx.community.louvain_communities(H, weight="weight",
                                             resolution=r, seed=SEED)
        lab = {n: i for i, cc in enumerate(c) for n in cc}
        best = max(best, ari(truth, [lab[n] for n in knoten]))
    out["Louvain"] = best

    c = nx.community.greedy_modularity_communities(H, weight="weight")
    lab = {n: i for i, cc in enumerate(c) for n in cc}
    out["Greedy Modularity"] = ari(truth, [lab[n] for n in knoten])

    c = list(nx.community.asyn_lpa_communities(H, weight="weight", seed=SEED))
    lab = {n: i for i, cc in enumerate(c) for n in cc}
    out["Label Propagation"] = ari(truth, [lab[n] for n in knoten])

    try:
        A = nx.to_numpy_array(H, nodelist=knoten, weight="weight")
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=SEED, assign_labels="kmeans")
        out["Spektral"] = ari(truth, sc.fit_predict(A))
    except Exception:
        out["Spektral"] = float("nan")

    return out

# ===================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    zeilen_null, zeilen_verf, zeilen_ebene = [], [], []

    for name, pfad in GERAETE:
        print(f"\nComputing {name} ...")
        teile = lade(pfad)
        G = kontaktgraph(teile)
        H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        knoten = list(H.nodes())
        truth = [H.nodes[n]["modul"] for n in knoten]
        k = len(set(truth))

        null = nullmodelle(teile, knoten, k, truth, rng)
        verf = verfahren(H, knoten, truth, k)

        zeilen_null.append((name, null))
        zeilen_verf.append((name, verf))

        eb = {}
        for lvl in ["ebene1", "ebene2", "ebene3"]:
            t_lvl = [teile[n][lvl] for n in knoten]
            n_grp = len(set(t_lvl))
            best = -1
            for r in RES:
                c = nx.community.louvain_communities(H, weight="weight",
                                                     resolution=r, seed=SEED)
                lab = {n: i for i, cc in enumerate(c) for n in cc}
                best = max(best, ari(t_lvl, [lab[n] for n in knoten]))
            eb[lvl] = (n_grp, best)
        zeilen_ebene.append((name, eb))

    # ---------------- Output ----------------
    print("\n" + "=" * 78)
    print("A) NULL HYPOTHESIS - baselines without a contact graph")
    print("=" * 78)
    print(f"{'Method':22s}" + "".join(f"{n[:14]:>16s}" for n, _ in zeilen_null))
    print("-" * 78)
    reihen = ["Zufall", "Bauteilgroesse", "Raumgitter", "k-Means Lage"]
    beschriftung = {"Zufall": "Random", "Bauteilgroesse": "Part size",
                    "Raumgitter": "Spatial grid", "k-Means Lage": "k-Means position"}
    for r in reihen:
        print(f"{beschriftung[r]:22s}" + "".join(f"{d[r]:16.3f}" for _, d in zeilen_null))
    print(f"{'CONTACT GRAPH (Louvain)':22s}" +
          "".join(f"{d['Louvain']:16.3f}" for _, d in zeilen_verf))
    print("-" * 78)
    gew = [d["Louvain"] for _, d in zeilen_verf]
    bestnull = [max(d["k-Means Lage"], d["Raumgitter"]) for _, d in zeilen_null]
    print(f"{'Lead over best':22s}" +
          "".join(f"{g-b:+16.3f}" for g, b in zip(gew, bestnull)))
    print(f"{' naive method':22s}")
    faktor = np.mean([g / b if b > 0.01 else float('inf')
                      for g, b in zip(gew, bestnull)])
    print(f"\n Mean factor contact graph / best naive method: "
          f"{faktor:.2f}x")

    print("\n" + "=" * 78)
    print("B) METHOD COMPARISON - same graph, different algorithms")
    print("=" * 78)
    print(f"{'Algorithm':22s}" + "".join(f"{n[:14]:>16s}" for n, _ in zeilen_verf))
    print("-" * 78)
    for alg in ["Louvain", "Greedy Modularity", "Label Propagation", "Spektral"]:
        print(f"{alg:22s}" + "".join(f"{d[alg]:16.3f}" for _, d in zeilen_verf))
    alle = [d[a] for _, d in zeilen_verf
            for a in ["Louvain", "Greedy Modularity", "Label Propagation", "Spektral"]
            if not np.isnan(d[a])]
    print("-" * 78)
    print(f" Range over all methods and machines: "
          f"{min(alle):.3f} to {max(alle):.3f}")

    print("\n" + "=" * 78)
    print("C) MULTI-LEVEL COMPARISON - the CAD hierarchy has several levels")
    print("=" * 78)
    print(f"{'Machine':20s} {'Level 1':>18s} {'Level 2':>18s} {'Level 3':>18s}")
    print("-" * 78)
    for name, eb in zeilen_ebene:
        s = f"{name[:19]:20s}"
        for lvl in ["ebene1", "ebene2", "ebene3"]:
            n_grp, a = eb[lvl]
            s += f"{a:11.3f} ({n_grp:3d})"
        print(s)
    print("\n (number in parentheses = number of groups at this level)")

    json.dump({
        "null": [(n, d) for n, d in zeilen_null],
        "verfahren": [(n, d) for n, d in zeilen_verf],
        "ebenen": [(n, {k2: list(v) for k2, v in d.items()})
                   for n, d in zeilen_ebene],
    }, open("absicherung.json", "w"), indent=1)
    print("\nSaved: absicherung.json")

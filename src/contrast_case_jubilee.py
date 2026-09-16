"""
CONTRAST CASE: Jubilee (Machine Agency Lab, University of Washington)

First check outside the Voron family. Differences from the previous data set:
- different team, different institution (university lab instead of community)
- different CAD system (SolidWorks instead of Fusion 360)
- different machine class (toolchanger platform instead of a plain printer)
- different scale (755 instead of 712-1428 parts)

The same tests as for the Voron machines are computed:
stability over random seeds, comparison graph, method matrix,
granularity test.
"""
import json
import os
import sys
import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import SpectralClustering

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_real import klassifiziere, kontaktgraph

SEED = 42
N_SEEDS = 12
DATA = os.environ.get("MODULGRENZEN_DATA", "data")

def lade(pfad, ebene):
    """ebene: index in the path from which the module assignment applies.
    Fusion 360 creates named reference labels (level 1),
    SolidWorks does not (level 2 there)."""
    teile = json.load(open(pfad))
    for t in teile:
        t["art"] = klassifiziere(t["name"])
        b = t["bbox"]
        t["volumen"] = max((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), 1e-9)
        p = t["pfad"]
        t["modul"] = p[ebene] if len(p) > ebene else "ROOT"
    return teile

def labels(comms, kn):
    m = {n: i for i, c in enumerate(comms) for n in c}
    return [m[n] for n in kn]

def analysiere(name, pfad, ebene):
    teile = lade(pfad, ebene)
    G = kontaktgraph(teile)
    H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    kn = list(H.nodes())
    truth = [teile[n]["modul"] for n in kn]
    k_cad = len(set(truth))

    print(f"\n{'='*74}\n{name}\n{'='*74}")
    print(f" parts total           : {len(teile)}")
    print(f" in largest subgraph   : {len(kn)}")
    print(f" contacts              : {H.number_of_edges()}")
    print(f" modules per CAD        : {k_cad}")

    # --- stability at resolution 1.0 ---
    parts = [labels(nx.community.louvain_communities(
        H, weight="weight", resolution=1.0, seed=s), kn)
        for s in range(N_SEEDS)]
    selbst = [ari(a, b) for a, b in combinations(parts, 2)]
    gegen = [ari(truth, p) for p in parts]

    # --- comparison graph ---
    grade = [d for _, d in H.degree()]
    null = []
    for rep in range(4):
        CM = nx.Graph(nx.configuration_model(grade, seed=200 + rep))
        CM.remove_edges_from(nx.selfloop_edges(CM))
        knc = list(CM.nodes())
        ps = [labels(nx.community.louvain_communities(CM, resolution=1.0, seed=s), knc)
              for s in range(4)]
        null += [ari(a, b) for a, b in combinations(ps, 2)]

    # --- granularity aligned ---
    best = None
    for r in np.arange(0.05, 1.55, 0.05):
        ks = [len(nx.community.louvain_communities(H, weight="weight",
              resolution=float(r), seed=s)) for s in range(4)]
        d = abs(np.mean(ks) - k_cad)
        if best is None or d < best[1]:
            best = (float(r), d, np.mean(ks))
    r_ang = best[0]
    pa = [labels(nx.community.louvain_communities(
        H, weight="weight", resolution=r_ang, seed=s), kn)
        for s in range(N_SEEDS)]
    selbst_a = [ari(a, b) for a, b in combinations(pa, 2)]
    gegen_a = [ari(truth, p) for p in pa]

    # --- method matrix ---
    P = {}
    P["Louvain"] = labels(nx.community.louvain_communities(
        H, weight="weight", resolution=1.0, seed=SEED), kn)
    P["Greedy"] = labels(nx.community.greedy_modularity_communities(
        H, weight="weight"), kn)
    P["LabelProp"] = labels(list(nx.community.asyn_lpa_communities(
        H, weight="weight", seed=SEED)), kn)
    A = nx.to_numpy_array(H, nodelist=kn, weight="weight")
    P["Spektral"] = list(SpectralClustering(n_clusters=k_cad,
        affinity="precomputed", random_state=SEED).fit_predict(A))
    algs = list(P)
    inter = [ari(P[a], P[b]) for a, b in combinations(algs, 2)]
    gegen_alle = {a: ari(truth, P[a]) for a in algs}

    print(f"\n resolution 1.0")
    print(f"   self             : {np.mean(selbst):.3f} "
          f"(min {min(selbst):.3f})")
    print(f"   vs. CAD          : {np.mean(gegen):.3f}")
    print(f"   comparison graph : {np.mean(null):.3f}")
    print(f"   gap              : {np.mean(selbst)-np.mean(gegen):.3f}")
    print(f"\n cluster count aligned (res {r_ang:.2f} -> {best[2]:.1f} clusters)")
    print(f"   self             : {np.mean(selbst_a):.3f} "
          f"(min {min(selbst_a):.3f})")
    print(f"   vs. CAD          : {np.mean(gegen_a):.3f}")
    print(f"   gap              : {np.mean(selbst_a)-np.mean(gegen_a):.3f}")
    print(f"\n method matrix")
    print(f"   among each other    : {np.mean(inter):.3f}")
    print(f"   vs. CAD (mean)      : {np.mean(list(gegen_alle.values())):.3f}")
    for a in algs:
        print(f"   {a:12s} : {gegen_alle[a]:.3f}")

    return dict(name=name, n=len(teile), n_graph=len(kn), k_cad=k_cad,
                selbst=float(np.mean(selbst)), selbst_min=float(min(selbst)),
                gegen=float(np.mean(gegen)), null=float(np.mean(null)),
                selbst_a=float(np.mean(selbst_a)), gegen_a=float(np.mean(gegen_a)),
                inter=float(np.mean(inter)),
                gegen_alle=float(np.mean(list(gegen_alle.values()))),
                res_ang=r_ang)

if __name__ == "__main__":
    print("=" * 74)
    print("CONTRAST CASE OUTSIDE THE VORON FAMILY")
    print("=" * 74)

    r = analysiere("Jubilee (UW Machine Agency, SolidWorks)",
                   os.path.join(DATA, "jubilee_step", "teile.json"), ebene=2)

    print("\n" + "=" * 74)
    print("COMPARISON WITH THE VORON FAMILY")
    print("=" * 74)
    voron = json.load(open("stabilitaet.json"))
    print(f"{'Machine':30s}{'self':>9s}{'vs.CAD':>9s}"
          f"{'null':>8s}{'gap':>9s}")
    print("-" * 74)
    for k, v in voron.items():
        print(f"{k:30s}{v['t1']:9.3f}{v['gt']:9.3f}"
              f"{v['t3']:8.3f}{v['t1']-v['gt']:9.3f}")
    print("-" * 74)
    print(f"{'Jubilee (different team)':30s}{r['selbst']:9.3f}{r['gegen']:9.3f}"
          f"{r['null']:8.3f}{r['selbst']-r['gegen']:9.3f}")

    vs = np.mean([v['t1'] for v in voron.values()])
    vg = np.mean([v['gt'] for v in voron.values()])
    print("\n" + "=" * 74)
    print("FINDING")
    print("=" * 74)
    print(f" Voron family : self {vs:.3f} | vs. CAD {vg:.3f} | "
          f"gap {vs-vg:.3f}")
    print(f" Jubilee      : self {r['selbst']:.3f} | "
          f"vs. CAD {r['gegen']:.3f} | gap {r['selbst']-r['gegen']:.3f}")
    abw = abs((r['selbst']-r['gegen']) - (vs-vg))
    if abw < 0.12:
        print(f"\n -> The gap is confirmed outside the Voron family")
        print(f"    (deviation {abw:.3f}). The finding is not limited to a")
        print(f"    single design community.")
    else:
        print(f"\n -> The gap deviates markedly ({abw:.3f}).")
        print(f"    The finding could be design-culture dependent.")

    json.dump(r, open("jubilee_ergebnis.json", "w"), indent=1)
    print("\nSaved: jubilee_ergebnis.json")

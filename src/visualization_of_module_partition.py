"""
VISUALISATION OF THE MODULE PARTITION - real part geometry

Draws the actual part contours, not bounding boxes.
The solids are tessellated with BRepMesh and projected isometrically.

Produces a figure with four panels:
  1.   overview of the whole machine, coloured by designer modules
  2-4. three selected modules, coloured by Louvain cluster

Usage (from the project folder):
    python3 src/visualisierung_en.py data/raw/Voron_2.4r2_Assembly.step grafiken/module.png

Runtime: 8-15 minutes. Tessellation is the slow part.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from collections import Counter, defaultdict

import networkx as nx
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopLoc import TopLoc_Location
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool

# ---- settings ------------------------------------------------------
DEFLECTION = 3.0   # mm, coarser = faster
# Fusion 360 creates named reference labels (level 1),
# SolidWorks does not (level 2 there). Detected automatically.
MODULEBENE = None   # None = automatic, otherwise 1 or 2
RESOLUTION = 0.8
SEED = 42
PAL = ["#1f4e79", "#e07a5f", "#2a9d8f", "#f2cc8f", "#9b5de5",
       "#d1495b", "#8ab6d6", "#588157", "#bc6c25", "#7f7f7f",
       "#264653", "#e9c46a", "#457b9d"]

# ---- read STEP -----------------------------------------------------

def label_name(l):
    n = TDataStd_Name()
    return n.Get().ToExtString() if l.FindAttribute(TDataStd_Name.GetID_s(), n) else "?"

def lade(pfad):
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-CAF"), doc)
    r = STEPCAFControl_Reader()
    r.SetNameMode(True); r.SetColorMode(False); r.SetLayerMode(False)
    if not r.ReadFile(pfad):
        raise RuntimeError("Reading failed")
    r.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    teile = []

    def geh(lab, loc, pf):
        nm = label_name(lab); neu = pf + [nm]
        if st.IsAssembly_s(lab):
            k = TDF_LabelSequence(); st.GetComponents_s(lab, k)
            for i in range(1, k.Length() + 1):
                geh(k.Value(i), loc, neu)
            return
        if st.IsReference_s(lab):
            ref = TDF_Label(); st.GetReferredShape_s(lab, ref)
            rl = st.GetLocation_s(lab); rn = label_name(ref)
            geh(ref, loc.Multiplied(rl), pf + [nm if nm not in ("?", "") else rn])
            return
        sh = st.GetShape_s(lab)
        if sh.IsNull():
            return
        sh = sh.Moved(loc)
        bx = Bnd_Box()
        try:
            BRepBndLib.Add_s(sh, bx, False)
            if bx.IsVoid():
                return
        except Exception:
            return
        teile.append({"name": neu[-1], "pfad": neu,
                      "bbox": [round(v, 3) for v in bx.Get()], "shape": sh})

    frei = TDF_LabelSequence(); st.GetFreeShapes(frei)
    for i in range(1, frei.Length() + 1):
        geh(frei.Value(i), TopLoc_Location(), [])

    # determine module level: choose the level with meaningful names
    eb = MODULEBENE
    if eb is None:
        def sprechend(e):
            n = [t["pfad"][e] for t in teile if len(t["pfad"]) > e]
            if not n:
                return 0.0
            return sum(0 if x.startswith("NAUO") else 1 for x in n) / len(n)
        eb = 1 if sprechend(1) >= 0.5 else 2
        print(f" module level determined automatically: {eb}")
    for t in teile:
        t["modul"] = t["pfad"][eb] if len(t["pfad"]) > eb else "ROOT"
    return teile

# ---- contact graph (identical to poc_real.py) ----------------------

def kontaktgraph(teile, tol=1.0):
    B = np.array([t["bbox"] for t in teile]); lo, hi = B[:, :3], B[:, 3:]
    G = nx.Graph()
    for i, t in enumerate(teile):
        b = t["bbox"]
        G.add_node(i, vol=max((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), 1e-9))
    for i in range(len(teile)):
        j = np.arange(i + 1, len(teile))
        if not len(j):
            break
        ov = np.all((lo[i] - tol <= hi[j]) & (hi[i] + tol >= lo[j]), axis=1)
        for jj in j[ov]:
            v = min(G.nodes[i]["vol"], G.nodes[int(jj)]["vol"])
            G.add_edge(i, int(jj), weight=1.0 / (1.0 + np.log10(max(v, 1))))
    return G

# ---- tessellation and isometric projection -------------------------

def isometrisch(p):
    """3D -> 2D, standard isometry"""
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    return np.column_stack([(p[:, 0] - p[:, 1]) * c,
                            p[:, 2] + (p[:, 0] + p[:, 1]) * s])

def dreiecke(shape, defl=DEFLECTION):
    """tessellate a solid and return triangles as 2D polygons"""
    try:
        BRepMesh_IncrementalMesh(shape, defl, False, 0.5, True)
    except Exception:
        return [], []
    polys, tiefen = [], []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        face = TopoDS.Face_s(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        ex.Next()
        if tri is None:
            continue
        trsf = loc.Transformation()
        pts = []
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            pts.append([p.X(), p.Y(), p.Z()])
        pts = np.array(pts)
        if len(pts) == 0:
            continue
        pts2 = isometrisch(pts)
        for i in range(1, tri.NbTriangles() + 1):
            n1, n2, n3 = tri.Triangle(i).Get()
            polys.append(pts2[[n1 - 1, n2 - 1, n3 - 1]])
            tiefen.append(float(pts[[n1-1, n2-1, n3-1], 1].mean()))
    return polys, tiefen

def zeichne(ax, teile, idx, farbe_je_teil, lw=0.15):
    """draw the given parts, back to front"""
    alle_p, alle_c, alle_t = [], [], []
    for i in idx:
        polys, tiefen = dreiecke(teile[i]["shape"])
        c = farbe_je_teil[i]
        alle_p += polys; alle_t += tiefen; alle_c += [c] * len(polys)
    if not alle_p:
        return
    ordn = np.argsort(alle_t)[::-1]   # back first
    pc = PolyCollection([alle_p[k] for k in ordn],
                        facecolors=[alle_c[k] for k in ordn],
                        edgecolors="white", linewidths=lw, alpha=0.95)
    ax.add_collection(pc)

# ====================================================================

if __name__ == "__main__":
    pfad = sys.argv[1]
    aus = sys.argv[2] if len(sys.argv) > 2 else "module.png"

    print(f"Reading {pfad} ...", flush=True)
    teile = lade(pfad)
    print(f" parts: {len(teile)}", flush=True)

    print(" building contact graph ...", flush=True)
    G = kontaktgraph(teile)
    H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    kn = list(H.nodes())
    comms = nx.community.louvain_communities(H, weight="weight",
                                             resolution=RESOLUTION, seed=SEED)
    lab = {n: i for i, c in enumerate(comms) for n in c}
    print(f" {len(comms)} clusters found", flush=True)

    # purity per designer module
    vert = defaultdict(Counter)
    for n in kn:
        vert[teile[n]["modul"]][lab[n]] += 1
    stat = []
    for m, cnt in vert.items():
        ges = sum(cnt.values())
        if ges < 30:
            continue
        stat.append((m, ges, len(cnt), cnt.most_common(1)[0][1] / ges))
    stat.sort(key=lambda x: -x[3])

    print("\n recovery per designer module:")
    for m, ges, ncl, rein in stat:
        print(f" {m.replace(':1',''):28s} {ges:4d} parts | "
              f"{ncl:2d} clusters | {rein*100:5.1f} %", flush=True)

    gut, mittel, schlecht = stat[0], stat[len(stat) // 2], stat[-1]

    # ---- figure ----
    B = np.array([t["bbox"] for t in teile])
    ecken = np.array([[B[:, 0].min(), B[:, 1].min(), B[:, 2].min()],
                      [B[:, 3].max(), B[:, 4].max(), B[:, 5].max()]])
    gitter = np.array([[x, y, z] for x in ecken[:, 0]
                       for y in ecken[:, 1] for z in ecken[:, 2]])
    g2 = isometrisch(gitter)
    xlim = (g2[:, 0].min() - 30, g2[:, 0].max() + 30)
    ylim = (g2[:, 1].min() - 30, g2[:, 1].max() + 30)

    fig, axes = plt.subplots(1, 4, figsize=(19, 6.4))

    # panel 1: whole machine by designer modules
    print("\n drawing overview ...", flush=True)
    module = [m for m, _ in Counter(t["modul"] for t in teile).most_common()]
    fm = {m: PAL[k % len(PAL)] for k, m in enumerate(module)}
    zeichne(axes[0], teile, range(len(teile)),
            {i: fm[t["modul"]] for i, t in enumerate(teile)}, lw=0.08)
    axes[0].set_title(f"Complete machine\n{len(teile)} parts \u00b7 "
                      f"{len(module)} designer modules", fontsize=11, pad=8)
    axes[0].text(0.5, -0.05, "REFERENCE", transform=axes[0].transAxes,
                 ha="center", fontsize=12, fontweight="bold", color="#555")

    # panels 2-4: example modules by Louvain clusters
    for ax, (m, ges, ncl, rein), note, col in zip(
            axes[1:], [gut, mittel, schlecht],
            ["FULLY RECOVERED", "PARTIAL", "FRAGMENTED"],
            ["#2a9d8f", "#e0a458", "#d1495b"]):
        print(f" drawing {m} ...", flush=True)
        idx = [i for i in kn if teile[i]["modul"] == m]
        cl = [lab[i] for i in idx]
        fc = {c: PAL[k % len(PAL)]
              for k, (c, _) in enumerate(Counter(cl).most_common())}
        zeichne(ax, teile, idx, {i: fc[lab[i]] for i in idx})
        ax.set_title(f"{m.replace(':1','')}\n{ges} parts \u00b7 {ncl} clusters \u00b7 "
                     f"{rein*100:.0f} % in largest", fontsize=11, pad=8)
        ax.text(0.5, -0.05, note, transform=ax.transAxes, ha="center",
                fontsize=12, fontweight="bold", color=col)

    for ax in axes:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#ddd")

    fig.suptitle("Designer modules versus geometric module partition\n"
                 "isometric view, tessellated part geometry, "
                 "left: reference; right: one colour per Louvain cluster",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(aus, dpi=160)
    print(f"\nSaved: {aus}")

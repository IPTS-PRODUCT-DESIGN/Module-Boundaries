# Module Boundaries from CAD Geometry – Analysis Scripts

Script collection for the pre-study *"Can a module boundary be determined
from geometry?"*, August 2026.

All scripts work on publicly available full STEP assemblies and are
reproducible. Results and their interpretation are given in the accompanying
report.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install cadquery-ocp networkx scikit-learn numpy matplotlib pandas pypdf
```

`cadquery-ocp` provides the Python binding to Open CASCADE (about 400 MB).
Tested with Python 3.11 on Linux.

---

## Data sources

The STEP files are not part of this collection. Where to obtain them:

| Machine | Repository | File |
|---|---|---|
| Voron 2.4r2 | `VoronDesign/Voron-2`, branch `Voron2.4` | `CAD/Voron_2.4r2_Assembly_STEP.zip` |
| Voron Trident | `VoronDesign/Voron-Trident` | `CAD/trident_assembly.step.zip` |
| Voron Switchwire | `VoronDesign/Voron-Switchwire` | `CAD/Switchwire_Assembly_v1_STEP.zip` |
| Jubilee | `machineagency/jubilee` | `frame/cads/STEP/jubilee.STEP` |
| Assembly manual | `VoronDesign/Voron-2` | `Manual/Assembly_Manual_2.4r2.pdf` |

Sparse checkout without a full clone:

```bash
git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/VoronDesign/Voron-2
```

Unpacked sizes: Voron 2.4r2 241 MB, Trident 185 MB, Switchwire 90 MB,
Jubilee 55 MB.

---

## Directory layout

The scripts expect the extracted data under a data directory
that is set via the environment variable `MODULGRENZEN_DATA`
(default: `data`):

```
data/
├── voron_step/teile.json
├── trident_step/teile.json
├── switchwire_step/teile.json
├── jubilee_step/teile.json
└── voron/Manual/Assembly_Manual_2.4r2.pdf
```

Note: the JSON keys (`modul`, `pfad`, `bbox`, `teile`, `kanten_exakt`, …)
and the intermediate file names are kept in their original form so that
previously computed result files remain compatible.

---

## Run order

`geo_extract.py` must run first – the scripts that build on `teile.json`
depend on it.

```
geo_extract.py                creates teile.json (prerequisite for the rest)
│
├──> poc_real.py              base analysis, contact graph, classification
├──> stabilitaet.py           self-consistency, comparison graph (Table 2)
├──> absicherung.py           null models, methods, hierarchy levels
├──> gewichtung.py            weighting sensitivity (Table 3)
├──> kontrastfall.py          Jubilee against the Voron family
└──> triangulation.py         assembly manual, method matrix

Standalone, read STEP directly:
   kontaktvergleich.py        AABB / OBB / exact sample
   exakter_graph.py           complete exact contact graph → *_graph.json
   visualisierung_en.py       Figure 1 (four panels)
```

---

## The scripts in detail

### `geo_extract.py`

Reads a full STEP assembly with Open CASCADE, resolves the assembly
hierarchy recursively and computes the world coordinates and the
axis-aligned bounding box for each part instance.

```bash
python3 src/geo_extract.py data/raw/Voron_2.4r2_Assembly.step
```

Creates `teile.json` with name, hierarchy path, module assignment and
bounding box per part. Runtime 3–7 minutes depending on file size.

> **Note on the hierarchy level.** Fusion 360 creates named reference
> labels, SolidWorks does not. For Voron files the module level is at
> `pfad[1]`, for Jubilee at `pfad[2]`. A glance at the generated
> `teile.json` shows which level carries meaningful names.

### `poc_real.py`

Builds the contact graph from the bounding boxes (touch ≤ 1 mm), weights
the edges with the inverse log volume, partitions with Louvain and compares
the result against the assembly structure recorded by the designer via the
adjusted Rand index. Also contains the standard-part recognition via name
patterns.

Imported as a module by the other scripts (`kontaktgraph`, `klassifiziere`).

### `stabilitaet.py`

Four checks whether the measured deviation is an artefact of the algorithms:

* seed stability – same method, 12 random starts
* resolution stability – resolution 0.6 to 1.6
* comparison graph – configuration model with approximated degree sequence
* same objective function – Louvain against Greedy Modularity

Creates `stabilitaet.json`. Runtime about 10 minutes for three machines.

### `absicherung.py`

Three counter-checks:

* **null hypothesis** – random assignment, k-Means on centroids, spatial
  grid, part size as a control
* **method comparison** – Louvain, Greedy Modularity, Label Propagation,
  Spectral
* **multi-level comparison** – hierarchy levels 1 to 3

Creates `absicherung.json`.

### `gewichtung.py`

Recomputes the base analysis under four edge weightings (unweighted,
inverse log volume, contact-extent proxy, inverse cube-root volume) and
compares self-consistency, agreement with the CAD structure and the gap.
Produces the values for Table 3. With `--exakt` it reads the `*_graph.json`
produced by `exakter_graph.py` instead of the bounding-box contacts.

### `kontrastfall.py`

Runs the same tests for Jubilee and compares them against the Voron family
(reads `stabilitaet.json`). Contains the granularity test that chooses the
resolution so that the cluster count roughly matches the CAD module count.
Creates `jubilee_ergebnis.json`.

### `triangulation.py`

Two independent checks of uniqueness:

* **assembly manual as a second reference** – parts are matched to the 16
  chapters of the PDF via name occurrence
* **method matrix** – pairwise agreement of the four clustering methods

> The name matching is weak: parts appear on average in several chapters.
> Only the subset with a unique assignment is meaningful. The attempt is
> treated as failed and reported as such in the report.

### `kontaktvergleich.py`

Compares three stages of contact measurement against each other:

1. axis-aligned bounding box (AABB) – the method of the main pipeline
2. oriented bounding box (OBB) – rotates with the part
3. exact minimum distance via `BRepExtrema_DistShapeShape`

```bash
python3 src/kontaktvergleich.py data/raw/jubilee.STEP result.json 1.0 150
# <step> <output> <tol> <sample>
```

> The exact distance computation takes about 2 seconds per part pair. The
> sample mode (last argument) serves the quick comparison; the complete
> exact evaluation of all Jubilee pairs is provided in `exakter_graph.py`
> and reported as completed in the report.

### `exakter_graph.py`

Builds the contact graph not from bounding boxes but from the exact B-rep
minimum distances of all candidate pairs (`BRepExtrema_DistShapeShape`) and
stores a cleaned graph. Progress is checkpointed every 100 pairs; an aborted
run can be resumed.

```bash
python3 src/exakter_graph.py data/raw/jubilee.STEP ergebnisse/jubilee_exakt.json 1.0
```

Creates `jubilee_exakt.json` (raw distances) and `jubilee_exakt_graph.json`
(cleaned graph, Jubilee-exact row in Table 2). Runtime for Jubilee about
100 minutes.

### `visualisierung_en.py`

Reads the Voron 2.4r2 STEP assembly directly, tessellates the solids with
BRepMesh and projects them isometrically. Produces a figure with four
panels: the whole machine by designer modules plus three example modules,
coloured by assigned Louvain cluster (fully recovered / partial /
fragmented).

```bash
python3 src/visualisierung_en.py data/raw/Voron_2.4r2_Assembly.step grafiken/module.png
```

Runtime 8–15 minutes; tessellation is the slow part.

---

## Known limitations

| Point | Meaning |
|---|---|
| Bounding-box contact | Overestimates contacts for oblique parts. OBB removes only a few percent on Jubilee – the distortion is smaller than expected but not ruled out |
| Tolerance 1 mm | Not sensitivity-tested |
| Hierarchy level | Must be checked per CAD system (see note above) |
| Compactness measure | Self-defined, not literature-validated |

---

## Next steps

1. Add a compact assembly outside the portal-machine class
   (gearbox, pump, robot arm)
2. Collect a second human module assignment as an upper bound (inter-rater study)
3. Test tolerance sensitivity
4. Compute the exact contact graph for the Voron machines as well

---

## License and citation

The scripts are research code for the pre-study. The CAD data used are
subject to the licenses of the respective projects (VoronDesign, Machine
Agency Lab) and are not part of this collection.

Methodological foundations: Blondel et al. (2008) for Louvain, Hubert &
Arabie (1985) for the adjusted Rand index.

# EM ↔ 2p cell issue report

Follow-up on `issues/celltype_explorer_log.xlsx` (38 rows logged from
`scripts/tools/celltype_explorer/celltype_explorer.py`, 35 unique cells).

Scope: the 376 ROI-matched cells produced by `data_loader.load_df_rois_morph`
(2p ROIs that map, via the EM-2p mapping sheet, to a row in the proofread-cells
master list).

---

## What was already fixed

Three changes are applied and verified:

1. **Master list bumped to `Eyewire II Proofread Cells Main List - All Cells 2026-08-04.csv`**
   (`data_loader.MAIN_ALL_CELLS_SHEET`). The file was copied into
   `data/spreadsheets/` from
   `../data/morphologies-ew2/csv/main_all_cells/`, byte-identical
   (sha256 `85c1edc0391ae52a…`). The old 2026-07-03 sheet is left in place.
2. **`Proofread SegID` fallback added to `data_loader.add_skels`.** Previously
   the fallback in `load_df_rois_morph`'s `seg_col_master` fold used
   `combine_first`, so `Proofread SegID` was consulted only when `Latest SegID`
   was *null* — never when the ID existed but its SWC file didn't. `add_skels`
   now falls back on a missing *file*, and records which column each row's SWC
   came from in a new **`swc_segid_source`** column
   (`'Latest SegID'` / `'Proofread SegID'` / `None`).
   Pass `fallback_seg_col=None` to restore the old behaviour.
3. **IPL profile cache regenerated** (`scripts/preprocessing/compute_ipl_profiles.py`).
   Profiles are keyed by `Latest SegID`, and the sheet update changed SegIDs, which
   had stranded 17 previously-cached profiles.

### Effect on skeleton coverage

| | with skeleton | missing |
|---|---|---|
| before (2026-07-03 sheet, no fallback) | 330 / 376 | **46** |
| after sheet update alone | 347 / 376 | 29 |
| after sheet update + `Proofread SegID` fallback | **355 / 376** | **21** |

Two log entries are resolved outright by the sheet update:

- **GCL0-38** was retyped `EW5to` → `OFF transient small RF`. Your
  *"Differs from other EW5to?"* note was right — it isn't an EW5to. Against its
  new peers its cable length / extent are 0.98× / 1.05× the median.
- Other relabels touching ROI-matched cells: **GCL3-37** `A17 large` → `dA17 large`;
  **GCL0-21** and **GCL1-51** Cell Class `RGC` → `AC`.

Four "No morphology" complaints are fixed by the sheet update (GCL0-41, GCL4-67,
GCL3-56, GCL2-47) and three more by the fallback (GCL0-23, GCL2-21, GCL2-45).

---

## A. Truly missing skeletons — 21 cells

No SWC on disk under either `Latest SegID` or `Proofread SegID`. These need a
skeletonization pass. Searched `data/`, `eyewire2-morphological-clustering`,
`eyewire2-datajoint`, `eyewire2-figures`, `eyewire2-explorer`, `cell_mosaics`,
`retinal_neuron_data_collection` — only GCL2-8 turned up elsewhere.

| # | field | ROI | Class | Cell Type | Latest NucID | Latest SegID | Proofread SegID | Status | elsewhere on disk |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GCL0 | 2 | RGC | ON-OFF DS - ventral | `720575940534317101` | `720575940574776056` | `720575940568652126` | Complete | |
| 2 | GCL0 | 37 | AC | Midnight | `720575940564979397` | `720575940582620680` | `720575940549685474` | Complete (cut off) | |
| 3 | GCL0 | 52 | AC | Separate dendrites and axons | `720575940571413949` | `720575940568348405` | `720575940568177741` | Complete (cut off) | |
| 4 | GCL0 | 82 | AC | Wide-field ON non-spiny dense | `720575940568129302` | `720575940560476040` | `720575940561105113` | Complete | |
| 5 | GCL1 | 2 | AC | Medium-field low in IPL transient | `720575940553185617` | `720575940560249322` | `720575940562488181` | Complete | |
| 6 | GCL1 | 20 | AC | ON SAC | `720575940570979195` | `720575940565329821` | `720575940565252969` | Complete | |
| 7 | GCL1 | 56 | AC | Separate dendrites and axons | `720575940549202881` | `720575940569357168` | `720575940557233636` | Complete (cut off) | |
| 8 | GCL1 | 60 | AC | nNOS-1 | `720575940549247681` | `720575940564482075` | `720575940568046196` | Complete (cut off) | |
| 9 | GCL2 | 2 | AC | nNOS-1 | `720575940561931950` | `720575940558175104` | `720575940567610509` | Complete (cut off) | |
| 10 | GCL2 | 8 | RGC | ON alpha | `720575940559161547` | `720575940558177239` | `720575940580068789` | Complete | **`eyewire2-figures/data/swc` (Latest SegID)** |
| 11 | GCL2 | 10 | RGC | EW915 | `720575940551984936` | `720575940565541085` | `720575940564229277` | Complete | |
| 12 | GCL2 | 14 | — | *(untyped)* | `720575940554690648` | `720575940574467389` | `720575940564462948` | Can't Complete | |
| 13 | GCL2 | 18 | AC | Wide-field ON non-spiny dense | `720575940545753692` | `720575940555937262` | `720575940580299373` | *(blank)* | |
| 14 | GCL2 | 24 | RGC | Sustained suppressed-by-contrast no surround EW27 | `720575940554121124` | `720575940571732140` | `720575940575091193` | Complete | |
| 15 | GCL2 | 61 | RGC | OFF transient small RF | `720575940549841109` | `720575940577835935` | `720575940566730337` | Complete | |
| 16 | GCL3 | 25 | AC | Wide-field ON non-spiny sparse | `720575940551878109` | `720575940558489815` | `720575940569068651` | *(blank)* | |
| 17 | GCL3 | 35 | AC | nNOS-1 | `720575940546415816` | `720575940564123035` | `720575940559008563` | Complete | |
| 18 | GCL3 | 37 | AC | dA17 large | `720575940569767457` | `720575940567417708` | `720575940579988825` | Complete | |
| 19 | GCL3 | 41 | AC | nNOS-1 | `720575940554560563` | `720575940582330120` | `720575940545438195` | Complete (cut off) | |
| 20 | GCL3 | 57 | RGC | ON transient EW6t | `720575940561701193` | `720575940561102769` | `720575940557844538` | Complete | |
| 21 | GCL4 | 77 | AC | m8:X | `720575940569367221` | `720575940587288061` | `720575940587288061` | Complete | |

**Composition:** 14 AC, 6 RGC, 1 untyped. The gap is skewed toward amacrine
cells — 19/160 AC (11.9%) vs 9/206 RGC (4.4%) missing before the fallback — and
toward `Complete (cut off)` cells (8 of 29).

**Note on #21 (GCL4-77):** `Latest SegID` and `Proofread SegID` are identical
(`720575940587288061`), so the fallback can never help this one.

### Quick win

GCL2-8's SWC exists at
`eyewire2-figures/data/swc/720575940558177239.swc` (keyed by its current
`Latest SegID`). That directory holds 447 SWCs vs 368 in
`eyewire2-functional-analysis/data/swc`, with 79 not present here — worth
diffing the two stores before commissioning new skeletonization, though it fills
only this one gap in the ROI-matched set.

---

## B. Resolved via `Proofread SegID` fallback — 8 cells, need spot-checking

These now show a morphology, but the SWC is keyed to an **older proofreading
state** of the cell (its `Latest SegID` has since advanced). Worth eyeballing
before trusting them in analysis; `swc_segid_source == 'Proofread SegID'`
selects exactly this set.

| field | ROI | Class | Cell Type | Latest SegID (no SWC) | Proofread SegID (SWC used) | Status |
|---|---|---|---|---|---|---|
| GCL0 | 16 | AC | nNOS-1 | `720575940555441116` | `720575940561733293` | Complete |
| GCL0 | 23 | RGC | OFF transient alpha | `720575940584309627` | `720575940558082827` | Complete |
| GCL1 | 10 | AC | ON SAC | `720575940553935194` | `720575940559127957` | Complete |
| GCL2 | 21 | RGC | M6 | `720575940570933529` | `720575940545166421` | Complete |
| GCL2 | 45 | RGC | OFF sustained EW3o | `720575940564663865` | `720575940567190859` | Complete |
| GCL3 | 50 | AC | Wide-field ON non-spiny dense | `720575940565117029` | `720575940553331287` | Complete (cut off) |
| GCL3 | 65 | AC | nNOS-1 | `720575940552687231` | `720575940555723642` | Complete (cut off) |
| GCL4 | 1 | AC | ON SAC | `720575940569675536` | `720575940559596123` | Complete |

All 8 parse cleanly (7.8k–20.1k nodes, plausible soma positions).

---

## C. ROI split — GCL0-10 / GCL0-12

**Confirmed real, and not fixable from the master list.**

Both ROIs carry the *identical* `Latest NucID` **`720575940562919113`**, the
identical `Latest SegID` **`720575940565590863`**, and identical
`Nuc Coords` / `Soma Coords` — i.e. two 2p ROIs, 9.23 µm apart, assigned to one
EM cell (`Bursty suppressed by contrast`). This lives in the **EM-2p mapping
sheet** (`Eyewire II Proofread Cells Main List - EM-2p-mapping 2026-07-08e v2-final.csv`),
not the master list.

**Decision needed:** merge the two ROIs' traces, or drop one. Until then this
cell is double-counted in every per-type average.

For context, 6 ROI pairs share a `Latest SegID`, but the other 5 are legitimate —
GCL0 and GCL1 physically overlap, so the same cell was recorded in both fields:

| Latest NucID | Latest SegID | ROIs | Cell Type | 2p sep | verdict |
|---|---|---|---|---|---|
| `720575940562919113` | `720575940565590863` | GCL0-10, GCL0-12 | Bursty suppressed by contrast | 9.23 µm | **ROI split — same field, needs fix** |
| `720575940552618508` | `720575940551713151` | GCL0-21, GCL1-51 | OFF transient medium RF | 1.31 µm | legitimate (GCL0/GCL1 overlap) |
| `720575940552583180` | `720575940569897106` | GCL0-9, GCL1-46 | ON transient medium RF | 1.88 µm | legitimate |
| `720575940574221509` | `720575940563567973` | GCL0-31, GCL1-61 | PixON | 2.33 µm | legitimate |
| `720575940555393530` | `720575940549771616` | GCL0-39, GCL1-72 | ON-OFF DS - direction unknown | 2.48 µm | legitimate |
| `720575940545403331` | `720575940568909174` | GCL0-50, GCL1-78 | F-mini-OFF | 4.48 µm | legitimate |

---

## D. "Same location" pairs — 3 pairs, mosaic violations

**Not a registration or data-plumbing bug.** Each pair is two genuinely distinct
cells (different NucIDs), and their EM and 2p separations agree to within 3.1 µm,
so the 2p↔EM registration is sound.

The real problem: they are far too close together *for cells sharing a type*.
RGCs of one type tile the retina, so a same-type pair at the 4th percentile of
its own type's nearest-neighbour distribution means one of the two labels is
probably wrong.

| Cell Type | pair | NucIDs | Latest SegIDs | EM sep | type median NN | pctile | machine conf | human typer |
|---|---|---|---|---|---|---|---|---|
| OFF vertical OS - asymmetric | GCL2-27 / GCL2-35 | `720575940555560280`<br>`720575940559738416` | `720575940576478333`<br>`720575940577707986` | 9.9 µm | 31.9 µm | 4th | 0.89 / 0.89 | **none** |
| ON DS sustained - direction unknown | GCL4-3 / GCL4-19 | `720575940556558833`<br>`720575940566853181` | `720575940572858144`<br>`720575940557118818` | 18.5 µm | 39.9 µm | 15th | 0.95 / 0.86 | **none** |
| OFF sustained EW3o | GCL3-26 / GCL3-42 | `720575940560514319`<br>`720575940546608072` | `720575940554826538`<br>`720575940556779011` | 21.3 µm | 44.8 µm | 13th | 0.51 / 0.73 | Julia / Greg |

(Mosaic NN computed over all cells of each type in the master list with
`Nuc Coords`, converting EM voxels at 16×16×40 nm — the same scale
`scripts/preprocessing/em-2p-mapping.py` uses. n = 201 / 107 / 63 cells.)

**Action:** human type review on all 6. Note that 4 of the 6 have *no* human
typer recorded and were auto-typed identically at high confidence — the machine
labels agreeing is not independent evidence. The `EW3o` pair is the weakest
machine call (0.51) but the only one a human has looked at.

**Why they look co-located in the explorer:** `space_mapping.align_and_place_skel`
places each skeleton's soma at its *2p ROI* position, and the morphology panel
auto-zooms to fit skeletons spanning hundreds of µm — so a 10–20 µm offset is
visually invisible. The display is behaving correctly.

---

## E. "Different morphology?" / "Morphology complete?" — quantified

Cable length (total skeleton) and dendritic xy extent (axon excluded, matching
`plot_morph.plot_morph`'s `ntype != 2` convention), as a ratio to the median of
same-type ROI-matched peers. Sorted by deviation.

| field | ROI | Cell Type | Latest SegID | Status | cable | extent | n peers | your comment | reading |
|---|---|---|---|---|---|---|---|---|---|
| GCL3 | 51 | ON small OFF large | `720575940565437186` | Complete | 1.74× | 1.90× | 2 | Different morphology? | **oversized — merge error or wrong type** |
| GCL1 | 5 | F-mini-OFF | `720575940572873585` | Complete (cut off) | 0.34× | 0.71× | 11 | Morphology looks incomplete, response different | **truncated** |
| GCL0 | 67 | OFF vertical OS - asymmetric | `720575940553717784` | Complete | 1.54× | 1.09× | 10 | Different morphology?! | **oversized — merge error or wrong type** |
| GCL0 | 19 | ON DS sustained - direction unknown | `720575940550200928` | Complete | 1.50× | 1.10× | 9 | Different morphology? On/Off? | **oversized — merge error or wrong type** |
| GCL1 | 14 | ON-OFF DS - direction unknown | `720575940562681623` | Complete | 0.54× | 0.87× | 9 | Morphology complete? | **truncated** |
| GCL4 | 39 | HD1 | `720575940557521752` | Complete (cut off) | 0.57× | 0.66× | 5 | Morphology complete? | **truncated** |
| GCL4 | 10 | ON-OFF DS - direction unknown | `720575940570589739` | Complete | 0.57× | 0.69× | 9 | Different morphology? Complete? | **truncated** |
| GCL4 | 13 | F-mini-ON | `720575940570271455` | Complete (cut off) | 0.60× | 1.08× | 8 | Morphology complete? | **truncated** |
| GCL1 | 29 | ON vertical OS small RF | `720575940550156327` | Complete | 0.66× | 1.04× | 1 | Different morphology? | **truncated** |
| GCL1 | 3 | ON-OFF DS - ventral | `720575940565010676` | Complete (cut off) | 0.66× | 0.79× | 2 | Different morphology? | **truncated** |
| GCL4 | 43 | ON DS sustained - direction unknown | `720575940552823621` | Complete | 0.79× | 0.79× | 9 | Different morphology? | within normal range |
| GCL3 | 11 | OFF sustained EW1no | `720575940555117956` | Complete | 0.83× | 0.85× | 3 | Morphology different? | within normal range |
| GCL3 | 26 | OFF sustained EW3o | `720575940554826538` | Complete | 1.08× | 1.16× | 3 | Same location as GCL3-42, different morphology? | within normal range |
| GCL3 | 58 | OFF vertical OS - asymmetric | `720575940565664932` | Complete | 0.98× | 1.06× | 10 | Different morphology | within normal range |
| GCL0 | 38 | OFF transient small RF | `720575940550007010` | Complete | 0.98× | 1.05× | 4 | Differs from other EW5to? | resolved by retype (was `EW5to`) |

**Priorities:**

- **GCL1-5 F-mini-OFF** (`720575940572873585`) — worst case at 0.34× cable. Status
  already admits `Complete (cut off)`. Your "response different" note is likely a
  *consequence* of the truncation, not independent evidence of a wrong type.
- **GCL3-51 ON small OFF large** (`720575940565437186`) — 1.74× / 1.90×, biggest
  overshoot; check for a segmentation merge. Only 2 peers, so the baseline is weak.
- **GCL1-14** (`720575940562681623`) and **GCL4-10** (`720575940570589739`) are
  marked `Complete` but sit at 0.54× / 0.57× cable. Their `Status` looks wrong.
- The 5 rows marked *within normal range* are plausibly fine on size alone; if they
  still look wrong, the issue is branching pattern or stratification, which these
  two scalars don't capture. Compare their IPL depth profiles next.

**Caveat:** peer counts are small (1–11), so ratios against types with ≤3 peers
(GCL3-51, GCL1-29, GCL1-3, GCL3-11, GCL3-26) are indicative, not conclusive.

---

## F. Log bookkeeping

Two entries in `celltype_explorer_log.xlsx` are duplicates or mis-entries:

- **GCL3-51** is logged twice, once as *"No morphology"* — but it has both a
  skeleton (`720575940565437186`) and a cached IPL profile, under both the old and
  new sheets. That comment appears to belong to a different cell.
- **GCL0-19** is logged three times with near-identical comments
  (*"Different morphology?"* ×2, *"Different morphology? On/Off?"*).

Net: 38 rows → 35 unique cells.

---

## Remaining TODO

| # | item | scope |
|---|---|---|
| 1 | Commission skeletonization for the 21 cells in **§A** (14 AC / 6 RGC / 1 untyped) | external |
| 2 | Copy GCL2-8's SWC from `eyewire2-figures/data/swc/720575940558177239.swc`; diff the two SWC stores (79 files present there but not here) | local |
| 3 | Spot-check the 8 older-state fallback skeletons in **§B** | review |
| 4 | Fix GCL0-10 / GCL0-12 in the **EM-2p mapping sheet** — merge or drop (**§C**) | data fix |
| 5 | Human type review on the 6 cells in the 3 mosaic-violating pairs (**§D**) | review |
| 6 | Investigate GCL1-5 and GCL3-51 first among the morphology outliers (**§E**) | review |
| 7 | Correct `Status` for GCL1-14 and GCL4-10 (marked `Complete`, clearly truncated) | data fix |
| 8 | Clean up the duplicate log rows (**§F**) | housekeeping |

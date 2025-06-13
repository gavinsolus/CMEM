# CMEM
Curved Membrane (Cmem) Builder is a versatile tool specifically designed to construct coarse-grained (CG) MD initial configurations for lipid membranes with defined curvatures and associated protein-membrane complexes.
##  Environment

Make sure your Python environment includes:

```bash
python 3.9
scipy
```

You can install the required packages using:

```bash
pip install scipy
```

------

## Usage

To see available options and usage instructions, run:

```bash
python cmem.py -h
```

------

## Input PDB 

Please ensure that the PDB file contains information related to membrane curvature.

You can obtain input PDB files from the **PPM server** in either of the following ways:

- **Download a PDB file with precomputed membrane curvature** from the PPM database.
- **Generate a PDB file with membrane curvature using PPM 3.0** and download it.

> PPM (Positioning of Proteins in Membranes): https://opm.phar.umich.edu/ppm_server



## Protein-Membrane System Construction

### 1. Extract Bezier Control Points

Use the membrane-positioned PDB file (from PPM server) to extract control points:

```bash
python refpoints.py 1xq8_ppm.pdb
```


### 2. Coarse-Grain the Protein Structure (Example: Martini2)

Clean the original PDB and generate coarse-grained coordinates.
 This example uses **Martini2** with `martinize2`, but other coarse-graining workflows (e.g., **Martini3**) can be adapted similarly:

```bash
grep -E '^ATOM|^TER' 1xq8_ppm > clean.pdb
martinize2 -f clean.pdb -x cg.pdb -o output.top -ff martini22 -dssp mkdssp -p backbone -ignh
```

> You may replace `-ff martini22` with other supported Martini force fields, such as `martini3001` for Martini 3.0.

### 3. Align CG Model to Membrane Frame

Align the coarse-grained protein (`cg.pdb`) to the membrane curvature frame (`1xq8_ppm.pdb`):

```bash
python fit_to_ref.py 1xq8_ppm.pdb cg.pdb cg_aligned.pdb
```


### 4. Build the Curved Membrane System

Use the aligned protein to build the protein-membrane system:

```bash
python cmem.py -input_pdb cg_aligned.pdb \
                -x 300 -z_u 100 -z_l 100 \
                -points "0,0;44,0;44,37.67;88,37.67" \
                -u "POPC:1,DOPC:2" -l "POPC:1" \
                -ff martini2 -output proteinMembrane \
                > CurvedMembrane.pdb
```

------

## Membrane-Only System

If you want to build a **membrane-only** system without a protein, simply omit the `-input_pdb` option and specify the `-output`:

```bash
python cmem.py -x 300 -z_u 100 -z_l 100 \
                -points "0,0;44,0;44,37.67;88,37.67" \
                -u "POPC:1" -l "POPC:1" \
                -ff martini2 -output membraneOnly \
                > CurvedMembrane.pdb
```

------

##  Only `cg.pdb` Available

If you only have a **coarse-grained protein structure (`cg.pdb`)**, you can still construct a curved membrane system by following the steps below.


### 1. Obtain Bezier Control Points

If you have a PDB file positioned in the membrane (e.g., from PPM), use it to extract Bezier control points:

```bash
python refpoints.py xxxx_ppm.pdb
```

Alternatively, you can **manually define control points** using the web server’s interactive Bezier curve editor.


### 2. Center the Protein Geometry

Align the geometric center (centroid) of the coarse-grained protein to the origin:

```bash
python centerize.py cg.pdb cg_centered.pdb
```


### 3. Build the Curved Membrane System

Use the centered structure to construct the protein-membrane system.
 You can tune the `-trans` value to shift the protein along the **z-axis**, helping it better match the membrane position estimated by PPM:

```bash
python cmem.py -input_pdb cg_centered.pdb \
                -x 300 -z_u 100 -z_l 100 \
                -points "0,0;44,0;44,37.67;88,37.67" \
                -u "POPC:1" -l "POPC:1" \
                -ff martini2 -output proteinMembrane \
                -trans 10 > CurvedMembrane.pdb
```

> Adjust `-trans` (e.g., `-trans 10`) to fine-tune the vertical alignment between the protein and membrane surface.

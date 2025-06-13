import numpy as np
import argparse
from typing import Dict, Any


def parse_control_points(entry: str) -> np.ndarray:
    points = [p.replace(" ", ",").split(",") for p in entry.split(";")]
    return np.array([[float(x) for x in point] for point in points])

def parse_lipid_entry(entry: str) -> Dict[str, float]:
    lipid_dict = {}
    for item in entry.split(","):
        parts = item.split(":")
        key = parts[0].upper()
        value = float(parts[1]) if len(parts) > 1 else 1.0
        lipid_dict[key] = value
    return lipid_dict

def supplement_control_points(control_points: np.ndarray) -> np.ndarray:
    if control_points.shape[0] == 4:
        x, y = control_points[:, 0], control_points[:, 1]
        x_new = np.linspace(x[3], 10 * x[3], 4)[1:].astype(int)
        y_new = np.full_like(x_new, y[3])
        return np.vstack((control_points, np.column_stack((x_new, y_new))))
    return control_points

def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run combined script with specified options",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-input_pdb", type=str, default=None, help="Input PDB file (required if output type is proteinMembrane)")

    parser.add_argument("-x", type=float, default=300.0, help="Box dimension X")
    parser.add_argument("-z_u", type=int, default=100, help="Upper Z dimension")
    parser.add_argument("-z_l", type=int, default=100, help="Lower Z dimension")
    parser.add_argument("-points", type=parse_control_points, default=None, help="Control points in format 'x1,y1;x2,y2;...'")
    parser.add_argument("-u", type=parse_lipid_entry, default={}, help="Lipid for upper layer")
    parser.add_argument("-l", type=parse_lipid_entry, default={}, help="Lipid for lower layer")
    parser.add_argument("-ff", type=str, default=None, help="Force field (martini2 or martini3)")
    parser.add_argument("-output", type=str, default=None, help="Output type (membraneOnly or proteinMembrane)")
    parser.add_argument("-bin_area", type=float, default=0.6, help="Bin area value")
    parser.add_argument("-bin_u", type=float, default=0.6, help="Bin area value upper")
    parser.add_argument("-bin_l", type=float, default=0.6, help="Bin area value lower")
    parser.add_argument("-trans", type=float, default=0.0, help="Translation value")
    parser.add_argument("-arc", type=float, default=0.2, help="Arc value")
    parser.add_argument("-nv_l", type=float, default=30.0, help="Length NV value")
    parser.add_argument("-nv_u", type=float, default=30.0, help="Length NV value")

    args = parser.parse_args()

    return {
        "input_pdb": args.input_pdb,
        "box_X": args.x,
        "box_Y": args.x,  
        "Z_u": args.z_u,
        "Z_l": args.z_l,
        "control_points": args.points,
        "lipid_dict_upper": args.u,
        "lipid_dict_lower": args.l,
        "ff": args.ff,
        "ot": args.output,
        "bin_area_value": args.bin_area,
        "bin_area_upper": args.bin_u,
        "bin_area_lower": args.bin_l,
        "trans": args.trans,
        "arc": args.arc,
        "length_nv_u": args.nv_u,
        "length_nv_l": args.nv_l,       
    }

def main() -> Dict[str, Any]:
    results = parse_arguments()
    if results["control_points"] is not None:
        results["control_points"] = supplement_control_points(results["control_points"])
    
    if results["ot"] == "proteinMembrane" and not results["input_pdb"]:
        raise ValueError("Error: input_pdb must be specified when output type is 'proteinMembrane'.")

    return results


if __name__ == "__main__":
    results = main()
    input_pdb = results["input_pdb"]
    box_X = results["box_X"]
    box_Y = results["box_Y"]
    Z_u = results["Z_u"]
    Z_l = results["Z_l"]
    control_points = results["control_points"]
    lipid_dict_upper = results["lipid_dict_upper"]
    lipid_dict_lower = results["lipid_dict_lower"]
    ff = results["ff"]
    ot = results["ot"]
    trans = results["trans"]
    arc = results["arc"]
    bin_area_value = results["bin_area_value"]
    bin_area_upper =  results["bin_area_upper"]
    bin_area_lower = results["bin_area_lower"]
    length_nv_upper = results["length_nv_u"]
    length_nv_lower = results["length_nv_l"]
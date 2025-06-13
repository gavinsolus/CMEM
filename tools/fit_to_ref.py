import numpy as np
import sys
from scipy.spatial import KDTree

def extract_dum_coordinates_by_atom_type(dum_pdb):
    n_coordinates = []
    o_coordinates = []
    
    with open(dum_pdb, 'r') as file:
        for line in file:
            if line.startswith("HETATM") and " DUM " in line:
                atom_type = line[12:14].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                if atom_type == "N":
                    n_coordinates.append((x, y, z))
                elif atom_type == "O":
                    o_coordinates.append((x, y, z))
    
    return n_coordinates, o_coordinates

def extract_zero_line(coords, axis='x'):
    """提取 x=0 或 y=0 的坐标"""
    if axis == 'x':
        zero_line_coords = [(coord[1], coord[2]) for coord in coords if coord[0] == 0]
    elif axis == 'y':
        zero_line_coords = [(coord[0], coord[2]) for coord in coords if coord[1] == 0]
    return zero_line_coords

def fit_circle_from_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    A = np.array([
        [x1 - x2, y1 - y2],
        [x1 - x3, y1 - y3]
    ])
    
    B = np.array([
        [(x1**2 - x2**2) + (y1**2 - y2**2)],
        [(x1**2 - x3**2) + (y1**2 - y3**2)]
    ]) / 2

    center = np.linalg.solve(A, B)
    x_c, y_c = center.flatten()

    r = np.sqrt((x1 - x_c)**2 + (y1 - y_c)**2)
    return x_c, y_c, r

def fit_circle_from_extremes(coords):
    coords_sorted = sorted(coords, key=lambda coord: coord[0])
    p1 = coords_sorted[0]
    p2 = coords_sorted[len(coords_sorted) // 2]
    p3 = coords_sorted[-1]
    return fit_circle_from_points(p1, p2, p3)

def process_atom_data(n_coords, o_coords):
    x_zero_coords_n = extract_zero_line(n_coords, axis='x')
    x_zero_coords_o = extract_zero_line(o_coords, axis='x')

    circle_params_n = fit_circle_from_extremes(x_zero_coords_n)
    circle_params_o = fit_circle_from_extremes(x_zero_coords_o)

    R1 = circle_params_n[2]
    R2 = circle_params_o[2]
    translation_distance = (R1 + R2) / 2
    return translation_distance

def translate_z_axis(input_pdb_file, output_pdb_file, z_offset):
    with open(input_pdb_file, 'r') as f:
        lines = f.readlines()

    with open(output_pdb_file, 'w') as f:
        for line in lines:
            if line.startswith('ATOM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54]) + z_offset
                new_line = line[:30] + f'{x:8.3f}' + f'{y:8.3f}' + f'{z:8.3f}' + line[54:]
                f.write(new_line)
            else:
                f.write(line)

def insert_protein_in_membrane(protein_pdb, dum_pdb, output_pdb):
    n_coords, o_coords = extract_dum_coordinates_by_atom_type(dum_pdb)
    translation_distance = process_atom_data(n_coords, o_coords)
    translation_distance = round(translation_distance, 3)

    translate_z_axis(protein_pdb, output_pdb, translation_distance)


if __name__ == "__main__":
    dum_pdb = sys.argv[1] 
    input_pdb_file = sys.argv[2]
    output_pdb_file = sys.argv[3]

    insert_protein_in_membrane(input_pdb_file, dum_pdb, output_pdb_file)

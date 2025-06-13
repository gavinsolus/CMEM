import numpy as np
import sys
import json


def extract_dum_coordinates_by_atom_type(pdb_file):
    n_coordinates = [] 
    o_coordinates = []  
    positive_flag = None  

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("HETATM") and " DUM " in line:
                atom_type = line[12:14].strip()  

                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())


                if positive_flag is None:
                    positive_flag = 1 if z >= 0 else -1
                
                if atom_type == "N":
                    n_coordinates.append((x, y, z))
                elif atom_type == "O":
                    o_coordinates.append((x, y, z))
    
    return n_coordinates, o_coordinates, positive_flag

def extract_zero_line(coords, axis='x'):
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

def calculate_translated_y(radius, x, flag, translation):
    try:
        y = np.sqrt(radius**2 - x**2) if radius**2 - x**2 >= 0 else 0
    except:
        y = 0
    return flag * y + translation

def process_atom_data(n_coords, o_coords, positive_flag):
    x_zero_coords_n = extract_zero_line(n_coords, axis='x')
    x_zero_coords_o = extract_zero_line(o_coords, axis='x')

    circle_params_n = fit_circle_from_extremes(x_zero_coords_n)
    circle_params_o = fit_circle_from_extremes(x_zero_coords_o)

    R1 = circle_params_n[2]
    R2 = circle_params_o[2]
    translation_distance = (R1 + R2) / 2

    min_x_n = min([p[0] for p in x_zero_coords_n])
    max_x_n = max([p[0] for p in x_zero_coords_n])
    min_x_o = min([p[0] for p in x_zero_coords_o])
    max_x_o = max([p[0] for p in x_zero_coords_o])

    y_n = next((p[1] for p in x_zero_coords_n if p[0] == max_x_n), None)
    y_o = next((p[1] for p in x_zero_coords_o if p[0] == max_x_o), None)
    y_comparison_result = 1 if y_n < y_o else -1

    points = [{"x": 0, "y": 0}, {}, {}, {}]

    if positive_flag == -1:
        points[1]["x"] = max_x_o
        points[1]["y"] = 0
        points[2]["x"] = max_x_o

        yN = calculate_translated_y(R1, max_x_n, positive_flag, translation_distance)
        yO = calculate_translated_y(R2, max_x_o, positive_flag, translation_distance)
        points[2]["y"] = max(yN, yO)
    
    elif positive_flag == 1:
        points[1]["x"] = max_x_n
        points[1]["y"] = 0
        points[2]["x"] = max_x_n

        yN = calculate_translated_y(R1, max_x_n, positive_flag, translation_distance)
        yO = calculate_translated_y(R2, max_x_o, positive_flag, translation_distance)
        points[2]["y"] = min(yN, yO)
    
    else:
        raise ValueError("Invalid positive_flag: must be 1 or -1")

    points[3]["x"] = points[2]["x"] * 2
    points[3]["y"] = points[2]["y"]

    result = {
        "min_x_N": min_x_n,
        "max_x_N": max_x_n,
        "min_x_O": min_x_o,
        "max_x_O": max_x_o,
        "radius_N": R1,
        "radius_O": R2,
        "translation": translation_distance,
        "positive_flag": positive_flag,
        "y_comparison_result": y_comparison_result,
        "points": points 
    }

    return json.dumps(result)

pdb_file_path = sys.argv[1]
n_coords, o_coords, positive_flag = extract_dum_coordinates_by_atom_type(pdb_file_path)

result = process_atom_data(n_coords, o_coords, positive_flag)
result_dict = json.loads(result)
rounded_points = [
    {"x": round(p["x"], 2), "y": round(p["y"], 2)}
    for p in result_dict["points"]
]
print(rounded_points)


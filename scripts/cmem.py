import numpy as np
import math
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
import os
from cm_parser import main as parser_main

def pdb_readline(lipid_type, ff):
    filepath1 = "../lipid_lib/" + "M3_" + lipid_type + ".pdb"  
    filepath2 = "../lipid_lib/" + "M2_" + lipid_type + ".pdb"  

    if ff == "martini3" or ff == "elnedyn3" or  ff == "go3" or ff == "olive":
        
        if os.path.exists(filepath1):
            filepath = filepath1
        else:
            raise FileNotFoundError(f"{filepath1} does not exist. Martini3/Elnedyn3 requires its own file.")
    elif ff == "martini2" or ff == "elnedyn2": 
        
        if os.path.exists(filepath2):
            filepath = filepath2
        else:
            raise FileNotFoundError(f"{filepath2} does not exist. Martini2/Elnedyn2 requires its own file.")
    else:
        raise ValueError("Unsupported force field. Please choose 'martini2', 'martini3', 'elnedyn2', or 'elnedyn3'.")

    
    with open(filepath, 'r') as file:
        file_lines = file.readlines()

    return file_lines

def lipid_ratios(lipid_dict):
    
    max_key = max(lipid_dict, key=lipid_dict.get)
    max_value = lipid_dict[max_key]

    
    ratios = {}
    total = sum(lipid_dict.values())
    for key, value in lipid_dict.items():
        ratios[key] = value / total  
    sorted_ratios = dict(sorted(ratios.items(), key=lambda item: item[1], reverse=True))
    return max_key, sorted_ratios


def pdb2xyz(pdbline):
    x = float(pdbline[30:38].strip())
    y = float(pdbline[38:46].strip())
    z = float(pdbline[46:54].strip())
    return x,y,z


def clean_lipid(pdb_lines, length_nv):
    z_position_content = []
    new_pdb_lines = []
    atom_num = 0
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x, y, z = pdb2xyz(line)
            atom_name = line[12:16].strip()
            res_name  = line[16:21].strip()
            if atom_name in ["GL2", "R3", "AM2"]:
                virtual_site_x = x
                virtual_site_y = y
                virtual_site_z = z
            z_position_content.append(z)
            new_pdb_lines.append(line)
    z_min = min(z_position_content)

    virtual_site_z = z_min - 1.5   
    virtual_site_line  = "ATOM      0   VS  DUM    1   " + "{:8.3f}{:8.3f}{:8.3f}".format(virtual_site_x, virtual_site_y, virtual_site_z)
    virtual1_site_line = "ATOM      0   VS  DUU    1   " + "{:8.3f}{:8.3f}{:8.3f}".format(virtual_site_x, virtual_site_y, virtual_site_z + length_nv) 
    new_pdb_lines.insert(0, virtual_site_line)
    new_pdb_lines.insert(1, virtual1_site_line)
    return new_pdb_lines



def rotate_points_Zaxis_variable(points, normals, arc_length=0.1):
    
    
    radii = np.linalg.norm(points[:, :2], axis=1)  

    
    mask = radii > 0  
    points = points[mask]
    normals = normals[mask]
    radii = radii[mask]

    
    all_rotated_points = []
    all_rotated_normals = []

    
    for i, (point, normal, R) in enumerate(zip(points, normals, radii)):
        
        circumference = 0.25 * np.pi * R
        num_steps = max(1, int(np.ceil(circumference / arc_length)))

        delta_theta = 0.25 * np.pi / num_steps
        angles = np.linspace(0, 0.25 * np.pi, num_steps + 1, endpoint=True)

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        rotation_matrices = np.array([
            [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
            for cos, sin in zip(cos_angles, sin_angles)
        ])

        rotated_points = np.einsum('ijk,k->ij', rotation_matrices, point)
        all_rotated_points.append(rotated_points)

        rotated_normals = np.einsum('ijk,k->ij', rotation_matrices, normal)
        all_rotated_normals.append(rotated_normals)
        
        

    
    rotated_points = np.vstack(all_rotated_points)  
    rotated_normals = np.vstack(all_rotated_normals)  

    return rotated_points, rotated_normals


def rotate_vector(v_point, v1, lipid):
    
    l_x1, l_y1, l_z1 = pdb2xyz(lipid[0])
    l_x2, l_y2, l_z2 = pdb2xyz(lipid[1])
    l_a1 = np.array([l_x1, l_y1, l_z1])
    l_a2 = np.array([l_x2, l_y2, l_z2])
    
    v2 = l_a2 - l_a1
    
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    
    v = np.cross(unit_v2, unit_v1)
    s = np.linalg.norm(v)
    
    c = np.dot(unit_v2, unit_v1)
    
    rotated_lipid = []

    if s == 0 and c > 0:  
        delta_xyz = v_point - l_a1
        for line in lipid:
            x, y, z = pdb2xyz(line)
            coords = np.array([x, y, z])
            new_coords = coords + delta_xyz
            new_line = f"{line[:30]}{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}{line[54:]}"
            rotated_lipid.append(new_line)
    elif s== 0 and c < 0:
        delta_xyz = v_point - l_a1 * -1
        for line in lipid:
            x, y, z = pdb2xyz(line)
            coords = np.array([x, y, z])
            new_coords = coords * -1 + delta_xyz
            new_line = f"{line[:30]}{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}{line[54:]}"
            rotated_lipid.append(new_line)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        delta_xyz = np.array([0,0,0])
        for line in lipid:
            res_name = line[16:21].strip()
            x, y, z = pdb2xyz(line)
            r_coords = R.dot(np.array([x, y, z]))
            if res_name == "DUM":
                delta_xyz = v_point - r_coords
            
            new_coords = r_coords + delta_xyz
            new_line = f"{line[:30]}{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}{line[54:]}"
            rotated_lipid.append(new_line)
    return rotated_lipid


def random_rotation_matrix(axis, theta):
    
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])

def random_rotate_lipid(pdb_lines):
    
    Ax,Ay,Az = pdb2xyz(pdb_lines[0])
    Bx,By,Bz = pdb2xyz(pdb_lines[1])
    point_A = [Ax,Ay,Az]
    point_B = [Bx,By,Bz]
    theta = np.random.uniform(0, 2 * np.pi)
    rotated_lines = []
    
    axis = np.array(point_B) - np.array(point_A)
    
    rot_mat = random_rotation_matrix(axis, theta)
    
    for line in pdb_lines:
            x, y, z = pdb2xyz(line)
            
            original_point = np.array([x, y, z])
            
            trans_point = original_point - np.array(point_A)
            
            rotated_point = np.dot(rot_mat, trans_point)
            
            final_point = rotated_point + np.array(point_A)
            
            
            new_line = (
                line[:30]
                + f"{final_point[0]:8.3f}"
                + f"{final_point[1]:8.3f}"
                + f"{final_point[2]:8.3f}"
                + line[54:]
            )
            rotated_lines.append(new_line)
    return rotated_lines



def parse_pdb_lines(pdb_lines):
    atoms = []
    if isinstance(pdb_lines, list):  
        if len(pdb_lines) == 0:  
            return "This is an empty lipid."
        if all(not isinstance(i, list) for i in pdb_lines):  
            for line in pdb_lines:
                if line.startswith("ATOM") and line[16:21].strip() not in ["DUM", "DUU"]:
                    x, y, z = pdb2xyz(line)
                    atoms.append([x, y, z])
            return atoms
        elif all(isinstance(i, list) for i in pdb_lines):  
            for lines in pdb_lines:
                for line in lines:
                    if line.startswith("ATOM") and line[16:21].strip() not in ["DUM", "DUU"]:
                        x, y, z = pdb2xyz(line)
                        atoms.append([x, y, z])
            return atoms

def parse_pdb_lines_DUU(pdb_lines):
    atoms = []
    if isinstance(pdb_lines, list):  
        if len(pdb_lines) == 0:
            return "This is an empty lipid."
        if all(not isinstance(i, list) for i in pdb_lines):
            for line in pdb_lines:
                if line.startswith("ATOM") and line[16:21].strip() in ["DUU"]:
                    x, y, z = pdb2xyz(line)
                    atoms.append([x, y, z])
            return atoms
        elif all(isinstance(i, list) for i in pdb_lines):  
            for lines in pdb_lines:
                for line in lines:
                    if line.startswith("ATOM") and line[16:21].strip() in ["DUU"]:
                        x, y, z = pdb2xyz(line)
                        atoms.append([x, y, z])
            return atoms

def parse_pdb_lines_DUM(pdb_lines):
    atoms = []
    if isinstance(pdb_lines, list):  
        if len(pdb_lines) == 0:
            return "This is an empty lipid."
        if all(not isinstance(i, list) for i in pdb_lines):
            for line in pdb_lines:
                if line.startswith("ATOM") and line[16:21].strip() in ["DUM"]:
                    x, y, z = pdb2xyz(line)
                    atoms.append([x, y, z])
            return atoms
        elif all(isinstance(i, list) for i in pdb_lines):  
            for lines in pdb_lines:
                for line in lines:
                    if line.startswith("ATOM") and line[16:21].strip() in ["DUM"]:
                        x, y, z = pdb2xyz(line)
                        atoms.append([x, y, z])
            return atoms



def find_nearest_kdtree(point, points):
    
    
    points = np.array(points)
    point = np.array(point)  

    
    tree = KDTree(points)

    
    if point.ndim == 1:  
        
        nearest_distance, index = tree.query(point)
        nearest_point = points[index]
        return nearest_distance
    elif point.ndim == 2:  
        
        nearest_distances, indices = tree.query(point)
        nearest_points = points[indices]
        return np.min(nearest_distances)


def check_collision_total(lipid_a, lipid_b, colli_threshold = 3.0):
    lipid_a_coords = parse_pdb_lines(lipid_a)
    lipid_b_coords = parse_pdb_lines(lipid_b)
    nearest_distance = find_nearest_kdtree(lipid_a_coords, lipid_b_coords)
    if nearest_distance < colli_threshold:
        return True
    else:
        return False

def check_collision_total_DUU(lipid_a, lipid_b, colli_threshold = 7.6):
    lipid_a_coords = parse_pdb_lines_DUU(lipid_a)
    lipid_b_coords = parse_pdb_lines_DUU(lipid_b)
    nearest_distance = find_nearest_kdtree(lipid_a_coords, lipid_b_coords)
    if nearest_distance < colli_threshold:
        return True
    else:
        return False

def check_collision_total_DUM(lipid_a, lipid_b, colli_threshold = 7.6):
    lipid_a_coords = parse_pdb_lines_DUM(lipid_a)
    lipid_b_coords = parse_pdb_lines_DUM(lipid_b)
    nearest_distance = find_nearest_kdtree(lipid_a_coords, lipid_b_coords)
    if nearest_distance < colli_threshold:
        return True
    else:
        return False

def check_collision(lipid_a, lipid_b, lipd, colli_threshold = 3.0):
    lipid_a_coords = parse_pdb_lines(lipid_a)
    lipid_b_coords = parse_pdb_lines(lipid_b)
    x1, y1, z1 = pdb2xyz(lipid_a[1])
    x2, y2, z2 = pdb2xyz(lipid_b[1])
    lipid_a_head = np.array([x1, y1, z1])
    lipid_b_head = np.array([x2, y2, z2])
    
    
    distance_head = np.linalg.norm(lipid_a_head - lipid_b_head)
    if distance_head < lipd:
        return True
    else:
        nearest_distance = find_nearest_kdtree(lipid_a_coords, lipid_b_coords)
        if nearest_distance < colli_threshold:
            return True
        else:
            return False


def bezier_curve(P0, P1, P2, P3, t):
    return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3

def bezier_tangent(P0, P1, P2, P3,t):
    return -3*(1-t)**2 * P0 + 3*(1-t)**2 * P1 - 6*t*(1-t) * P1 + 6*t*(1-t) * P2 - 3*t**2 * P2 + 3*t**2 * P3


def fit_spline_and_normals(points, interval=0.2):
    
    
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))  
    t = np.concatenate([[0], np.cumsum(distances)])  
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)

    
    t_new = np.arange(t[0], t[-1], interval)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)

    
    dx = cs_x(t_new, 1)
    dy = cs_y(t_new, 1)
    dz = cs_z(t_new, 1)
    tangents = np.column_stack((dx, dy, dz))  

    
    ddx = cs_x(t_new, 2)
    ddy = cs_y(t_new, 2)
    ddz = cs_z(t_new, 2)
    curvatures = np.column_stack((ddx, ddy, ddz))  

    
    normals = np.cross(tangents, np.cross(curvatures, tangents))  
    normal_magnitude = np.linalg.norm(normals, axis=1, keepdims=True)  
    normals /= normal_magnitude  

    
    new_points = np.column_stack((x_new, y_new, z_new))
    return new_points, normals



def generate_surface_points(points, bt=0.1, cti=0.2, al=0.2):
    
    bezier_points         = []
    bezier_points_normals = []
    for i in range(0, len(points)-3, 3):
        p0, p1, p2, p3 = points[i], points[i+1], points[i+2], points[i+3]
        
        _t = max(1, int(np.linalg.norm(np.array(p3) - np.array(p0)) / bt))
        
        t_values = np.linspace(0, 1, _t)  
        
        
        
        curve_points_XZ = np.array([bezier_curve(p0, p1, p2, p3, t) for t in t_values])
        tangents_XZ     = np.array([bezier_tangent(p0, p1, p2, p3, t) for t in t_values])
        
        tangents_XZ = tangents_XZ / np.linalg.norm(tangents_XZ, axis=1, keepdims=True)
        
        normals_XZ = np.array([[-dy, dx] for dx, dy in tangents_XZ])  
        normals_XZ = normals_XZ / np.linalg.norm(normals_XZ, axis=1, keepdims=True)
        
        
        curve_points = np.insert(curve_points_XZ, 1, 0, axis=1)
        normals = np.insert(normals_XZ, 1, 0, axis=1)
        
        bezier_points += curve_points[:-1].tolist()
        bezier_points_normals += normals[:-1].tolist()

    rotated_points, rotated_points_normals = rotate_points_Zaxis_variable(np.array(bezier_points), np.array(bezier_points_normals), arc_length=al)
    
    surface_points         = np.insert(rotated_points, 0, bezier_points[0], axis=0)
    surface_points_normals = np.insert(rotated_points_normals, 0, bezier_points_normals[0], axis=0)
    return bezier_points, bezier_points_normals, surface_points, surface_points_normals


def find_nearest_point(points, normals, target_point):
    points = np.array(points)
    target_point = np.array(target_point)

    
    distances = np.linalg.norm(points - target_point, axis=1)

    
    index = np.argmin(distances)

    
    nearest_point = points[index]
    distance = distances[index]

    
     
    tail_points = points[index:]
    tail_normals = normals[index:]
    return tail_points, tail_normals

def get_cut_points(cut_value, cut_pannel, points, normals, boundary_value, mdist):
    
    try:
        if cut_pannel == 'XZ':  
            
            mask = (np.abs(points[:, 1] - cut_value) <= mdist) & (points[:, 0] <= boundary_value + mdist)
            
        elif cut_pannel == 'YZ':  
            
            mask = (np.abs(points[:, 0] - cut_value) <= mdist) & (points[:, 1] <= boundary_value + mdist)
            
        elif cut_pannel == 'XY': 
            mask = (np.abs(points[:, 0] - points[:, 1]) <= mdist) & (points[:, 0] <= boundary_value + mdist) & (points[:, 1] <= boundary_value + mdist)
            
        else:
            raise ValueError("Invalid axis. Use 'X' for XZ plane or 'Y' for YZ plane.")
    
        cut_points  = points[mask]
        cut_normals = normals[mask]
        return cut_points, cut_normals
    
    except Exception as e:
        
        return [], []
    



def get_curve_length(curve_points):
    
    x, y, z = curve_points.T
    
    tck, u = splprep([x, y, z], k=3, s=0)
    def deriv(t):
        der = splev(t, tck, der=1)
        return np.sqrt(der[0]**2 + der[1]**2 + der[2]**2)

    length, _ = quad(deriv, 0, 1)
    return length



def get_lipd(length, bin_area = 0.6):  
    lipn = int(length/lipd_o + 0.5)
    lipd = length / lipn
    return lipd

def get_xyz_symmetry(pdb_lines, symmetry="XZ"):
    
    mirrored_pdb = []
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x, y, z = pdb2xyz(line)
            sx, sy, sz = x, y, z
            
            if symmetry == "XZ":
                sx = x
                sy = -y
            elif symmetry == "YZ":
                sx = -x
                sy = y
            elif symmetry == "XYZ":
                sx = -x
                sy = -y
            elif symmetry == "XY1":
                sx = y
                sy = x
            elif symmetry == "XY2":
                sx = -y
                sy = x
            elif symmetry == "XY3":
                sx = -y
                sy = -x
            elif symmetry == "XY4":
                sx = y
                sy = -x
            
            new_line = (line[:30] + "{:8.3f}{:8.3f}{:8.3f}".format(sx, sy, sz) + line[54:])
            mirrored_pdb.append(new_line)
        else:
            
            mirrored_pdb.append(line)
    return mirrored_pdb

def get_cut_pannel_lipids(cut_value, cut_pannel, lipid_list, lc_s4_start, lc_s8_start, surface_points, surface_points_normals, mdist, points_occupy, orientation=1, bin_area=0.6, lastp=False):
    pannel_lipids    = []
    pannel_points    = []
    pannel_normals   = []
    
    cut_points, cut_normals = get_cut_points(cut_value, cut_pannel, surface_points, surface_points_normals, boundary_value, mdist)
    if len(cut_points) == 0:
        return [], [], [], []
    
    cut_normals = cut_normals * orientation
    
    lc_s4 = 0
    lc_s8 = 0
    
    if lastp:
        point0, normal0 = cut_points[0], cut_normals[0]
        pointS1 = [-point0[0],  point0[1], point0[2]]
        pointS2 = [ point0[0], -point0[1], point0[2]]
        pointS3 = [-point0[0], -point0[1], point0[2]]
        lipid_lines = lipid_list[lc_s4 + lc_s4_start]
        lc_s4 = lc_s4 + 1 
        new_lipid0 = random_rotate_lipid(rotate_vector(point0, normal0, lipid_lines))
        new_lipidS1 = get_xyz_symmetry(new_lipid0, symmetry="XZ")
        new_lipidS2 = get_xyz_symmetry(new_lipid0, symmetry="YZ")
        new_lipidS3 = get_xyz_symmetry(new_lipid0, symmetry="XYZ")
        
        pannel_points.append(point0)
        pannel_points.append(pointS1)
        pannel_points.append(pointS2)
        pannel_points.append(pointS3)
        
        pannel_lipids.append(new_lipid0)
        pannel_lipids.append(new_lipidS1)
        pannel_lipids.append(new_lipidS2)
        pannel_lipids.append(new_lipidS3)
    else:    
        
        cut_curve_length = get_curve_length(cut_points)    
        if cut_curve_length >= lipd_o:
            lipd = get_lipd(cut_curve_length, bin_area = bin_area_value)
        else:
            lipd = lipd_o

        point0, normal0 = cut_points[0], cut_normals[0]
        
        lipid_lines = lipid_list[lc_s4 + lc_s4_start]
        lc_s4 = lc_s4 + 1
        
        if point0[0] == 0 and point0[1] == 0:
            normal0[0] = 0 
            normal0[1] = 0
        new_lipid0 = random_rotate_lipid(rotate_vector(point0, normal0, lipid_lines))
        pannel_lipids.append(new_lipid0)
        pannel_points.append(point0)
        
        
        
        if point0[0] != 0 and point0[1] != 0:
            pointS1 = [-point0[0],  point0[1], point0[2]]
            pointS2 = [ point0[0], -point0[1], point0[2]]
            pointS3 = [-point0[0], -point0[1], point0[2]]
            new_lipidS1 = get_xyz_symmetry(new_lipid0, symmetry="XZ")
            new_lipidS2 = get_xyz_symmetry(new_lipid0, symmetry="YZ")
            new_lipidS3 = get_xyz_symmetry(new_lipid0, symmetry="XYZ")
            pannel_points.append(pointS1)
            pannel_points.append(pointS2)
            pannel_points.append(pointS3)
            pannel_lipids.append(new_lipidS1)
            pannel_lipids.append(new_lipidS2)
            pannel_lipids.append(new_lipidS3)

        
        _dist = 0
    
        for index, (point, normal) in  enumerate(zip(cut_points, cut_normals)):
            _dist = np.linalg.norm(point - point0)
            if _dist >= lipd_o - 1 and point[0] + mdist >  point[1]:
                if abs(point[1] - 0) <= lipd_o/2 - 1:
                    lipid_lines = lipid_list[lc_s4 + lc_s4_start] 
                elif abs(point[1] - 0) > lipd_o/2 -1 and (abs(point[0]-cut_value) < lipd_o - 1 or abs(point[0] - point[1]) < lipd_o - 1):
                    lipid_lines = lipid_list[lc_s4 + lc_s4_start] 
                else:
                    lipid_lines = lipid_list[lc_s8 + lc_s8_start] 
                new_lipid   = random_rotate_lipid(rotate_vector(point, normal, lipid_lines))
                
                
                colli_x     = check_collision(new_lipid, new_lipid0, lipd_o - 0.25, colli_threshold = c_threshold)
                if colli_x == False:  
                  if len(points_occupy) == 0:
                    colli_total = False
                    colli_DUU   = False
                    colli_DUM   = False
                  else:
                    occupy = points_occupy + pannel_lipids
                    colli_total = check_collision_total(new_lipid, occupy, colli_threshold = c_threshold)
                    colli_DUU   = check_collision_total_DUU(new_lipid, occupy, colli_threshold = lipd_o - 1)
                    colli_DUM   = check_collision_total_DUM(new_lipid, occupy, colli_threshold = lipd_o - 1)
                    
                  if colli_total == False and colli_DUU == False and colli_DUM == False:
                    pannel_lipids.append(new_lipid)
                    pannel_points.append(point)
                    new_lipid0 = new_lipid
                    
                    point0  = point
                    normal0 = normal
                    
                    if abs(point[1] - 0) <= lipd_o/2 - 1:
                        pointS1 = [-point[0],  point[1], point[2]]  
                        pointS4 = [ point[1],  point[0], point[2]]  
                        pointS6 = [-point[1], -point[0], point[2]]  

                        new_lipidS1 = get_xyz_symmetry(new_lipid, symmetry="YZ")
                        new_lipidS4 = get_xyz_symmetry(new_lipid, symmetry="XY1")
                        new_lipidS6 = get_xyz_symmetry(new_lipid, symmetry="XY3")

                        pannel_points.append(pointS1)
                        pannel_points.append(pointS4)
                        pannel_points.append(pointS6)
                        pannel_lipids.append(new_lipidS1)
                        pannel_lipids.append(new_lipidS4)
                        pannel_lipids.append(new_lipidS6)
                        lc_s4      += 1 
                        
                    elif abs(point[1] - 0) > lipd_o/2 - 1 and abs(point[0]-cut_value) >= lipd_o - 1 and abs(point[0] - point[1]) >= lipd_o - 1:
                        pointS1 = [-point[0],  point[1], point[2]]  
                        pointS2 = [ point[0], -point[1], point[2]]  
                        pointS3 = [-point[0], -point[1], point[2]]  
                        pointS4 = [ point[1],  point[0], point[2]]  
                        pointS5 = [-point[1],  point[0], point[2]]  
                        pointS6 = [-point[1], -point[0], point[2]]  
                        pointS7 = [ point[1], -point[0], point[2]]  

                        new_lipidS1 = get_xyz_symmetry(new_lipid, symmetry="XZ")
                        new_lipidS2 = get_xyz_symmetry(new_lipid, symmetry="YZ")
                        new_lipidS3 = get_xyz_symmetry(new_lipid, symmetry="XYZ")
                        new_lipidS4 = get_xyz_symmetry(new_lipid, symmetry="XY1")
                        new_lipidS5 = get_xyz_symmetry(new_lipid, symmetry="XY2")
                        new_lipidS6 = get_xyz_symmetry(new_lipid, symmetry="XY3")
                        new_lipidS7 = get_xyz_symmetry(new_lipid, symmetry="XY4")

                        pannel_points.append(pointS1)
                        pannel_points.append(pointS2)
                        pannel_points.append(pointS3)
                        pannel_points.append(pointS4)
                        pannel_points.append(pointS5)
                        pannel_points.append(pointS6)
                        pannel_points.append(pointS7)
                        pannel_lipids.append(new_lipidS1)
                        pannel_lipids.append(new_lipidS2)
                        pannel_lipids.append(new_lipidS3)
                        pannel_lipids.append(new_lipidS4)
                        pannel_lipids.append(new_lipidS5)
                        pannel_lipids.append(new_lipidS6)
                        pannel_lipids.append(new_lipidS7)
                        
                        lc_s8      += 1 
                        
                    elif abs(point[1] - 0) > lipd_o/2 -1 and (abs(point[0]-cut_value) < lipd_o - 1 or abs(point[0] - point[1]) < lipd_o - 1):
                        pointS1 = [-point[0],  point[1], point[2]]  
                        pointS2 = [ point[0], -point[1], point[2]]  
                        pointS3 = [-point[0], -point[1], point[2]]  

                        new_lipidS1 = get_xyz_symmetry(new_lipid, symmetry="XZ")
                        new_lipidS2 = get_xyz_symmetry(new_lipid, symmetry="YZ")
                        new_lipidS3 = get_xyz_symmetry(new_lipid, symmetry="XYZ")

                        pannel_points.append(pointS1)
                        pannel_points.append(pointS2)
                        pannel_points.append(pointS3)
                        
                        pannel_lipids.append(new_lipidS1)
                        pannel_lipids.append(new_lipidS2)
                        pannel_lipids.append(new_lipidS3)
                        lc_s4      += 1 

        if max(point[0], point[1]) >= boundary_value - 1.0 and _dist >= lipd_o - 1.0:
          if abs(point[1] - 0) <= lipd_o/2 -1:
            lipid_lines = lipid_list[lc_s4 + lc_s4_start]
          else:
            lipid_lines = lipid_list[lc_s8 + lc_s8_start]

          new_lipid = rotate_vector(point, normal, lipid_lines)
          occupy = points_occupy + pannel_lipids
          colli_total = check_collision_total(new_lipid, occupy, colli_threshold = c_threshold)
          colli_DUU   = check_collision_total_DUU(new_lipid, occupy, colli_threshold = lipd_o - 1)
          colli_DUM   = check_collision_total_DUM(new_lipid, occupy, colli_threshold = lipd_o - 1)
          if colli_total == False and colli_DUU == False and colli_DUM == False:
            pannel_lipids.append(new_lipid)
            pannel_points.append(point)
            if abs(point[1] - 0) <= lipd_o/2 -1:
                pointS1 = [-point[0],  point[1], point[2]]  
                pointS4 = [ point[1],  point[0], point[2]]  
                pointS6 = [-point[1], -point[0], point[2]]  

                new_lipidS1 = get_xyz_symmetry(new_lipid, symmetry="YZ")
                new_lipidS4 = get_xyz_symmetry(new_lipid, symmetry="XY1")
                new_lipidS6 = get_xyz_symmetry(new_lipid, symmetry="XY3")

                pannel_points.append(pointS1)
                pannel_points.append(pointS4)
                pannel_points.append(pointS6)
                pannel_lipids.append(new_lipidS1)
                pannel_lipids.append(new_lipidS4)
                pannel_lipids.append(new_lipidS6)
            else:
                pointS1 = [-point[0],  point[1], point[2]]  
                pointS2 = [ point[0], -point[1], point[2]]  
                pointS3 = [-point[0], -point[1], point[2]]  
                pointS4 = [ point[1],  point[0], point[2]]  
                pointS5 = [-point[1],  point[0], point[2]]  
                pointS6 = [-point[1], -point[0], point[2]]  
                pointS7 = [ point[1], -point[0], point[2]]  

                new_lipidS1 = get_xyz_symmetry(new_lipid, symmetry="XZ")
                new_lipidS2 = get_xyz_symmetry(new_lipid, symmetry="YZ")
                new_lipidS3 = get_xyz_symmetry(new_lipid, symmetry="XYZ")
                new_lipidS4 = get_xyz_symmetry(new_lipid, symmetry="XY1")
                new_lipidS5 = get_xyz_symmetry(new_lipid, symmetry="XY2")
                new_lipidS6 = get_xyz_symmetry(new_lipid, symmetry="XY3")
                new_lipidS7 = get_xyz_symmetry(new_lipid, symmetry="XY4")

                pannel_points.append(pointS1)
                pannel_points.append(pointS2)
                pannel_points.append(pointS3)
                pannel_points.append(pointS4)
                pannel_points.append(pointS5)
                pannel_points.append(pointS6)
                pannel_points.append(pointS7)
                pannel_lipids.append(new_lipidS1)
                pannel_lipids.append(new_lipidS2)
                pannel_lipids.append(new_lipidS3)
                pannel_lipids.append(new_lipidS4)
                pannel_lipids.append(new_lipidS5)
                pannel_lipids.append(new_lipidS6)
                pannel_lipids.append(new_lipidS7)
    
    return pannel_points, pannel_lipids, lc_s4, lc_s8

def generate_lipids(diagonal_points, lipid_list, surface_points, surface_points_normals, mdist, lipid_orientation, bin_area=0.6):
    points_occupy = []
    lipids        = []
    l_points      = []
    dist          = 0
    diagonal_length = get_curve_length(diagonal_points)
    d_lipd_o      = math.sqrt(2*(lipd_o**2))
    dist_x_value  = math.sqrt(0.5 * (d_lipd_o**2))
    d_bin         = int(diagonal_length/d_lipd_o + 0.5)
    d_lipd        = diagonal_length/d_bin
    lc_s4_start   = 0
    lc_s8_start   = 1
    for index, point in enumerate(diagonal_points):
        if index == 0:
            start_point = point
            cut_value   = point[0]
            s_points, s_lipids, lc_s4, lc_s8 = get_cut_pannel_lipids(cut_value, "XZ", lipid_list, lc_s4_start, lc_s8_start, surface_points, surface_points_normals, mdist, lipids, orientation=lipid_orientation, bin_area=bin_area_value)
            points_occupy += s_points
            lipids        += s_lipids
            lc_s4_start    = lc_s4_start + lc_s4
            lc_s8_start    = lc_s8_start + lc_s8
            
        elif index == len(diagonal_points)-1:
            if dist > d_lipd_o - 2 and point[0] > boundary_value - 2:
                start_point = point
                cut_value = point[0]
                s_points, s_lipids, lc_s4, lc_s8 = get_cut_pannel_lipids(cut_value, "XZ", lipid_list, lc_s4_start, lc_s8_start, surface_points, surface_points_normals, lipids, mdist, orientation=lipid_orientation, bin_area=bin_area_value, lastp=True)

                points_occupy += s_points
                lipids        += s_lipids
        else:
            dist = np.linalg.norm(point - start_point)
            dist_x = point[0] - cut_value
            x, y, z = point[0], point[1], point[2]
            if dist > d_lipd - 0.5 and dist_x > 1: 
                d_normal = diagonal_points_normals[index] * lipid_orientation
                new_lipid = random_rotate_lipid(rotate_vector(point, d_normal, lipid_list[lc_s4_start]))
                if len(lipids) == 0:
                    colli_total = False
                else:
                    colli_total = check_collision_total(new_lipid, lipids, colli_threshold = c_threshold)
                    dnd         = find_nearest_kdtree(point, points_occupy)
                    
                if colli_total == False and dnd > lipd_o:
                    
                    cut_value = point[0]
                    s_points, s_lipids, lc_s4, lc_s8 = get_cut_pannel_lipids(cut_value, "XZ", lipid_list, lc_s4_start, lc_s8_start, surface_points, surface_points_normals, mdist, lipids, orientation=lipid_orientation, bin_area=bin_area_value)
                    if len(s_points) != 0:
                        start_point = point
                        points_occupy += s_points
                        lipids        += s_lipids
                        lc_s4_start    = lc_s4_start + lc_s4
                        lc_s8_start    = lc_s8_start + lc_s8
                        
    return lipids

def generate_lipid_list(lipid_dict, length, lipid2atom):
    
    
    lipid_list = []  
    lipid_name = []
    lipid_counts = lipid_dict.copy()  

    
    lipid_order = list(lipid_counts.keys())
    
    while len(lipid_list) < length:
        
        for lipid in lipid_order:
            if lipid_counts[lipid] > 0:  
                lipid_list.append(lipid2atom[lipid])
                lipid_counts[lipid] -= 1  
                lipid_name.append(lipid)

        
        lipid_order = [lipid for lipid in lipid_order if lipid_counts[lipid] > 0]

        
        if not lipid_order:
            lipid_counts = lipid_dict.copy()
            lipid_order = list(lipid_counts.keys())
    lipid_list.insert(0, lipid_list[0])
    
    return lipid_list[:length]  


def get_required_lipids(lipid_dict):

    # 直接返回该字典中所有键组成的集合（即脂质种类）
    return set(lipid_dict.keys())



def generate_lipid2atom(length_lnd, lipid_dict, lipid_folder="lipid_lib"):

    # 获取所需的脂质类型
    lipid_list = get_required_lipids(lipid_dict)

    # 初始化结构数据映射字典
    lipid2atom = {}

    for lipid in lipid_list:
        try:
            lipid_data = clean_lipid(pdb_readline(lipid, ff), length_lnd)
            lipid2atom[lipid] = lipid_data
            globals()[f"lipid_{lipid}"] = lipid_data  # 可选：用于调试或临时使用
        except FileNotFoundError:
            print(f"Warning: PDB file for lipid '{lipid}' not found in {lipid_folder}. Skipping...")
        except ValueError as e:
            print(f"Warning: {e} for lipid '{lipid}'. Skipping...")

    return lipid2atom

def generate_lipid_dict(lipid_dict_upper, lipid_dict_lower):
    lipid_dict = {}
    
    all_lipids = set(lipid_dict_upper.keys()).union(set(lipid_dict_lower.keys()))
    
    for lipid in all_lipids:
        upper_count = lipid_dict_upper.get(lipid, 0)  
        lower_count = lipid_dict_lower.get(lipid, 0)
        
        
        lipid_dict[lipid] = upper_count + lower_count
        
    return lipid_dict


def reorder_lipid(lipids, lipid_list):
    new_lipids = {}
    new_lipids_list = []
    for key in lipid_list:
        new_lipids[key] = []
    for lipid in lipids:
        lipidname = lipid[2][17:21]
        new_lipids[lipidname].append(lipid)
    for key, value in new_lipids.items():
        print(key, len(value))
        new_lipids_list += value
    return(new_lipids_list)


def generate_water_xyz(box_X, Z_u, Z_l, bin_water=5.0):
    
    box_X_b   = box_X - bin_water
    Z_u_b     = Z_u - bin_water/2
    Z_l_b     = Z_l - bin_water/2
    
    water_xyz = []
    water     = []
    
    
    box_x_num = int( box_X / bin_water + 0.5)
    bin_x = box_X_b / box_x_num
    box_z_num = int((Z_u + Z_l) / bin_water + 0.5)
    bin_z = (Z_u_b + Z_l_b) / box_z_num
    
    for i in range(box_x_num + 1):
        for j in range(box_x_num + 1):
            for k in range(box_z_num + 1):
                x = -box_X_b/2 + i * bin_x
                y = -box_X_b/2 + j * bin_x
                z = -Z_l_b + k * bin_z
                water_xyz.append(np.array([x, y, z]))
                water.append(f"ATOM {1:6d}  W   W    {1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           W")
    return water_xyz, water

def parse_protein_structure(input_pdb, ot, trans):
    protein = []
    protein_xyz = []

    if ot == "proteinMembrane":
        with open(input_pdb) as f:
            for line in f:
                if line.startswith("ATOM"):
                    x, y, z = pdb2xyz(line)
                    z_translated = z + trans
                    line_modified = line[:46] + f"{z_translated:8.3f}" + line[54:]
                    protein.append(line_modified)
                    protein_xyz.append(np.array([x, y, z_translated]))

    elif ot == "membraneOnly":
        protein = []
        protein_xyz = []

    return protein, protein_xyz



def find_collision_free_points(array1, array2, grid_size=4):
    
    
    occupied_cells = {}

    
    neighbor_offsets = [-1, 0, 1]  


    for lines in array1:
        if isinstance(lines, list):
            for line in lines:
                res_name = line[16:21].strip()
                if res_name not in ["DUM", "DUU"]:
                    for dx in neighbor_offsets:
                        for dy in neighbor_offsets:
                            for dz in neighbor_offsets:
                                x, y, z = pdb2xyz(line)
                                key = (int((x + dx * grid_size) // grid_size),
                                       int((y + dy * grid_size) // grid_size),
                                       int((z + dz * grid_size) // grid_size))
                                occupied_cells[key] = True
        else:
            line = lines
            res_name = line[16:21].strip()
            if res_name not in ["DUM", "DUU"]:
                for dx in neighbor_offsets:
                    for dy in neighbor_offsets:
                        for dz in neighbor_offsets:
                            x, y, z = pdb2xyz(line)
                            key = (int((x + dx * grid_size) // grid_size),
                                  int((y + dy * grid_size) // grid_size),
                                  int((z + dz * grid_size) // grid_size))
                            occupied_cells[key] = True

    
    collision_free_points = []
    for lines in array2:
        if isinstance(lines, list):
            collision = False
            for line in lines:
                res_name = line[16:21].strip()
                if res_name not in ["DUM", "DUU"]:
                    x, y, z = pdb2xyz(line)
                    key = (int(x // grid_size), int(y // grid_size), int(z // grid_size))
                    if key in occupied_cells:
                        collision = True
                        break
            if collision == False:
                collision_free_points.append(lines)
        else:
            line = lines
            res_name = line[16:21].strip()
            if res_name not in ["DUM", "DUU"]:
                x, y, z = pdb2xyz(line)
                key = (int(x // grid_size), int(y // grid_size), int(z // grid_size))
                if key not in occupied_cells:
                    collision_free_points.append(line)
    
    return collision_free_points


def format_protein_atoms(protein, box_X, box_Y, Z_l):
    residue_num = 0
    atom_num = 0

    for line in protein:
        residue_num += 1
        atom_id = atom_num
        atom_name = line[12:16].strip()
        res_name = line[16:21].strip()
        chain_id = line[21].strip()

        _x, _y, _z = pdb2xyz(line)
        x = _x + box_X / 2
        y = _y + box_Y / 2
        z = _z + Z_l

        if res_name != "XDUM":
            atom_num += 1
            if atom_num >= 100000:
                atom_num -= 100000
            if residue_num >= 10000:
                residue_num -= 10000
            print(f"ATOM {atom_num:6d}{atom_name:>5s}{res_name:>5s} {residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")

        
def format_lipids_and_water(reorder_replaced_lipids_free, water_free, box_X, box_Y, Z_l):
    residue_num = 0
    atom_num = 0

    for lipid in reorder_replaced_lipids_free:
        residue_num += 1
        for line in lipid:
            atom_id = atom_num
            atom_name = line[12:16].strip()
            res_name = line[16:21].strip()
            chain_id = line[21].strip()
            _x, _y, _z = pdb2xyz(line)
            x = _x + box_X / 2
            y = _y + box_Y / 2
            z = _z + Z_l

            if res_name not in {"DUM", "DUU"}:
                atom_num += 1
                if atom_num >= 100000:
                    atom_num -= 100000
                if residue_num >= 10000:
                    residue_num -= 10000
                print(f"ATOM {atom_num:6d}{atom_name:>5s}{res_name:>5s} {residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")

    for line in water_free:
        residue_num += 1
        atom_num += 1
        if atom_num >= 100000:
            atom_num -= 100000
        if residue_num >= 10000:
            residue_num -= 10000
        _x, _y, _z = pdb2xyz(line)
        x = _x + box_X / 2
        y = _y + box_Y / 2
        z = _z + Z_l

        print(f"ATOM {atom_num:6d}  W   W    {residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           W")

if __name__ == "__main__":

    results = parser_main()
    input_pdb = results["input_pdb"]
    box_X = results["box_X"]
    box_Y = results["box_Y"]
    Z_u = results["Z_u"]
    Z_l = results["Z_l"]
    control_points = results["control_points"]
    lipid_dict_upper = results["lipid_dict_upper"]
    lipid_dict_lower = results["lipid_dict_lower"]
    plasma = results["plasma"]
    ff = results["ff"]
    ot = results["ot"]
    trans = results["trans"]
    arc = results["arc"]
    bin_area_value = results["bin_area_value"]
    bin_area_upper =  results["bin_area_upper"]
    bin_area_lower = results["bin_area_lower"]
    length_nv_upper = results["length_nv_u"]
    length_nv_lower = results["length_nv_l"]
    
    c_threshold          = 3.0
    lipd_o = math.sqrt(bin_area_value) * 10
    boundary_value = 0.5*(box_X-math.sqrt(bin_area_value)*10 + 0.5)
    bezier_points, bezier_points_normals, surface_points, surface_points_normals = generate_surface_points(control_points, bt=0.2, cti=0.2, al=arc)
    lipid_dict_upper = lipid_dict_upper
    lipid_dict_lower = lipid_dict_lower
    lipid_dict = generate_lipid_dict(lipid_dict_upper, lipid_dict_lower)
    lipid2atom_upper = generate_lipid2atom(length_nv_upper, lipid_dict_upper, lipid_folder="lipid_lib")
    lipid2atom_lower = generate_lipid2atom(length_nv_lower, lipid_dict_lower, lipid_folder="lipid_lib")

    lipid_list_upper = generate_lipid_list(lipid_dict_upper, 2000, lipid2atom_upper)
    lipid_list_lower = generate_lipid_list(lipid_dict_lower, 2000, lipid2atom_lower)

    mdist_value = 0.25
    diagonal_points, diagonal_points_normals = get_cut_points(0, "XY", surface_points, surface_points_normals, boundary_value, mdist=mdist_value)
    lipid_orientation = 1
    lipids_upper = generate_lipids(diagonal_points, lipid_list_upper, surface_points, surface_points_normals, mdist_value, lipid_orientation, bin_area=bin_area_upper)

    lipid_orientation = -1
    lipids_lower = generate_lipids(diagonal_points, lipid_list_lower, surface_points, surface_points_normals, mdist_value, lipid_orientation, bin_area=bin_area_lower)

    reorder_lipids_upper = reorder_lipid(lipids_upper, lipid_dict_upper)
    reorder_lipids_lower = reorder_lipid(lipids_lower, lipid_dict_lower)
    replaced_lipids      = reorder_lipids_upper + reorder_lipids_lower
    
    water_xyz, water = generate_water_xyz(box_X, Z_u, Z_l)
    protein, protein_xyz = parse_protein_structure(input_pdb, ot, trans)
    replaced_lipids_free = find_collision_free_points(protein, replaced_lipids, grid_size=3.0)
    pl = replaced_lipids_free + [protein]
    water_free = find_collision_free_points(pl, water, grid_size=3.5)
    print("free")
    reorder_replaced_lipids_free = reorder_lipid(replaced_lipids_free, lipid_dict)
    print("W", len(water_free))
    print("lipid", len(replaced_lipids_free))
    print("CRYST1  %7.3f  %7.3f  %7.3f  90.00  90.00  90.00 P 1           1"%(box_X, box_Y, (Z_u + Z_l)))
    format_protein_atoms(protein, box_X, box_Y, Z_l)
    format_lipids_and_water(reorder_replaced_lipids_free, water_free, box_X, box_Y, Z_l)
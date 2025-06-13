import numpy as np
import sys

def calculate_centroid(pdb_filename):
    coordinates = []  
    atoms = []        
    with open(pdb_filename, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                x, y, z = map(float, line[30:54].split()[0:3])
                coordinates.append([x, y, z])
                atoms.append(line)
    coordinates_array = np.array(coordinates)
    centroid = np.mean(coordinates_array, axis=0)
    return atoms, centroid

def translate_atoms(atoms, centroid):
    dx, dy, dz = -centroid[0], -centroid[1], -centroid[2]
    translated_atoms = []

    for atom in atoms:
        x = float(atom[30:38].strip())
        y = float(atom[38:46].strip())
        z = float(atom[46:54].strip())

        x_new = x + dx
        y_new = y + dy
        z_new = z + dz

        new_line = atom[:30] + f"{x_new:8.3f}{y_new:8.3f}{z_new:8.3f}" + atom[54:]
        translated_atoms.append(new_line)

    return translated_atoms

def main():
    pdb_filename = sys.argv[1]
    atoms, centroid = calculate_centroid(pdb_filename)
    translated_atoms = translate_atoms(atoms, centroid)

    with open(sys.argv[2], "w") as out_file:
        for line in translated_atoms:
            out_file.write(line)


if __name__ == "__main__":
    main()

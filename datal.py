import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import BondDir as BD
from rdkit.Chem.rdchem import ChiralType as CT
from rdkit.Chem.rdchem import HybridizationType as HT
from rdkit import DataStructs
from rdkit import rdBase

blocker = rdBase.BlockLogs()
cuda_is_available = torch.cuda.is_available()
DEVICE = torch.device ("cuda:0" if cuda_is_available else "cpu")
print ("That code will be running on ", end="")
print (DEVICE)
pt = Chem.GetPeriodicTable()

H_END = 15
C_END = 250
GAMMA = 0.005
FREQUENCY = 500.0
VECTOR_LENGTH = 7500
ATOMS = 80
ATOMS_FEATURES = 136
BOND_FEATURES = 7
DTYPE = torch.float32
FEATURE_VECTOR_LENGTH = ATOMS_FEATURES * ATOMS + BOND_FEATURES * int((ATOMS - 1) * ATOMS / 2)
ATOMS_THRESHOLD = 0.01001
BOND_THRESHOLD = 0.03
ALLOWABE_VALENCIES = [0] + [max(pt.GetValenceList(a)) for a in range(1, 119)]

CHIRAL = [
    CT.CHI_ALLENE,
    CT.CHI_OCTAHEDRAL,
    CT.CHI_OTHER,
    CT.CHI_SQUAREPLANAR,
    CT.CHI_TETRAHEDRAL,
    CT.CHI_TETRAHEDRAL_CCW,
    CT.CHI_TETRAHEDRAL_CW,
    CT.CHI_TRIGONALBIPYRAMIDAL,
    CT.CHI_UNSPECIFIED
]
CHIRAL_LEN = len(CHIRAL)

HYBRIDIZATION = [
    HT.OTHER,
    HT.S,
    HT.SP,
    HT.SP2,
    HT.SP2D,
    HT.SP3,
    HT.SP3D,
    HT.SP3D2,
    HT.UNSPECIFIED
]
HYBRIDIZATION_LEN = len(HYBRIDIZATION)

BONDTYPE = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]

BONDDIR = [
    BD.NONE,
    BD.ENDUPRIGHT,
    BD.ENDDOWNRIGHT
]
BONDTYPE_LEN = len(BONDTYPE)
BONDDIR_LEN = len(BONDDIR)

def compare_molecules(target, mol):
    ms = [target, mol]
    fs = [Chem.RDKFingerprint(x) for x in ms]
    s = DataStructs.FingerprintSimilarity(fs[0], fs[1])
    return s

def build_molecule(tens, check=False):
    tens_ = tens.reshape(-1).cpu().numpy()
    locator = 0
    mol_buider = Chem.RWMol()
    atoms_count = 0
    
    for _ in range(ATOMS):
        atom_f = tens_[locator:locator+ATOMS_FEATURES]
        num_f = atom_f[0:118]
        max_val = np.max(num_f)
        if max_val < ATOMS_THRESHOLD:
            break
        num = np.argmax(num_f) + 1
        atom = Chem.Atom(int(num))
        
        border = 118+CHIRAL_LEN
        chi_f = atom_f[118:border]
        chi = np.argmax(chi_f)
        atom.SetChiralTag(CHIRAL[chi])
        
        hyb_f = atom_f[border:border+HYBRIDIZATION_LEN]
        hyb = np.argmax(hyb_f)
        atom.SetHybridization(HYBRIDIZATION[hyb])
        locator += ATOMS_FEATURES
        mol_buider.AddAtom(atom)
        atoms_count += 1
        
    locator = ATOMS * ATOMS_FEATURES
    atoms_valencies = [0 for _ in range(atoms_count)]
    for t1 in range(atoms_count):
        for t2 in range(t1 + 1, atoms_count):
            bond_f = tens_[locator:locator+BOND_FEATURES]
            bond_t_f = bond_f[0:BONDTYPE_LEN]
            max_val = np.max(bond_t_f)
            if max_val > BOND_THRESHOLD:
                bond_t = np.argmax(bond_t_f)
                num_b = mol_buider.AddBond(t1, t2, BONDTYPE[bond_t]) - 1
                
                bond_d_f = bond_f[BONDTYPE_LEN:BONDTYPE_LEN+BONDDIR_LEN]
                bond_d = np.argmax(bond_d_f)
                bond = mol_buider.GetBondWithIdx(num_b)
                bond.SetBondDir(BONDDIR[bond_d])
                if check:
                    
                    atom1 = mol_buider.GetAtomWithIdx(t1)
                    atom2 = mol_buider.GetAtomWithIdx(t1)
                    a1_num = atom1.GetAtomicNum()
                    a2_num = atom2.GetAtomicNum()
                    bond_as_double = bond.GetBondTypeAsDouble()
                    a1_val = atoms_valencies[t1] + bond_as_double
                    a2_val = atoms_valencies[t2] + bond_as_double
                    if (a1_val > ALLOWABE_VALENCIES[a1_num]) or (a2_val > ALLOWABE_VALENCIES[a2_num]):
                        mol_buider.RemoveBond(t1, t2)
                        continue
                    atoms_valencies[t1] += bond_as_double
                    atoms_valencies[t2] += bond_as_double
            locator += BOND_FEATURES
        locator += BOND_FEATURES * (ATOMS - atoms_count)
        
    mol = mol_buider.GetMol()
    return mol

def one_hot_encode(current_value, vector_lenght):
    ohe = [0.01 for i in range (0,vector_lenght)]
    ohe[current_value] = 0.99
    return ohe

def smiles_to_tensor(sm):
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(sm))
    mol = Chem.AddHs(mol)
    if len(mol.GetAtoms()) > ATOMS:
        return None
    if mol is None:
        return None
    fea = []
    for atom in mol.GetAtoms():
        if (atom.GetFormalCharge() != 0):
            return None
        feature = one_hot_encode(atom.GetAtomicNum() - 1, 118)
        feature += one_hot_encode(CHIRAL.index(atom.GetChiralTag()), CHIRAL_LEN)
        feature += one_hot_encode(HYBRIDIZATION.index(atom.GetHybridization()), HYBRIDIZATION_LEN)
        fea += feature
    
    i = 0
    while i < ATOMS - len(mol.GetAtoms()):
        fea += [0.01 for _ in range(ATOMS_FEATURES)]
        i += 1

    fea += [0.01 for _ in range(BOND_FEATURES * int((ATOMS - 1) * ATOMS / 2))]
    start_ = ATOMS_FEATURES * ATOMS
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i > j:
            _a = i
            i = j
            j = _a
        shift = start_ + BOND_FEATURES * int((2 * ATOMS - i - 1) * i / 2 + j - i - 1)
        fea[shift + BONDTYPE.index(bond.GetBondType())] = 0.99
        shift += BONDTYPE_LEN
        fea[shift + BONDDIR.index(bond.GetBondDir())] = 0.99
    return torch.tensor(fea, dtype=DTYPE).reshape(1, -1)

def get_end(atom_type):
    if atom_type == "H":
        return H_END
    elif atom_type == "C":
        return C_END

def render_nmr_spectrum(intensity_vector, atom_type, title='1H NMR Spectrum'):
    n_points = len(intensity_vector)
    END = get_end(atom_type)
    ppm = np.linspace(0, END, n_points, endpoint=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ppm, intensity_vector, 'k-', linewidth=1)
    ax.invert_xaxis()
    ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=np.max(intensity_vector)*1.2)
    ax.yaxis.set_visible(False)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks(np.arange(0, END, int(END / 15)))
    if (END < 20):
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.tick_params(axis='x', which='both', labelsize=10)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def binomial_coeff(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    num = 1
    for i in range(1, r + 1):
        num = num * (n - i + 1) // i
    return num

def generate_complex_multiplet_vector(A, atom_type, multiplet_str, J_constants, gamma=0.01, frequency=500.0):
    if len(multiplet_str) != len(J_constants) and multiplet_str != 's'and multiplet_str != 'm':
        raise ValueError("Multiplet string and J constants must have same length")
    k_map = {
        's': 0,   # singlet
        'd': 1,   # doublet
        't': 2,   # triplet
        'q': 3,   # quartet
        'p': 4,   # quintet
        'm': None # multiplet
    }
    peaks = [(A, 1.0)]
    total_norm_factor = 1.0
    for symbol, J in zip(multiplet_str, J_constants):
        if symbol not in k_map:
            raise ValueError(f"Unknown multiplicity symbol: '{symbol}'")
        J /= frequency
        k = k_map[symbol]
        new_peaks = []
        
        if symbol == 'm':
            new_peaks = [(pos, inten) for pos, inten in peaks]
            gamma = 0.1
        else:
            for pos, inten in peaks:
                centers = [pos + J * (i - k/2) for i in range(k+1)]
                coeffs = [binomial_coeff(k, i) for i in range(k+1)]
                for i in range(k+1):
                    new_peaks.append((centers[i], inten * coeffs[i]))
            total_norm_factor *= (2 ** k)
        peaks = new_peaks
    peaks = [(pos, inten / total_norm_factor) for pos, inten in peaks]
    n_points = VECTOR_LENGTH
    ppm = np.linspace(0, get_end(atom_type), n_points, endpoint=False)
    
    spectrum = np.zeros(n_points)
    for center, intensity in peaks:
        lorentzian = gamma / ((ppm - center)**2 + gamma**2)
        spectrum += intensity * lorentzian
        
    return spectrum

def build_spectrum_from_data(file_name, atom_type, gamma=0.01, frequency=500.0):
    total_spectrum = np.zeros(VECTOR_LENGTH)
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            if line[0] == atom_type:
                try:
                    A = float(line[2])
                    multiplet_str = line[3]
                except (ValueError, IndexError):
                    continue
                J_constants = tuple([0]) if (line[4] == "" or line[4] == " ") else tuple(float(i) for i in line[4].split(','))
                peak_vector = generate_complex_multiplet_vector(
                    A, 
                    atom_type,
                    multiplet_str, 
                    J_constants,
                    gamma,
                    frequency
                )
                total_spectrum += peak_vector
    max_val = np.max(total_spectrum)
    if max_val > 0.01:
        total_spectrum = total_spectrum / max_val
    
    return total_spectrum
    

def load_data(folder_path, length=10, encoding='utf-8'):
    db = []
    i = 0
    with open("smiles_NP0100001_NP0150000.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        
        for line in csv_reader:
            files = glob.glob(folder_path+"\\" + line["NP_MRD_ID"] + "*.txt")
            if files != None:
                if len(files) > 0:    
                    if len(files) > 1:
                        os.remove(files[1])
                    smiles = smiles_to_tensor(line["SMILES"])
                    if smiles == None:
                        continue
                    h_nmr = build_spectrum_from_data(files[0], "H", gamma=GAMMA, frequency=FREQUENCY)
                    c_nmr = build_spectrum_from_data(files[0], "C", gamma=GAMMA, frequency=FREQUENCY)
                    h_nmr = np.concatenate((h_nmr, c_nmr))
                    val = torch.tensor(h_nmr, dtype = DTYPE).reshape(1, -1)
                    db.append([val, smiles]) 
                    i += 1
                    if i == length:
                        return db
    return db
 
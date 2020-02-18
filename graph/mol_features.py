import numpy as np
import rdkit.Chem as Chem

# The default valid symbols for atom features
SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
           'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
           'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
           'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
           'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re',
           'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm',
           'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', '*', 'UNK']

# The default valid formal charges for atom features
FORMAL_CHARGES = [-2, -1, 0, 1, 2]

# The default valid bond types for bond features
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    None,  # Zero, no bond
]

BT_MAPPING = {
    0.: None,
    1.: Chem.rdchem.BondType.SINGLE,
    2.: Chem.rdchem.BondType.DOUBLE,
    3.: Chem.rdchem.BondType.TRIPLE,
    1.5: Chem.rdchem.BondType.AROMATIC,
}

BT_MAPPING_INV = {v: k for k, v in BT_MAPPING.items()}


def bt_index_to_float(bt_index):
    bond_type = BOND_TYPES[bt_index]
    return BT_MAPPING_INV[bond_type]


# Maximum number of neighbors for an atom
MAX_NEIGHBORS = 10
DEGREES = list(range(MAX_NEIGHBORS))

EXPLICIT_VALENCES = [0, 1, 2, 3, 4, 5, 6]
IMPLICIT_VALENCES = [0, 1, 2, 3, 4, 5]

N_ATOM_FEATS = (len(SYMBOLS) + len(FORMAL_CHARGES) + len(DEGREES) +
                len(EXPLICIT_VALENCES) + len(IMPLICIT_VALENCES) + 1)
N_BOND_FEATS = len(BOND_TYPES) + 1 + 1


def get_bt_index(bond_type):
    """Returns the feature index for a particular bond type.

    Args:
        bond_type: Either a rdchem bond type object (can be None) or a float
            representing the bond type
    """
    if bond_type not in BOND_TYPES:
        assert bond_type in BT_MAPPING
        bond_type = BT_MAPPING[bond_type]

    return BOND_TYPES.index(bond_type)


def onek_unk_encoding(x, set):
    """Returns a one-hot encoding of the given feature."""
    if x not in set:
        x = 'UNK'
    return [int(x == s) for s in set]


def get_atom_features(atom):
    """Given an atom object, returns a numpy array of features."""
    # Atom features are symbol, formal charge, degree, explicit/implicit
    # valence, and aromaticity

    if atom.is_dummy:
        symbol = onek_unk_encoding(atom.symbol, SYMBOLS)
        padding = [0] * (N_ATOM_FEATS - len(symbol))
        feature_array = symbol + padding
    else:
        symbol = onek_unk_encoding(atom.symbol, SYMBOLS)
        fc = onek_unk_encoding(atom.fc, FORMAL_CHARGES)
        degree = onek_unk_encoding(atom.degree, DEGREES)
        exp_valence = onek_unk_encoding(atom.exp_valence, EXPLICIT_VALENCES)
        imp_valence = onek_unk_encoding(atom.imp_valence, IMPLICIT_VALENCES)
        aro = [atom.aro]

        feature_array = symbol + fc + degree + exp_valence + imp_valence + aro
    return np.array(feature_array)


def get_bond_features(bond, bt_only=False):
    """Given an bond object, returns a numpy array of features.

    bond can be None, in which case returns default features for a non-bond.
    """
    # Bond features are bond type, conjugacy, and ring-membership
    if bond is None:
        bond_type = onek_unk_encoding(None, BOND_TYPES)
        conj = [0]
        ring = [0]
    else:
        bond_type = onek_unk_encoding(bond.bond_type, BOND_TYPES)
        conj = [bond.is_conjugated]
        ring = [bond.is_in_ring]

    if bt_only:
        feature_array = bond_type
    else:
        feature_array = bond_type + conj + ring
    return np.array(feature_array)


def get_bt_feature(bond_type):
    """Returns a one-hot vector representing the bond_type."""
    if bond_type in BT_MAPPING:
        bond_type = BT_MAPPING[bond_type]
    return onek_unk_encoding(bond_type, BOND_TYPES)

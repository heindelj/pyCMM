import torch

class Topology:
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.improper_dihedrals = []

    def add_atom(self, atom_type, charge):
        # Add atom to the topology
        pass

    def add_bond(self, atom1, atom2):
        # Add bond to the topology
        pass

    # Similar methods for angles and dihedrals
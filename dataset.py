import torch
import torch_geometric
from torch_geometric.data import Dataset
from pathlib import Path
from ase.io.cif import read_cif
from ase import Atoms
from ase.visualize.plot import plot_atoms
from typing import List
import numpy as np
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import radius_graph

# ============================================================================
# borrowed from Kai
# ============================================================================
# from mendeleev import element
# import re

# def _compute_spdf(Z):
#     """Helper to extract (s,p,d,f) counts for element Z."""
#     econf = element(Z).econf
#     s = sum(int(n) for n in re.findall(r's(\d+)', econf))
#     p = sum(int(n) for n in re.findall(r'p(\d+)', econf))
#     d = sum(int(n) for n in re.findall(r'd(\d+)', econf))
#     f = sum(int(n) for n in re.findall(r'f(\d+)', econf))
#     return s, p, d, f

MAX_Z = 118

# # (a) electronegativity lookup: shape = [MAX_Z]
# ELEC_TABLE = torch.zeros(MAX_Z+1, dtype=torch.float32)
# for Z in range(1, MAX_Z+1):
#     en = element(Z).electronegativity()
#     ELEC_TABLE[Z] = en if en is not None else 0.0

# # (b) Mendeleev feature lookup: shape = [MAX_Z, 11]
# M_FEATURE_TABLE = torch.zeros(MAX_Z+1, 11, dtype=torch.float32)

# for Z in range(1, MAX_Z+1):
#     el = element(Z)
#     s, p, d, f = _compute_spdf(Z)
#     ions = list(el.ionenergies.values())
#     M_FEATURE_TABLE[Z] = torch.tensor([
#         el.glawe_number      or 0.0,
#         el.heat_of_formation or 0.0,
#         el.vdw_radius        or 0.0,
#         ions[0] if len(ions)>0 else 0.0,
#         el.electron_affinity or 0.0,
#         el.dipole_polarizability or 0.0,
#         el.covalent_radius_cordero or 0.0,
#         el.nvalence()        or 0.0,
#         el.hardness()        or 0.0,
#         p,
#         d,
#     ], dtype=torch.float32)

# def compute_miller_index_angles(edge_index, positions, miller_index):
#     miller_normal = torch.tensor(miller_index, dtype=torch.float32)
#     miller_normal = miller_normal / miller_normal.norm()
#     src = positions[edge_index[0]]
#     dst = positions[edge_index[1]]
#     vecs = dst - src
#     vecs = vecs / vecs.norm(dim=1, keepdim=True)
#     cos = torch.sum(vecs * miller_normal, dim=1, keepdim=True)
#     return torch.clamp(cos, -1, 1)
# ============================================================================

M_FEATURE_TABLE = torch.load("M_FEATURE_TABLE.pt")

class AtomsDataset(Dataset):
    """Stores all Atomic-Simulation-Environment Atoms"""

    def __init__(self, folderpath: str):
        """Creates an `AtomsDataset` with cif files under `folderpath`.

        Parameters
        ----------
        folderpath: relative or absolute folder path to all Crystallograph Information Files
        
        Returns
        -------
        None
        """
        self.dataset: List[Atoms] = []
        self.ids: List[str] = []
        self._load(folderpath)
    
    def _load(self, folderpath: str) -> None:
        cif_files = Path(folderpath).iterdir()
        for cif_file in cif_files:
            atoms = read_cif(str(cif_file))
            id = cif_file.stem

            self.dataset.append(atoms)
            self.ids.append(id)
        
        self.tensors = [self._convert(atoms) for atoms in self.dataset]
    
    def _convert(self, atoms: Atoms, cutoff: float = 8.0) -> torch_geometric.data.Data:
        """Convert an `ase.Atoms` into a `torch_geometric.data.Data`.
        
        Parameters
        ----------
        atoms: an `ase.Atoms`
        cutoff: cutoff radius

        Returns
        -------
        tensor representation of `atoms`
        """
        # atom in `atoms`
        atomic_numbers = atoms.get_atomic_numbers()
        n_atoms = len(atomic_numbers)

        # one-hot encoding
        z_encodings = torch.zeros(n_atoms, MAX_Z+1, dtype=torch.long)
        for i in range(n_atoms):
            z = atomic_numbers[i]
            z_encodings[i][z] = 1
        
        # chemical features per node
        x_node_feats = M_FEATURE_TABLE[atomic_numbers]

        # position of every atom
        positions = torch.FloatTensor(atoms.get_positions())

        # 3 unit cell vectors
        cell = torch.FloatTensor(np.array(atoms.get_cell()))

        # tags
        tags = torch.LongTensor(atoms.get_tags())
        
        # edges via radius_graph
        edge_index = radius_graph(positions, r=cutoff, loop=False)

        # bulk edge-attr calculations
        row, col = edge_index
        vecs = positions[col] - positions[row] # [E,3]
        edge_len = vecs.norm(dim=1, keepdim=True) # [E,1]
        edge_dirs = vecs / (edge_len + 1e-9) # [E,3]

        # final edge_attr (4 dimensions: length + direction)
        edge_attr = torch.cat([edge_len, edge_dirs], dim=1)  # [E, 4]
        
        data = torch_geometric.data.Data(x=z_encodings,
                                         x_node_feats=x_node_feats,
                                         edge_attr=edge_attr,
                                         edge_index=edge_index,
                                         cell=cell,
                                         pos=positions,
                                         n_atoms=n_atoms,
                                         tags=tags)

        return data
    
    def __getitem__(self, index: int) -> torch_geometric.data.Data:
        """Get the `Data` at `index`.

        Parameters
        ----------
        index: index of an Atoms
        
        Returns
        -------
        `torch_geometric.data.Data` representation of this `Atoms`
        """
        return self.tensors[index]
    
    def get_atoms(self, index: int) -> Atoms:
        """Get the `Atoms` at `index`.

        Parameters
        ----------
        index: index of an Atoms
        
        Returns
        -------
        this `Atoms`
        """
        return self.dataset[index]
    
    def __len__(self) -> int:
        """Returns number of `Atoms` in `Self`.

        Parameters
        ----------
        None
        
        Returns
        -------
        number of Atoms
        """
        return len(self.dataset)
        
    def view(self, index: int):
        """`ase.visualize.plot_atoms` the `Atoms` at `index`.

        Parameters
        ----------
        index: index of an Atoms
        
        Returns
        -------
        None
        """
        plot_atoms(self.dataset[index])
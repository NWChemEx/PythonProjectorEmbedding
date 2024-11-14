from context import projectorEmbedding

# Various imports
from pyscf import gto 
from pyscf import scf 
from pyscf import dft 
from pyscf import cc
from pyscf.mp.dfump2_native import DFUMP2

from projectorEmbedding import embedding_procedure as embed
from projectorEmbedding import mulliken_partition
from projectorEmbedding import spade_partition
from projectorEmbedding import occupancy_partition

molecule = gto.M(atom='''
Fe     0.00000000     0.00000000     0.00000000
O      0.00000000     2.04032182     0.00000000
H      0.78446834     2.62140570     0.00000000
H     -0.78446834     2.62140570     0.00000000
O      0.00000000     0.00000000     2.04032296
H      0.00000000     0.78446832     2.62140684
H      0.00000000    -0.78446832     2.62140684
O      2.04032122     0.00000000     0.00000000
H      2.62140507     0.00000000     0.78446839
H      2.62140507     0.00000000    -0.78446839
O      0.00000000    -2.04032182     0.00000000
H     -0.78446834    -2.62140570     0.00000000
H      0.78446834    -2.62140570     0.00000000
O      0.00000000     0.00000000    -2.04032296
H      0.00000000    -0.78446832    -2.62140684
H      0.00000000     0.78446832    -2.62140684
O     -2.04032122     0.00000000     0.00000000
H     -2.62140507     0.00000000    -0.78446839
H     -2.62140507     0.00000000     0.78446839
''',
basis="cc-pvdz", 
verbose=4, unit='Angstrom', spin=5, charge=3)

dft_scf = dft.UKS(molecule).density_fit()
dft_scf.xc = "hf"
dft_scf.grids.atom_grid = { 
    'H': (60, 590),
    'N': (70, 590),
    'C': (70, 590),
    'S': (123, 770),
    'Fe': (140, 974),
    }   
dft_scf_e = dft_scf.kernel()

def spade_preset(pyscf_mf, active_atoms=None, c_occ=None):
    return spade_partition(pyscf_mf, active_atoms=active_atoms, c_occ=c_occ, n_act_mos=[14, 9]) 

embed_energy_breakdown = embed(
    dft_scf, [0], embed_meth="mp2", mu_val=None, distribute_mos=spade_preset)

# Print results
print(embed_energy_breakdown)
print(f"Total energy: {sum(embed_energy_breakdown)}")
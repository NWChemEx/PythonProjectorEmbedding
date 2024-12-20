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

molecule = gto.M(atom="""
O1        1.2496550990      3.9803050582      0.2189137483
H1        2.2412205934      3.9395267602      0.2191683946
H2        1.1732740342      2.0063373589      0.8147257804
H3        1.1842079357      2.3041449874     -0.9825219924
C1        0.8080368794      2.6696717552     -0.0009153216
C2       -0.7154512428      2.6375661391     -0.0154354960
F1       -1.1500007694      1.3428176534     -0.2325390021
F2       -1.2079657091      3.0820597461      1.1984483308
F3       -1.1945366304      3.4506923139     -1.0268507943
""",
                 basis='aug-cc-pvdz',
                 verbose=4,
                 unit='Angstrom')

# dft_scf = dft.RKS(molecule)  #.density_fit()
dft_scf = dft.UKS(molecule).density_fit()
dft_scf.xc = "PBE"
dft_scf.grids.prune = None
dft_scf.grids.atom_grid = {'H': (75, 302)}
dft_scf.small_rho_cutoff = 1e-12
dft_scf_e = dft_scf.kernel()

embed_energy_breakdown = embed(
    dft_scf,
    [0, 1, 2, 3, 4],
    embed_meth="mp2",
    #    mu_val=None,
    distribute_mos=spade_partition)

# Print results
print(embed_energy_breakdown)
print(f"Total energy: {sum(embed_energy_breakdown)}")

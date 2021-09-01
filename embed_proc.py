"""
Perform projector based embedding
"""
import numpy as np
from pyscf import scf, dft, mp, cc, df
from projectorEmbedding.embed_utils import get_occ_coeffs
from projectorEmbedding.embed_utils import get_mo_occ_a
from projectorEmbedding.embed_utils import flatten_basis
from projectorEmbedding.embed_utils import purify
from projectorEmbedding.embed_utils import screen_aos
from projectorEmbedding.embed_utils import truncate_basis
from projectorEmbedding.embed_partition import mulliken_partition as pmm
from projectorEmbedding.embed_pyscf_replacements import energy_elec

def embedding_procedure(init_mf, active_atoms=None, embed_meth=None,
                        mu_val=10**6, trunc_lambda=None,
                        distribute_mos=pmm()):
    """
    Manby-like embedding procedure.

    Parameters:
        init_mf:        Full system background calculation. Either an SCF or DFT object from PySCF.
        active_atoms:   List of atom numbers specifying active atoms.
        embed_meth:     String specifying embedded level of theory.
                        Can be "RHF", "MP2", "CCSD", "CCSD(T)", or a density functional.
        mu_val:         Value of level-shift. Uses Huzinaga projection if set to None.
        trunc_lambda:   Float charge threshold for AO truncation screening.
        distribute_mos: Function used to partition the density.

    Returns:
        results: A tuple containing the total embedded energy.
    """
    print("Start Projector Embedding")
    # initial information
    mol = init_mf.mol.copy()
    ovlp = init_mf.get_ovlp()
    c_occ = get_occ_coeffs(init_mf.mo_coeff, init_mf.mo_occ)

    # get active mos
    c_occ_a, _ = distribute_mos(init_mf, active_atoms=active_atoms, c_occ=c_occ)
    if isinstance(c_occ_a, tuple): 
        print(f"Number of active MOs: {c_occ_a[0].shape[1]}, {c_occ_a[1].shape[1]}")
    else:
        print(f"Number of active MOs: {c_occ_a.shape[1]}")

    # make full and subsystem densities
    mo_occ_active = get_mo_occ_a(c_occ_a, init_mf.mo_occ)

    dens = {}
    dens['ab'] = init_mf.make_rdm1()
    dens['a'] = init_mf.make_rdm1(c_occ_a, mo_occ_active)
    dens['b'] = dens['ab'] - dens['a']

    # build embedding potential
    f_ab = init_mf.get_fock()
    v_a = init_mf.get_veff(dm=dens['a'])
    hcore_a_in_b = f_ab - v_a
    if mu_val is None:
        # Huzinaga Projection
        matrix_sum = f_ab @ dens['b'] @ ovlp
        hcore_a_in_b -= 0.5 * (matrix_sum + matrix_sum.swapaxes(-1, -2))
    else:
        # Level-shift projection
        hcore_a_in_b += mu_val * (ovlp @ dens['b'] @ ovlp)

    # get electronic energy for A
    energy_a, _ = energy_elec(init_mf, dm=dens['a'], vhf=v_a, h1e=hcore_a_in_b)
#    energy_a, _ = init_mf.energy_elec(dm=dens['a'], vhf=v_a, h1e=hcore_a_in_b)

    # set new number of electrons
    if len(mo_occ_active) == 2:
        mol.nelectron = int(sum(mo_occ_active[0]) + sum(mo_occ_active[1]))
    else:
        mol.nelectron = int(sum(mo_occ_active))

    if trunc_lambda:
        # AO truncation
        print('Truncating AO Space')

        # alter basis set to facilitate screening
        print(' Flattening Basis Set')
        mol.build(basis=flatten_basis(mol))

        # screen basis sets for truncation
        active_aos, include = screen_aos(mol, active_atoms, dens['a'], ovlp, trunc_lambda)
        print("Active AOs:", len(active_aos), "/", mol.nao)

        if len(active_aos) != mol.nao:
            # make truncated basis set
            mol.build(dump_input=True, basis=truncate_basis(mol, include))

            # make appropiate mean field object with new molecule
            if hasattr(init_mf, 'xc'):
                tinit_mf = dft.RKS(mol)
                tinit_mf.xc = init_mf.xc
            else:
                tinit_mf = scf.RHF(mol)
            if hasattr(init_mf, 'with_df'):
                tinit_mf = df.density_fit(tinit_mf)
                tinit_mf.with_df.auxbasis = init_mf.with_df.auxbasis

            # make truncated tensors
            mesh = np.ix_(active_aos, active_aos)
            hcore_a_in_b = hcore_a_in_b[mesh]
            pure_d_a = 2 * purify(dens['a'][mesh] / 2, ovlp[mesh])

            # truncated initial method (self embedded)
            tinit_mf.get_hcore = lambda *args: hcore_a_in_b
            if np.isnan(pure_d_a).any():
                # Failsafe on purify
                tinit_mf.kernel(dens['a'][mesh])
            else:
                tinit_mf.kernel(pure_d_a)

            # overwrite previous values
            dens['a'] = tinit_mf.make_rdm1()
            v_a = tinit_mf.get_veff(dm=dens['a'])
            energy_a, _ = energy_elec(init_mf, dm=dens['a'], vhf=v_a, h1e=hcore_a_in_b)
#            energy_a, _ = tinit_mf.energy_elec(dm=dens['a'], vhf=v_a, h1e=hcore_a_in_b)
        else:
            print("No AOs truncated")

    print("Calculating A-in-B")

    # make embedding mean field object
    embed_meth = embed_meth.lower()
    if embed_meth in ('rhf', 'mp2', 'ccsd', 'ccsd(t)'):
        mf_embed = scf.RHF(mol)
    elif embed_meth in ('uhf', 'ump2', 'uccsd', 'uccsd(t)'):
        mf_embed = scf.UHF(mol)
    elif "rks" in embed_meth:
        mf_embed = dft.RKS(mol)
        mf_embed.xc = embed_meth.replace("rks-", "")
    elif "uks" in embed_meth:
        mf_embed = dft.UKS(mol)
        mf_embed.xc = embed_meth.replace("uks-", "")
    else: # assume anything else is just a functional name
        mf_embed = dft.RKS(mol)
        mf_embed.xc = embed_meth
    if hasattr(init_mf, 'with_df'):
        mf_embed = df.density_fit(mf_embed)
        mf_embed.with_df.auxbasis = init_mf.with_df.auxbasis
    mf_embed.get_hcore = lambda *args: hcore_a_in_b
    mf_embed.energy_elec = energy_elec.__get__(mf_embed, type(mf_embed))

    # run embedded SCF
    tot_energy_a_in_b = mf_embed.kernel(dens['a'])

    # get electronic energy for embedded part
    energy_a_in_b = tot_energy_a_in_b - mf_embed.energy_nuc()

    # recombined energy with embedded part
    results = (init_mf.e_tot - energy_a + energy_a_in_b, )

    # correlated WF methods
    if embed_meth.lower() in ('mp2', 'ump2'):
        embed_corr = mp.MP2(mf_embed)
        embed_corr.kernel()
        results = results + (embed_corr.e_corr,)
    elif embed_meth.lower() in ('ccsd', 'ccsd(t)', 'uccsd', 'uccsd(t)'):
        embed_corr = cc.CCSD(mf_embed)
        embed_corr.kernel()
        results = results + (embed_corr.emp2,)
        results = results + (embed_corr.e_corr - embed_corr.emp2,)
        if embed_meth.lower() in ('ccsd(t)', 'uccsd(t)'):
            results = results + (embed_corr.ccsd_t(),)

    print("Projector Embedding Complete")
    return results

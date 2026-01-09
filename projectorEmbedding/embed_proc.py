"""
Perform projector based embedding
"""

import numpy as np
import projectorEmbedding
from pyscf import scf, dft, mp, cc, df
from projectorEmbedding.embed_utils import get_occ_coeffs
from projectorEmbedding.embed_utils import get_mo_occ_a
from projectorEmbedding.embed_utils import flatten_basis
from projectorEmbedding.embed_utils import purify
from projectorEmbedding.embed_utils import screen_aos
from projectorEmbedding.embed_utils import truncate_basis
from projectorEmbedding.embed_partition import mulliken_partition as pmm
from projectorEmbedding.embed_partition import spade_partition as spade
from projectorEmbedding.embed_pyscf_replacements import energy_elec
from projectorEmbedding.concentric_localization import ConcentricLocalizer

def embedding_procedure(
    init_mf,
    active_atoms=None,
    embed_meth=None,
    mu_val=10**6,
    trunc_lambda=None,
    distribute_mos=pmm(),
    diis_space=8,
    max_cycle=50,
    chk_file=None,
    chk_start=None,
    cc_econv=1e-07,
    cc_tconv=1e-06,
):
    """
    Manby-like embedding procedure.

    Parameters:
        init_mf:        Full system background calculation.
                        Must be a HF or DFT object.
                        Restricted open-shell not supported.
        active_atoms:   List of atom numbers specifying active atoms.
        embed_meth:     String specifying embedded level of theory.
                        Can be "HF", "MP2", "CCSD", "CCSD(T)", or a density functional.
                        Prepend "U" to WFT methods to specify unresticted for closed-shelled systems.
                        Prepend "UKS-" to DFT methods for the same.
        mu_val:         Value of level-shift. Uses Huzinaga projection if set to None.
        trunc_lambda:   Float charge threshold for AO truncation screening.
        distribute_mos: Function used to partition the density.

    Returns:
        results: A tuple containing the total embedded energy.
    """
    print("Start Projector Embedding")
    print(active_atoms)

    # restricted open-shell not supported
    if isinstance(init_mf, scf.rohf.ROHF) or isinstance(init_mf, dft.roks.ROKS):
        raise RuntimeError("Restricted open-shell methods not supported")

    # unresticted or restricted initial method
    init_is_unrestricted = isinstance(init_mf, scf.uhf.UHF) or isinstance(
        init_mf, dft.uks.UKS
    )

    # initial information
    mol = init_mf.mol.copy()
    ovlp = init_mf.get_ovlp()
    c_occ = get_occ_coeffs(init_mf.mo_coeff, init_mf.mo_occ)

    # get active mos
    print("Partitioning MOs")
    c_occ_a, _ = distribute_mos(init_mf, active_atoms=active_atoms, c_occ=c_occ)
    # print(c_occ_a[0].shape)
    if init_is_unrestricted:
        print(f"Number of active MOs: {c_occ_a[0].shape[1]}, {c_occ_a[1].shape[1]}")
    else:
        print(f"Number of active MOs: {c_occ_a.shape[1]}")

    # get active occupancies
    mo_occ_active = get_mo_occ_a(c_occ_a, init_mf.mo_occ)

    # make full and subsystem densities
    dens = {}
    dens["ab"] = init_mf.make_rdm1()
    dens["a"] = init_mf.make_rdm1(c_occ_a, mo_occ_active)
    dens["b"] = dens["ab"] - dens["a"]

    # build embedding potential
    f_ab = init_mf.get_fock()
    v_a = init_mf.get_veff(dm=dens["a"])
    hcore_a_in_b = f_ab - v_a
    if mu_val is None:
        # Huzinaga Projection
        matrix_sum = f_ab @ dens["b"] @ ovlp
        coeff = 1.0 if dens["b"].ndim == 3 else 0.5
        hcore_a_in_b -= coeff * (matrix_sum + matrix_sum.swapaxes(-1, -2))
    else:
        # Level-shift projection
        hcore_a_in_b += mu_val * (ovlp @ dens["b"] @ ovlp)

    # get electronic energy for A
    energy_a, _ = energy_elec(init_mf, dm=dens["a"], vhf=v_a, h1e=hcore_a_in_b)

    # set new number of electrons
    if init_is_unrestricted:
        mol.nelectron = int(sum(mo_occ_active[0]) + sum(mo_occ_active[1]))
    else:
        mol.nelectron = int(sum(mo_occ_active))

    if trunc_lambda:
        # AO truncation
        print("Truncating AO Space")

        # alter basis set to facilitate screening
        print(" Flattening Basis Set")
        mol.build(basis=flatten_basis(mol))

        # screen basis sets for truncation
        active_aos, include = screen_aos(
            mol, active_atoms, dens["a"], ovlp, trunc_lambda
        )
        print("Active AOs:", len(active_aos), "/", mol.nao)

        if len(active_aos) != mol.nao:
            # make truncated basis set
            mol.build(dump_input=True, basis=truncate_basis(mol, include))

            # make appropiate mean field object with new molecule
            if hasattr(init_mf, "xc"):
                tinit_mf = dft.UKS(mol) if init_is_unrestricted else dft.RKS(mol)
                tinit_mf.xc = init_mf.xc
            else:
                tinit_mf = dft.UHF(mol) if init_is_unrestricted else scf.RHF(mol)
            if hasattr(init_mf, "with_df"):
                tinit_mf = df.density_fit(tinit_mf)
                tinit_mf.with_df.auxbasis = init_mf.with_df.auxbasis

            # make truncated tensors
            mesh3d = np.ix_([0, 1], active_aos, active_aos)
            mesh2d = np.ix_(active_aos, active_aos)
            masked = lambda mat: mat[mesh3d] if mat.ndim == 3 else mat[mesh2d]

            hcore_a_in_b = masked(hcore_a_in_b)
            factor = 1 if init_is_unrestricted else 2
            pure_d_a = factor * purify(masked(dens["a"]) / factor, masked(ovlp))

            # truncated initial method (self embedded)
            tinit_mf.get_hcore = lambda *args: hcore_a_in_b
            tinit_mf.energy_elec = energy_elec.__get__(tinit_mf, type(tinit_mf))
            if np.isnan(pure_d_a).any():
                # Failsafe on purify
                tinit_mf.kernel(masked(dens["a"]))
            else:
                tinit_mf.kernel(pure_d_a)

            # overwrite previous values
            dens["a"] = tinit_mf.make_rdm1()
            v_a = tinit_mf.get_veff(dm=dens["a"])
            energy_a, _ = energy_elec(init_mf, dm=dens["a"], vhf=v_a, h1e=hcore_a_in_b)
        else:
            print("No AOs truncated")

    print("Calculating A-in-B")

    # wavefunction method options
    general_options = ("hf", "mp2", "ccsd", "ccsd(t)")
    unrestricted = tuple("u" + opt for opt in general_options)
    embed_meth = embed_meth.lower()

    # make embedding mean field object
    if embed_meth in general_options + unrestricted:
        if init_is_unrestricted or embed_meth in unrestricted:
            mf_embed = scf.UHF(mol)
        else:
            mf_embed = scf.RHF(mol)
    else:  # assume anything else is just a functional name
        if "uks-" in embed_meth:  # deal with specification of unrestricted
            embed_meth = embed_meth.replace("uks-", "")
            init_is_unrestricted = True
        if init_is_unrestricted:
            mf_embed = dft.UKS(mol)
        else:
            mf_embed = dft.RKS(mol)
        mf_embed.xc = embed_meth
    if hasattr(init_mf, "with_df"):
        mf_embed = df.density_fit(mf_embed)
        mf_embed.with_df.auxbasis = init_mf.with_df.auxbasis
    mf_embed.diis_space = diis_space
    mf_embed.chkfile = chk_file
    mf_embed.max_cycle = max_cycle
    mf_embed.get_hcore = lambda *args: hcore_a_in_b
    mf_embed.energy_elec = energy_elec.__get__(mf_embed, type(mf_embed))

    # run embedded SCF
    if chk_start:
        init_dm = mf_embed.from_chk(chk_start)
        tot_energy_a_in_b = mf_embed.kernel(init_dm)
    else:
        tot_energy_a_in_b = mf_embed.kernel(dens["a"])
    
    cl = ConcentricLocalizer(mf_embed, active_atoms=active_atoms)
    mf_embed = cl.localize_virtual()

    # get electronic energy for embedded part
    energy_a_in_b = tot_energy_a_in_b - mf_embed.energy_nuc()
    print("Embedded region energy: ", energy_a_in_b)

    # recombined energy with embedded part
    results = (init_mf.e_tot - energy_a + energy_a_in_b,)

    # norb_alpha = mf_embed.mo_coeff[0].shape[1]
    # norb_beta = mf_embed.mo_coeff[1].shape[1]
    # total_orbitals = norb_alpha + norb_beta

    # print(f"Total number of orbitals: {total_orbitals}")
    print(hcore_a_in_b.shape)
    # correlated WF methods
    if "mp2" in embed_meth:
        embed_corr = mp.MP2(mf_embed)
        embed_corr.kernel()
        results = results + (embed_corr.e_corr,)
    elif "ccsd" in embed_meth or "ccsd(t)" in embed_meth:
        embed_corr = cc.CCSD(mf_embed)
        embed_corr.conv_tol = cc_econv
        embed_corr.conv_tol_normt = cc_tconv
        embed_corr.kernel()
        results = results + (embed_corr.emp2,)
        results = results + (embed_corr.e_corr - embed_corr.emp2,)
        if "ccsd(t)" in embed_meth:
            results = results + (embed_corr.ccsd_t(),)

    print("Projector Embedding Complete")
    return results, mf_embed

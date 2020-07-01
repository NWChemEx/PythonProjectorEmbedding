"""
Old version of the embedding procedure
Saved for legacy reasons
"""

from pyscf import scf, dft, lo, mp, cc
import numpy as np
from projectorEmbedding.embed_utils import make_dm
from projectorEmbedding.embed_proc import mulliken_partition


def embedding_procedure(pyscf_mol,
                        init_mf,
                        xc=None,
                        corr_meth=None,
                        active_atoms=None,
                        charge_threshold=0.4,
                        localize=True,
                        mu=10**6,
                        distribute_mos=mulliken_partition,
                        debug_dist=False,
                        molden=None):
    """Performs a projector embedding calculation"""
    # Manby-like embedding procedure. Kinda rigid at the moment.
    mol = pyscf_mol.copy()

    # get intermediates
    ovlp = init_mf.get_ovlp()
    mo_occ = init_mf.mo_occ
    C_occ = init_mf.mo_coeff[:,mo_occ>0]
    if localize:
        C_occ = lo.PM(mol, C_occ).kernel()
        
    # split mos
    init_mf.charge_threshold = charge_threshold
    C_occ_A, C_occ_B = distribute_mos(mol, init_mf,
                                      active_atoms=active_atoms,
                                      C_occ=C_occ,
                                      debug=debug_dist)

    if molden:
        from pyscf import tools
        tools.molden.from_mo(pyscf_mol, molden + 'AB_local.molden', C_occ)
        tools.molden.from_mo(pyscf_mol, molden + 'A_local.molden', C_occ_A)
        tools.molden.from_mo(pyscf_mol, molden + 'B_local.molden', C_occ_B)

    # make full and subsystem densities
    den_mat = make_dm(C_occ, mo_occ[mo_occ>0]) 
    den_mat_A = make_dm(C_occ_A, mo_occ[:C_occ_A.shape[1]])
    den_mat_B = make_dm(C_occ_B, mo_occ[:C_occ_B.shape[1]])

    # get full and subsystem potential terms and make potentials
    v_AB = init_mf.get_veff(mol=mol, dm=den_mat)
    v_A = init_mf.get_veff(mol=mol, dm=den_mat_A)
    v_B = init_mf.get_veff(mol=mol, dm=den_mat_B)

    # get electronic energies of parts
    energy_elec_AB, energy_coulomb_AB = init_mf.energy_elec(dm=den_mat, vhf=v_AB)
    energy_elec_A, energy_coulomb_A = init_mf.energy_elec(dm=den_mat_A, vhf=v_A)
    energy_elec_B, energy_coulomb_B = init_mf.energy_elec(dm=den_mat_B, vhf=v_B)
    energy_nonadd = (energy_coulomb_AB - energy_coulomb_A - energy_coulomb_B)
    energy_nuc = init_mf.energy_nuc()
    
    # build embedding potential
    hcore = init_mf.get_hcore()
    projector = np.dot(np.dot(ovlp, den_mat_B), ovlp)
    v_embed = v_AB - v_A
    hcore_A_in_B = hcore + v_embed + (mu * projector)

    # make embedding mean field object
    if (xc is None) or (xc.lower()=='rhf'):
        mf_embed = scf.RHF(mol)
    else:
        if corr_meth:
            print("***Attempting to perform correlated WF method on top of DFT***")
        mf_embed = dft.RKS(mol)
        mf_embed.xc = xc
    mf_embed.mol.nelectron = int(sum(mo_occ[:C_occ_A.shape[1]]))
    mf_embed.get_hcore = lambda *args: hcore_A_in_B

    # run embedded SCF
    mf_embed.kernel(den_mat_A)

    # get new values from embedded SCF
    C_occ_A_inB = mf_embed.mo_coeff[:,mf_embed.mo_occ>0]
    den_mat_A_inB = make_dm(C_occ_A_inB, mf_embed.mo_occ[mf_embed.mo_occ>0])

    # make potential for embedded result
    v_A_inB = mf_embed.get_veff(mol, den_mat_A_inB)
    
    # get electronic energy for embedded part
    energy_elec_A_inB, energy_coulomb_A_inB = mf_embed.energy_elec(dm=den_mat_A_inB, vhf=v_A_inB, h1e=hcore)
    
    # perturbation correction
    energy_projector = mu * np.dot(den_mat_A_inB, projector).trace()

    # first-order correction
    first_order_correction = np.dot(den_mat_A_inB - den_mat_A, v_AB - v_A).trace()

    # recombined energy with embedded part
    embed_energy = energy_elec_A_inB + energy_elec_B + energy_nonadd
    
    # correlated WF method
    if corr_meth:
        if corr_meth.lower()=='mp2':
            embed_corr = mp.MP2(mf_embed)
        elif corr_meth.lower() in ['ccsd', 'ccsd(t)']:
            embed_corr = cc.CCSD(mf_embed)
        embed_corr.kernel()
        embed_corr_energy = embed_corr.e_corr
        if corr_meth.lower()=='ccsd(t)':
            embed_corr_et = embed_corr.ccsd_t()
            embed_corr_energy += embed_corr_et
    else: # just return 0 
        embed_corr_energy = 0
        
    return (embed_energy, first_order_correction, energy_projector, embed_corr_energy, energy_nuc, energy_elec_AB)


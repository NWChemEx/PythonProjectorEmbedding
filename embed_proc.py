"""
Perform projector based embedding
"""
from pyscf import scf
from pyscf import dft
from pyscf import lo
from pyscf import mp
from pyscf import cc
import numpy as np
from scipy.linalg import fractional_matrix_power
from projectorEmbedding.embed_utils import make_dm
from projectorEmbedding.embed_utils import flatten_basis
from projectorEmbedding.embed_utils import purify
from projectorEmbedding.embed_utils import screen_aos
from projectorEmbedding.embed_utils import truncate_basis

def mulliken_partition(charge_threshold=0.4):
    """splits the MOs into active and frozen parts based on charge threshold."""
    def internal(pyscf_mf, active_atoms=None, c_occ=None):
        offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()

        # if occupied coeffs aren't provided, get the ones from the mean field results.
        if c_occ is None:
            c_occ = pyscf_mf.mo_coeff[:, pyscf_mf.mo_occ > 0]
        overlap = pyscf_mf.get_ovlp()

        # for each mo, go through active atoms and check the charge on that atom.
        # if charge on active atom is greater than threshold, mo added to active list.
        active_mos = []
        if active_atoms == []: # default case for NO active atoms
            return c_occ[:, []], c_occ[:, :]
        if active_atoms is None:
            return c_occ[:, :], c_occ[:, []]

        for mo_i in range(c_occ.shape[1]):

            rdm_mo = make_dm(c_occ[:, [mo_i]], pyscf_mf.mo_occ[mo_i])

            atoms = active_atoms

            for atom in atoms:
                offset = offset_ao_by_atom[atom, 2]
                extent = offset_ao_by_atom[atom, 3]

                overlap_atom = overlap[:, offset:extent]
                rdm_mo_atom = rdm_mo[:, offset:extent]

                q_atom_mo = np.einsum('ij,ij->', rdm_mo_atom, overlap_atom)

                if q_atom_mo > internal.charge_threshold:
                    active_mos.append(mo_i)
                    break

        # all mos not active are frozen
        frozen_mos = [i for i in range(c_occ.shape[1]) if i not in active_mos]

        return c_occ[:, active_mos], c_occ[:, frozen_mos]
    internal.charge_threshold = charge_threshold

    return internal

def spade_partition(pyscf_mf, active_atoms=None, c_occ=None):
    """SPADE partitioning scheme"""

    # things coming from molecule.
    offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()

    # things coming from mean field calculation.
    mo_occ = pyscf_mf.mo_occ
    if c_occ is None:
        c_occ = pyscf_mf.mo_coeff[:, mo_occ > 0]
    overlap = pyscf_mf.get_ovlp()

    active_aos = []
    for atom in active_atoms:
        active_aos += list(range(offset_ao_by_atom[atom, 2], offset_ao_by_atom[atom, 3]))

    overlap_sqrt = fractional_matrix_power(overlap, 0.5)
    c_orthogonal_ao = (overlap_sqrt @ c_occ)[active_aos, :]
    _, s_vals, v_vecs = np.linalg.svd(c_orthogonal_ao, full_matrices=True)

    if len(s_vals) == 1:
        n_act_mos = 1
    else:
        deltas = [-(s_vals[i + 1] - s_vals[i]) for i in range(len(s_vals)-1)]
        n_act_mos = np.argpartition(deltas, -1)[-1]+1

    c_a = c_occ @ v_vecs.T[:, :n_act_mos]
    c_b = c_occ @ v_vecs.T[:, n_act_mos:]

    return c_a, c_b

def embedding_procedure(init_mf, active_atoms=None, embed_meth=None,
                        mu_val=10**6, trunc_lambda=None, localize=True,
                        distribute_mos=mulliken_partition(0.4)):
    """Manby-like embedding procedure."""
    # initial information
    mol = init_mf.mol.copy()
    ovlp = init_mf.get_ovlp()
    c_occ = init_mf.mo_coeff[:, init_mf.mo_occ > 0]

    # localize orbitals
    if localize:
        c_occ = lo.PM(mol, c_occ).kernel()

    # get active mos
    c_occ_a, _ = distribute_mos(init_mf, active_atoms=active_atoms, c_occ=c_occ)

    # make full and subsystem densities
    dens = {}
    dens['ab'] = make_dm(c_occ, init_mf.mo_occ[init_mf.mo_occ > 0])
    dens['a'] = make_dm(c_occ_a, init_mf.mo_occ[:c_occ_a.shape[1]])
    dens['b'] = dens['ab'] - dens['a']

    # get subsystem A potential
    v_a = init_mf.get_veff(dm=dens['a'])

    # build embedding potential
    hcore_a_in_b = init_mf.get_hcore()
    hcore_a_in_b += init_mf.get_veff(dm=dens['ab']) - v_a
    hcore_a_in_b += mu_val * (ovlp @ dens['b'] @ ovlp)

    # get electronic energy for A
    energy_a, _ = init_mf.energy_elec(dm=dens['a'], vhf=v_a, h1e=hcore_a_in_b)

    # set new number of electrons
    mol.nelectron = int(sum(init_mf.mo_occ[:c_occ_a.shape[1]]))

    if trunc_lambda:
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
            tinit_mf = type(init_mf)(mol)
            if hasattr(init_mf, 'xc'):
                tinit_mf.xc = init_mf.xc

            # make truncated tensors
            mesh = np.ix_(active_aos, active_aos)
            hcore_a_in_b = hcore_a_in_b[mesh]
            pure_d_a = 2 * purify(dens['a'][mesh] / 2, ovlp[mesh])

            # make initial guess
            h_eff = hcore_a_in_b + tinit_mf.get_veff(dm=pure_d_a)
            e_mos, c_mos = tinit_mf.eig(h_eff, ovlp[mesh])
            occ_mos = tinit_mf.get_occ(e_mos, c_mos)
            guess_d = make_dm(c_mos[:, occ_mos > 0], occ_mos[occ_mos > 0])

            # truncated initial method (self embedded)
            tinit_mf.get_hcore = lambda *args: hcore_a_in_b
            tinit_mf.kernel(guess_d)

            # overwrite previous values
            dens['a'] = tinit_mf.make_rdm1()
            v_a = tinit_mf.get_veff(dm=dens['a'])
            energy_a, _ = tinit_mf.energy_elec(dm=dens['a'], vhf=v_a, h1e=hcore_a_in_b)

    # make embedding mean field object
    if embed_meth.lower() in ['rhf', 'mp2', 'ccsd', 'ccsd(t)']:
        mf_embed = scf.RHF(mol)
    else: # assume anything else is a functional name
        mf_embed = dft.RKS(mol)
        mf_embed.xc = embed_meth
    mf_embed.get_hcore = lambda *args: hcore_a_in_b

    # run embedded SCF
    tot_energy_a_in_b = mf_embed.kernel(dens['a'])

    # get electronic energy for embedded part
    energy_a_in_b = tot_energy_a_in_b - mf_embed.energy_nuc()

    # recombined energy with embedded part
    results = (init_mf.e_tot - energy_a + energy_a_in_b, )

    # correlated WF method
    if embed_meth.lower() == 'mp2':
        embed_corr = mp.MP2(mf_embed)
        embed_corr.kernel()
        results = results + (embed_corr.e_corr,)

    elif embed_meth.lower() in ['ccsd', 'ccsd(t)']:
        embed_corr = cc.CCSD(mf_embed)
        embed_corr.kernel()
        results = results + (embed_corr.emp2,)
        results = results + (embed_corr.e_corr - embed_corr.emp2,)
        if embed_meth.lower() == 'ccsd(t)':
            results = results + (embed_corr.ccsd_t(),)

    return results

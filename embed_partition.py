"""
Functions to partition the density
"""
import numpy as np

from scipy.linalg import fractional_matrix_power

from pyscf import lo

from projectorEmbedding.embed_utils import make_dm

def mulliken_partition(charge_threshold=0.4, localize=True):
    """Splits the MOs into active and frozen parts based on charge threshold."""
    def internal(pyscf_mf, active_atoms=None, c_occ=None):
        offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()

        # if occupied coeffs aren't provided, get the ones from the mean field results.
        if c_occ is None:
            c_occ = pyscf_mf.mo_coeff[:, pyscf_mf.mo_occ > 0]
        overlap = pyscf_mf.get_ovlp()

        # localize orbitals
        if internal.localize:
            c_occ = lo.PM(pyscf_mf.mol, c_occ).kernel()

        # for each mo, go through active atoms and check the charge on that atom.
        # if charge on active atom is greater than threshold, mo added to active list.
        active_mos = []
        if active_atoms == []: # default case for NO active atoms
            return c_occ[:, []], c_occ[:, :]
        if active_atoms is None:
            return c_occ[:, :], c_occ[:, []]

        for mo_i in range(c_occ.shape[1]):

            rdm_mo = make_dm(c_occ[:, [mo_i]], pyscf_mf.mo_occ[mo_i])

            for atom in active_atoms:
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
    internal.localize = localize

    return internal

def occupancy_partition(occupancy_threshold=0.2, localize=True):
    """Splits the MOs into active and frozen parts based on occupancy threshold."""
    def internal(pyscf_mf, active_atoms=None, c_occ=None):
        # Handle orbital coefficients
        if c_occ is None:
            c_occ = pyscf_mf.mo_coeff[:, pyscf_mf.mo_occ > 0]
        if internal.localize:
            c_occ = lo.PM(pyscf_mf.mol, c_occ).kernel()
        overlap = pyscf_mf.get_ovlp()

        # Handle active atoms
        if active_atoms == []: # default case for NO active atoms
            return c_occ[:, []], c_occ[:, :]
        if active_atoms is None:
            return c_occ[:, :], c_occ[:, []]

        # Find AOs on active atoms
        offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()
        active_aos = []
        for atom in active_atoms:
            active_aos += list(range(offset_ao_by_atom[atom, 2], offset_ao_by_atom[atom, 3]))
        mesh = np.ix_(active_aos, active_aos)

        # Find MO occupancies in active AOs and sort accordingly
        active_mos = []
        frozen_mos = []
        for mo_i in range(c_occ.shape[1]):
            rdm_mo = make_dm(c_occ[:, [mo_i]], 1)
            dm_mo = rdm_mo @ overlap
            if np.trace(dm_mo[mesh]) > internal.occupancy_threshold:
                active_mos.append(mo_i)
            else:
                frozen_mos.append(mo_i)

        return c_occ[:, active_mos], c_occ[:, frozen_mos]

    internal.occupancy_threshold = occupancy_threshold
    internal.localize = localize

    return internal

def spade_partition(pyscf_mf, active_atoms=None, c_occ=None, n_act_mos=None):
    """SPADE partitioning scheme"""
    # things coming from molecule.
    offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()

    # things coming from mean field calculation.
    mo_occ = pyscf_mf.mo_occ
    if c_occ is None:
        c_occ = pyscf_mf.mo_coeff[:, mo_occ > 0]
    overlap = pyscf_mf.get_ovlp()

    # Find AOs on active atoms
    active_aos = []
    for atom in active_atoms:
        active_aos += list(range(offset_ao_by_atom[atom, 2], offset_ao_by_atom[atom, 3]))

    # Convert to orthogonal AOs and SVD submatrix
    overlap_sqrt = fractional_matrix_power(overlap, 0.5)
    c_orthogonal_ao = (overlap_sqrt @ c_occ)[active_aos, :]
    _, s_vals, v_vecs = np.linalg.svd(c_orthogonal_ao, full_matrices=True)

    # Identify partitioning split
    if len(s_vals) == 1:
        n_act_mos = 1
    else:
        if not n_act_mos:
            if len(s_vals) != v_vecs.shape[0]:
                s_vals = np.append(s_vals, [0.0])
            deltas = [-(s_vals[i + 1] - s_vals[i]) for i in range(len(s_vals)-1)]
            n_act_mos = np.argpartition(deltas, -1)[-1]+1

    # Make SPADE orbitals
    c_a = c_occ @ v_vecs.T[:, :n_act_mos]
    c_b = c_occ @ v_vecs.T[:, n_act_mos:]

    return c_a, c_b

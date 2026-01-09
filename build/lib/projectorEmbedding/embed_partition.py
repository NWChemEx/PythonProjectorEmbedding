"""
Functions to partition the density
"""

import numpy as np
from math import ceil
from math import floor
from pyscf import lo
from scipy.linalg import fractional_matrix_power
from projectorEmbedding.embed_utils import get_occ_coeffs


def mulliken_partition(charge_threshold=0.4, localize=True):
    """Splits the MOs into active and frozen parts based on charge threshold."""

    def internal(pyscf_mf, active_atoms=None, c_occ=None):
        # if occupied coeffs aren't provided, get the ones from the mean field results.
        if c_occ is None:
            c_occ = get_occ_coeffs(pyscf_mf.mo_coeff, pyscf_mf.mo_occ)

        if isinstance(c_occ, tuple):
            alpha_active, alpha_inactive = internal(
                pyscf_mf, active_atoms, c_occ=c_occ[0]
            )
            beta_active, beta_inactive = internal(
                pyscf_mf, active_atoms, c_occ=c_occ[1]
            )
            return (alpha_active, beta_active), (alpha_inactive, beta_inactive)

        mo_occ = pyscf_mf.mo_occ
        if len(mo_occ) == 2:
            mo_occ = mo_occ[0]

        offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()
        overlap = pyscf_mf.get_ovlp()

        # localize orbitals
        if internal.localize:
            print("Localize = True")
            c_occ = lo.PM(pyscf_mf.mol, c_occ).kernel()
        # for each mo, go through active atoms and check the charge on that atom.
        # if charge on active atom is greater than threshold, mo added to active list.
        active_mos = []
        if active_atoms == []:  # default case for NO active atoms
            print("Active atoms = True")
            return c_occ[:, []], c_occ[:, :]
        if active_atoms is None:
            print("Active None = True")
            return c_occ[:, :], c_occ[:, []]

        # print(c_occ.shape[1],"Shape, Sahil")
        for mo_i in range(c_occ.shape[1]):
            rdm_mo = np.dot(c_occ[:, [mo_i]] * mo_occ[mo_i], c_occ[:, [mo_i]].conj().T)

            for atom in active_atoms:
                offset = offset_ao_by_atom[atom, 2]
                extent = offset_ao_by_atom[atom, 3]
                # print(offset,extent,"offset")
                ao_indices = np.arange(offset, offset + extent)
                overlap_atom = overlap[np.ix_(ao_indices, ao_indices)]
                rdm_mo_atom = rdm_mo[np.ix_(ao_indices, ao_indices)]

                # print(overlap_atom)
                # print(rdm_mo_atom)
                q_atom_mo = np.einsum("ij,ij->", rdm_mo_atom, overlap_atom)
                # print(q_atom_mo)

                if q_atom_mo > (internal.charge_threshold * (mo_occ[mo_i] / 2)):
                    # print("charge command is working", q_atom_mo ,internal.charge_threshold* (mo_occ[mo_i] / 2))
                    active_mos.append(mo_i)
                    break

        # all mos not active are frozen
        frozen_mos = [i for i in range(c_occ.shape[1]) if i not in active_mos]
        # print(c_occ.shape)

        return c_occ[:, active_mos], c_occ[:, frozen_mos]

    internal.charge_threshold = charge_threshold
    internal.localize = localize

    return internal


def occupancy_partition(occupancy_threshold=0.6, localize=True):
    """Splits the MOs into active and frozen parts based on occupancy threshold."""

    def internal(pyscf_mf, active_atoms=None, c_occ=None):
        # Handle orbital coefficients
        if c_occ is None:
            c_occ = get_occ_coeffs(pyscf_mf.mo_coeff, pyscf_mf.mo_occ)

        if isinstance(c_occ, tuple):
            alpha_active, alpha_inactive = internal(
                pyscf_mf, active_atoms, c_occ=c_occ[0]
            )
            beta_active, beta_inactive = internal(
                pyscf_mf, active_atoms, c_occ=c_occ[1]
            )
            return (alpha_active, beta_active), (alpha_inactive, beta_inactive)

        if internal.localize:
            c_occ = lo.PM(pyscf_mf.mol, c_occ).kernel()
        overlap = pyscf_mf.get_ovlp()

        # Handle active atoms
        if active_atoms == []:  # default case for NO active atoms
            return c_occ[:, []], c_occ[:, :]
        if active_atoms is None:
            return c_occ[:, :], c_occ[:, []]

        # Find AOs on active atoms
        offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()
        print(offset_ao_by_atom)
        active_aos = []
        for atom in active_atoms:
            active_aos += list(
                range(offset_ao_by_atom[atom, 2], offset_ao_by_atom[atom, 3])
            )
        print(active_aos)
        mesh = np.ix_(active_aos, active_aos)

        # Find MO occupancies in active AOs and sort accordingly
        active_mos = []
        frozen_mos = []

        print(c_occ.shape[1])
        print(mesh)

        for mo_i in range(c_occ.shape[1]):
            rdm_mo = np.dot(c_occ[:, [mo_i]], c_occ[:, [mo_i]].conj().T)
            dm_mo = rdm_mo @ overlap
            print(np.trace(dm_mo[mesh]))
            if np.trace(dm_mo[mesh]) > internal.occupancy_threshold:
                active_mos.append(mo_i)
            else:
                frozen_mos.append(mo_i)

        return c_occ[:, active_mos], c_occ[:, frozen_mos]

    internal.occupancy_threshold = occupancy_threshold
    internal.localize = localize

    return internal


def spade_partition():
    def internal(pyscf_mf, active_atoms=None, c_occ=None, n_act_mos=None):
        """SPADE partitioning scheme"""
        if c_occ is None:
            c_occ = get_occ_coeffs(pyscf_mf.mo_coeff, pyscf_mf.mo_occ)

        if isinstance(c_occ, tuple):
            if n_act_mos is None:
                n_act_mos_a = n_act_mos_b = None
            else:
                n_act_mos_a, n_act_mos_b = n_act_mos
            alpha_active, alpha_inactive = internal(
                pyscf_mf, active_atoms, c_occ=c_occ[0], n_act_mos=n_act_mos_a
            )
            beta_active, beta_inactive = internal(
                pyscf_mf, active_atoms, c_occ=c_occ[1], n_act_mos=n_act_mos_b
            )
            return (alpha_active, beta_active), (alpha_inactive, beta_inactive)

        offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()
        overlap = pyscf_mf.get_ovlp()

        # for each mo, go through active atoms and check the charge on that atom.
        # if charge on active atom is greater than threshold, mo added to active list.
        if active_atoms == []:  # default case for NO active atoms
            return c_occ[:, []], c_occ[:, :]
        if active_atoms is None:
            return c_occ[:, :], c_occ[:, []]

        # Find AOs on active atoms
        active_aos = []
        for atom in active_atoms:
            active_aos += list(
                range(offset_ao_by_atom[atom, 2], offset_ao_by_atom[atom, 3])
            )

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
                deltas = [-(s_vals[i + 1] - s_vals[i]) for i in range(len(s_vals) - 1)]
                n_act_mos = np.argpartition(deltas, -1)[-1] + 1

        # Make SPADE orbitals
        c_a = c_occ @ v_vecs.T[:, :n_act_mos]
        c_b = c_occ @ v_vecs.T[:, n_act_mos:]

        return c_a, c_b

    return internal


def single_atom_mulliken_partition(
    pyscf_mf, active_atoms=None, c_occ=None, n_act_mos=None, localize=True
):
    # if occupied coeffs aren't provided, get the ones from the mean field results.
    if len(active_atoms) > 1:
        raise Exception("This partition isn't intended for more than one active atom.")

    if c_occ is None:
        c_occ = get_occ_coeffs(pyscf_mf.mo_coeff, pyscf_mf.mo_occ)

    if isinstance(c_occ, tuple):
        n_act_mos_a, n_act_mos_b = n_act_mos
        alpha_active, alpha_inactive = single_atom_mulliken_partition(
            pyscf_mf, active_atoms, c_occ=c_occ[0], n_act_mos=n_act_mos_a
        )
        beta_active, beta_inactive = single_atom_mulliken_partition(
            pyscf_mf, active_atoms, c_occ=c_occ[1], n_act_mos=n_act_mos_b
        )
        return (alpha_active, beta_active), (alpha_inactive, beta_inactive)

    mo_occ = pyscf_mf.mo_occ
    if len(mo_occ) == 2:
        mo_occ = mo_occ[0]

    offset_ao_by_atom = pyscf_mf.mol.offset_ao_by_atom()
    overlap = pyscf_mf.get_ovlp()

    # localize orbitals
    if localize:
        c_occ = lo.PM(pyscf_mf.mol, c_occ).kernel()

    # for each mo, go through active atoms and check the charge on that atom.
    # if charge on active atom is greater than threshold, mo added to active list.
    if active_atoms == []:  # default case for NO active atoms
        return c_occ[:, []], c_occ[:, :]
    if active_atoms is None:
        return c_occ[:, :], c_occ[:, []]

    charges = []
    for mo_i in range(c_occ.shape[1]):
        rdm_mo = np.dot(c_occ[:, [mo_i]] * mo_occ[mo_i], c_occ[:, [mo_i]].conj().T)

        atom = active_atoms[0]
        offset = offset_ao_by_atom[atom, 2]
        extent = offset_ao_by_atom[atom, 3]

        overlap_atom = overlap[:, offset:extent]
        rdm_mo_atom = rdm_mo[:, offset:extent]

        charges.append(np.einsum("ij,ij->", rdm_mo_atom, overlap_atom))

    active_mos = sorted(
        [i for i in range(len(charges))], key=lambda x: charges[x], reverse=True
    )[:n_act_mos]

    # all mos not active are frozen
    frozen_mos = [i for i in range(c_occ.shape[1]) if i not in active_mos]

    return c_occ[:, active_mos], c_occ[:, frozen_mos]

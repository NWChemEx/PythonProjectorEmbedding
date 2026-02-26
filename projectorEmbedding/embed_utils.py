"""
Some utility functions
"""

from copy import deepcopy
import numpy as np


def get_occ_coeffs(coefficients, occupancies):
    if coefficients.ndim == 3:
        alpha_coeffs = get_occ_coeffs(coefficients[0], occupancies[0])
        beta_coeffs = get_occ_coeffs(coefficients[1], occupancies[1])
        return (alpha_coeffs, beta_coeffs)
    return coefficients[:, occupancies > 0]


def get_mo_occ_a(c_occ_a, mo_occ):
    if len(mo_occ.shape) == 2:
        mo_occ_alpha = get_mo_occ_a(c_occ_a[0], mo_occ[0])
        mo_occ_beta = get_mo_occ_a(c_occ_a[1], mo_occ[1])
        return (mo_occ_alpha, mo_occ_beta)
    return mo_occ[..., mo_occ > 0][..., -c_occ_a.shape[1] :]


def flatten_basis(mol):
    """Flattens out PySCF's basis set representation"""
    flatten_set = deepcopy(mol._basis)

    for atom_type in flatten_set:
        # step through basis set by atoms
        atom_basis = flatten_set[atom_type]

        for i, i_val in enumerate(atom_basis):
            # for each shell, see contains more than one contraction
            if len(i_val[1]) > 2:
                new_contractions = []
                i_nparray = np.asarray(i_val[1:])

                for contraction in range(len(i_val[1]) - 1):
                    # split individual contractions into seperate lists
                    new_contractions.append(
                        [i_val[0]] + i_nparray[:, [0, contraction + 1]].tolist()
                    )

                for i_ctr, new_contraction in enumerate(new_contractions):
                    # place the split contractions into the overall structure
                    if i_ctr != 0:
                        atom_basis.insert(i + i_ctr, new_contraction)
                    else:
                        atom_basis[i] = new_contraction

    return flatten_set


def screen_aos(mol, active_atoms, den_mat_a, ovlp, trunc_lambda):
    """Screen AOs for truncation"""
    include = [False] * mol.nbas
    active_aos = []

    for shell in range(mol.nbas):
        aos_in_shell = list(range(mol.ao_loc[shell], mol.ao_loc[shell + 1]))

        if (
            mol.bas_atom(shell) not in active_atoms
        ):  # shells on active atoms are always kept
            for ao_i in aos_in_shell:
                if den_mat_a.ndim == 3:
                    charge = den_mat_a[0, ao_i, ao_i] * ovlp[ao_i, ao_i]
                    if charge > (trunc_lambda / 2):
                        break
                    charge = den_mat_a[1, ao_i, ao_i] * ovlp[ao_i, ao_i]
                    if charge > (trunc_lambda / 2):
                        break
                else:
                    charge = den_mat_a[ao_i, ao_i] * ovlp[ao_i, ao_i]
                    if charge > trunc_lambda:
                        break
            else:  # if nothing trips the break, these AOs aren't kept and we move on
                continue

        include[shell] = True
        active_aos += aos_in_shell

    return active_aos, include


def truncate_basis(mol, mask):
    """Truncate the molecule basis set according to the shell mask"""
    print(" Making Truncated Basis Set")
    trunc_basis = deepcopy(mol._basis)
    trunc_basis1 = deepcopy(mol._basis)
    # print(trunc_basis)
    for i_atom in range(mol.natm):
        symbol = mol.atom_symbol(i_atom)
        shell_ids = mol.atom_shell_ids(i_atom)
        # print(symbol, shell_ids, mask )
        # keep only the AOs in shells that were not screened

        filtered = []
        # print(symbol,trunc_basis1[symbol],"Sahil")
        for i, shell in enumerate(shell_ids):
            # print(i,shell,trunc_basis[symbol][i])
            if mask[shell]:
                # print(i,trunc_basis[symbol][i])
                filtered.append(trunc_basis[symbol][i])
        trunc_basis1[symbol] = filtered

        # trunc_basis[symbol] = \
        #    [trunc_basis[symbol][i] for i, shell in enumerate(shell_ids) if mask[shell]]
        print(symbol, shell_ids, [mask[shell] for shell in shell_ids])

        if trunc_basis1[symbol] == []:  # if all AOs on an atom are gone, remove it
            del trunc_basis1[symbol]

    # print(trunc_basis1)
    return trunc_basis1


def purify(matrix, overlap, rtol=1e-5, atol=1e-8, max_iter=15):
    """McWeeny Purification of Density Matrix"""
    print("Begin Purification")

    if matrix.ndim == 3:
        matrix_alpha = purify(matrix[0], overlap, rtol, atol, max_iter)
        matrix_beta = purify(matrix[1], overlap, rtol, atol, max_iter)
        return np.array((matrix_alpha, matrix_beta))

    i = 0
    density = matrix @ overlap

    omega = np.trace(np.linalg.matrix_power(density, 2) - density) ** 2
    print(i, "Omega = ", omega)

    while (i < max_iter) and not np.allclose(omega, 0.0, rtol=rtol, atol=atol):
        i += 1
        matrix = 3 * (matrix @ overlap @ matrix) - 2 * (
            matrix @ overlap @ matrix @ overlap @ matrix
        )
        density = matrix @ overlap
        omega = np.trace(np.linalg.matrix_power(density, 2) - density) ** 2
        print(i, "Omega = ", omega)

    if i < max_iter:
        print("Purification Completed\n")
    else:
        print("Max Iterations hit\n")

    return matrix

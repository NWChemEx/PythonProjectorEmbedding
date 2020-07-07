"""
Some utility functions
"""
from copy import deepcopy
import numpy as np

def make_dm(coeffs, occupency):
    """Given MO coefficients and occupencies, return the density matrix"""
    return np.dot(coeffs * occupency, coeffs.T.conj())

def flatten_basis(basis_set):
    """Flattens out PySCF's basis set representation"""
    flatten_set = deepcopy(basis_set)

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
                    new_contractions.append([i_val[0]] \
                        + i_nparray[:, [0, contraction + 1]].tolist())

                for i_ctr, new_contraction in enumerate(new_contractions):
                    # place the split contractions into the overall structure
                    if i_ctr != 0:
                        atom_basis.insert(i + i_ctr, new_contraction)
                    else:
                        atom_basis[i] = new_contraction

    return flatten_set

def purify(matrix, overlap, rtol=1e-5, atol=1e-8, max_iter=15):
    """McWeeny Purification of Density Matrix"""
    print('Begin Purification')
    i = 0
    density = matrix @ overlap

    omega = np.trace(np.linalg.matrix_power(density, 2) - density)**2
    print(i, 'Omega = ', omega)

    while (i < max_iter) and not np.allclose(omega, 0.0, rtol=rtol, atol=atol):
        i += 1
        matrix = 3 * (matrix @ overlap @ matrix) \
            - 2 * (matrix @ overlap @ matrix @ overlap @ matrix)
        density = matrix @ overlap
        omega = np.trace(np.linalg.matrix_power(density, 2) - density)**2
        print(i, 'Omega = ', omega)

    if i < max_iter:
        print('Purification Completed\n')
    else:
        print("Max Iterations hit")

    return matrix

def screen_aos(mol, active_atoms, den_mat_a, ovlp, trunc_lambda):
    """Screen AOs for truncation"""
    include = [False] * mol.nbas
    active_aos = []

    for shell in range(mol.nbas):
        aos_in_shell = list(range(mol.ao_loc[shell], mol.ao_loc[shell + 1]))

        if mol.bas_atom(shell) not in active_atoms: # shells on active atoms are always kept
            for ao_i in aos_in_shell:
                if (den_mat_a[ao_i, ao_i] * ovlp[ao_i, ao_i]) > trunc_lambda:
                    break
            else: # if nothing trips the break, these AOs aren't kept and we move on
                continue

        include[shell] = True
        active_aos += aos_in_shell

    return active_aos, include

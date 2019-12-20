import numpy as np
from scipy import linalg
from copy import deepcopy

def make_dm(coeffs, occupency):
    return np.dot(coeffs * occupency, coeffs.T.conj())

def flatten_basis(basis_set):
    # flattens out PySCF's basis set representation
    for atom_type in basis_set: 
        # step through basis set by atoms
        atom_basis = basis_set[atom_type]

        for i, i_val in enumerate(atom_basis):
            # for each shell, see contains more than one contraction
            if len(i_val[1]) > 2:
                new_contractions = []
                i_nparray = np.asarray(i_val[1:])

                for contraction in range(len(i_val[1]) - 1):
                    # split individual contractions into seperate lists
                    new_contractions.append([i_val[0]] + i_nparray[:, [0, contraction + 1]].tolist())

                for i_ctr, new_contraction in enumerate(new_contractions):
                    # place the split contractions into the overall structure
                    if i_ctr != 0:
                        atom_basis.insert(i + i_ctr, new_contraction)
                    else:
                        atom_basis[i] = new_contraction
                        
def purify(M, S, rtol=1e-5, atol=1e-8, max_iter=15):
    # McWeeny Purification of Density Matrix
    print('Begin Purification')
    i = 0
    P = M @ S

    omega = np.trace(np.linalg.matrix_power(P, 2) - P)**2
    print(i, 'Omega = ', omega)

    while (i < max_iter) and not np.allclose(omega, 0.0, rtol=rtol, atol=atol):
        i+=1
        M = 3 * (M @ S @ M) - 2 * (M @ S @ M @ S @ M)
        P = M @ S
        omega = np.trace(np.linalg.matrix_power(P, 2) - P)**2
        print(i, 'Omega = ', omega)

    if (i < max_iter):print('Purification Completed\n')
    else:print("Max Iterations hit")

    return M


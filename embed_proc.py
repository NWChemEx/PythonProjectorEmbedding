"""
Perform projector based embedding
"""
from copy import deepcopy
from pyscf import scf
from pyscf import dft
from pyscf import lo
from pyscf import mp
from pyscf import cc
import numpy as np
from scipy import linalg
from projectorEmbedding.embed_utils import make_dm
from projectorEmbedding.embed_utils import flatten_basis
from projectorEmbedding.embed_utils import purify

def mulliken_partition(pyscf_mol, pyscf_mf, active_atoms=None, c_occ=None, charge_threshold=0.4):
    """splits the MOs into active and frozen parts based on charge threshold."""

    # things coming from molecule.
    offset_ao_by_atom = pyscf_mol.offset_ao_by_atom()

    # things coming from mean field calculation.
    mo_occ = pyscf_mf.mo_occ

    # if occupied coeffs aren't provided, get the ones from the mean field results.
    if c_occ is None:
        c_occ = pyscf_mf.mo_coeff[:, mo_occ > 0]
    overlap = pyscf_mf.get_ovlp()

    # Hacky way to handle passing threshold.
    # Allows the function call for both partition schemes to be the same in procedure function.
    try:
        charge_threshold = pyscf_mf.charge_threshold
    except:
        pass

    # for each mo, go through active atoms and check the charge on that atom.
    # if charge on active atom is greater than threshold, mo added to active list.
    active_mos = []
    if active_atoms == []: # default case for NO active atoms
        return c_occ[:, []], c_occ[:, :]
    if active_atoms is None:
        return c_occ[:, :], c_occ[:, []]

    for mo_i in range(c_occ.shape[1]):

        rdm_mo = make_dm(c_occ[:, [mo_i]], mo_occ[mo_i])

        atoms = active_atoms

        for atom in atoms:
            offset = offset_ao_by_atom[atom, 2]
            extent = offset_ao_by_atom[atom, 3]

            overlap_atom = overlap[:, offset:extent]
            rdm_mo_atom = rdm_mo[:, offset:extent]

            q_atom_mo = np.einsum('ij,ij->', rdm_mo_atom, overlap_atom)

            if q_atom_mo > charge_threshold:
                active_mos.append(mo_i)
                break

    # all mos not active are frozen
    frozen_mos = [i for i in range(c_occ.shape[1]) if i not in active_mos]

    return c_occ[:, active_mos], c_occ[:, frozen_mos]

def spade_partition(pyscf_mol, pyscf_mf, active_atoms=None, c_occ=None):
    """SPADE partitioning scheme"""

    # things coming from molecule.
    offset_ao_by_atom = pyscf_mol.offset_ao_by_atom()

    # things coming from mean field calculation.
    mo_occ = pyscf_mf.mo_occ
    if c_occ is None:
        c_occ = pyscf_mf.mo_coeff[:, mo_occ > 0]
    overlap = pyscf_mf.get_ovlp()

    active_aos = []
    for atom in active_atoms:
        active_aos += list(range(offset_ao_by_atom[atom, 2], offset_ao_by_atom[atom, 3]))

    overlap_sqrt = linalg.fractional_matrix_power(overlap, 0.5)
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

def embedding_procedure(pyscf_mol, init_mf, active_atoms=None, xc=None, corr_meth=None, mu=10**6,
                        trunc_lambda=None, localize=True, distribute_mos=mulliken_partition,
                        charge_threshold=0.4):
    """Manby-like embedding procedure."""
    mol = pyscf_mol.copy()
    tot_energy_ab = init_mf.e_tot

    # get intermediates
    ovlp = init_mf.get_ovlp()
    hcore = init_mf.get_hcore()
    mo_occ = init_mf.mo_occ
    c_occ = init_mf.mo_coeff[:, mo_occ > 0]
    if localize:
        c_occ = lo.PM(mol, c_occ).kernel()

    # split mos
    init_mf.charge_threshold = charge_threshold
    c_occ_a, _ = distribute_mos(mol, init_mf, active_atoms=active_atoms, c_occ=c_occ)

    # make full and subsystem densities
    den_mat_ab = make_dm(c_occ, mo_occ[mo_occ > 0])
    den_mat_a = make_dm(c_occ_a, mo_occ[:c_occ_a.shape[1]])
    den_mat_b = den_mat_ab - den_mat_a

    # get full and subsystem potential terms and new make potentials
    v_ab = init_mf.get_veff(mol=mol, dm=den_mat_ab)
    v_a = init_mf.get_veff(mol=mol, dm=den_mat_a)

    v_embed = v_ab - v_a
    projector = np.dot(np.dot(ovlp, den_mat_b), ovlp)

    # get electronic energy for A
    energy_a, _ = init_mf.energy_elec(dm=den_mat_a, vhf=v_a, h1e=hcore + v_embed + (mu * projector))

    # build embedding potential
    hcore_a_in_b = hcore + v_embed + (mu * projector)
    n_act_elecs = int(sum(mo_occ[:c_occ_a.shape[1]]))
    mol.nelectron = n_act_elecs

    if trunc_lambda:
        print('Truncating AO Space')

        # alter basis set to facilitate screening
        print(' Flattening Basis Set')
        flattened_basis = flatten_basis(mol._basis)
        mol.basis = flattened_basis
        mol.build()

        # screen basis sets for truncation
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

        print("Active AOs:", len(active_aos), "/", mol.nao)

        if len(active_aos) == mol.nao:
            print("No AOs Truncated")
            trunc_lambda = None
        else:            
            # make truncated basis set
            print(' Making Truncated Basis Set')
            trunc_basis = deepcopy(flattened_basis)
            for i_atom in range(mol.natm):
                symbol = mol.atom_symbol(i_atom)
                shell_ids = mol.atom_shell_ids(i_atom)

                # Keep on the AOs in shells that were not screened
                trunc_basis[symbol] = \
                    [trunc_basis[symbol][i] for i, shell in enumerate(shell_ids) if include[shell]]
                print(symbol, shell_ids, [include[shell] for shell in shell_ids])

                if trunc_basis[symbol] == []: # If all AOs on an atom are screened, remove the atom
                    del trunc_basis[symbol]

            # make molecule with smaller basis set
            mol.basis = trunc_basis
            mol.build(dump_input=True)

            # Make appropiate mean field object with new molecule
            tinit_mf = type(init_mf)(mol)
            try:
                tinit_mf.xc = init_mf.xc
            except:
                pass

            # make truncated tensors
            hcore_a_in_b = hcore_a_in_b[np.ix_(active_aos, active_aos)]
            tden_mat_a = den_mat_a[np.ix_(active_aos, active_aos)]
            tovlp = ovlp[np.ix_(active_aos, active_aos)]
            pure_d_a = 2 * purify(tden_mat_a / 2, tovlp)

            # make initial guess
            tveff = tinit_mf.get_veff(dm=pure_d_a)
            h_eff = hcore_a_in_b + tveff
            e_mos, c_mos = tinit_mf.eig(h_eff, tovlp)
            occ_mos = tinit_mf.get_occ(e_mos, c_mos)
            guess_d = make_dm(c_mos[:, occ_mos > 0], occ_mos[occ_mos > 0])

            # truncated initial method (self embedded)
            tinit_mf.get_hcore = lambda *args: hcore_a_in_b
            tinit_mf.kernel(guess_d)

            # Overwrite previous values
            den_mat_a = tinit_mf.make_rdm1()
            v_a = tinit_mf.get_veff(dm=den_mat_a)
            energy_a, _ = tinit_mf.energy_elec(dm=den_mat_a, vhf=v_a, h1e=hcore_a_in_b)

    # make embedding mean field object
    if (xc is None) or (xc.lower() == 'rhf'):
        mf_embed = scf.RHF(mol)
    else:
        if corr_meth:
            print("***Attempting to perform correlated WF method on top of DFT***")
        mf_embed = dft.RKS(mol)
        mf_embed.xc = xc
    mf_embed.get_hcore = lambda *args: hcore_a_in_b

    # run embedded SCF
    tot_energy_a_in_b = mf_embed.kernel(den_mat_a)

    # get electronic energy for embedded part
    energy_a_in_b = tot_energy_a_in_b - mf_embed.energy_nuc()

    # recombined energy with embedded part
    embed_energy = energy_a_in_b - energy_a + tot_energy_ab

    results = (embed_energy, )

    # correlated WF method
    if corr_meth:
        if corr_meth.lower()=='mp2':
            embed_corr = mp.MP2(mf_embed)
            embed_corr.kernel()
            results = results + (embed_corr.e_corr,)

        elif corr_meth.lower() in ['ccsd', 'ccsd(t)']:
            embed_corr = cc.CCSD(mf_embed)
            embed_corr.kernel()
            results = results + (embed_corr.emp2,)
            results = results + (embed_corr.e_corr - embed_corr.emp2,)

        if corr_meth.lower()=='ccsd(t)':
            results = results + (embed_corr.ccsd_t(),)

    return results


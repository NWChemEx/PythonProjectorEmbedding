"""
Perform projector based embedding
"""
from copy import deepcopy
from pyscf import gto, scf, dft, lo, mp, cc, tools
import numpy as np
from scipy import linalg
from projectorEmbedding.embed_utils import make_dm, flatten_basis, purify

def mulliken_partition(pyscf_mol, pyscf_mf, active_atoms=None, C_occ=None, charge_threshold=0.4, debug=False):
    # splits the MOs into active and frozen parts based on charge threshold.
    
    # things coming from molecule.
    natoms = pyscf_mol.natm
    naos = pyscf_mol.nao
    offset_ao_by_atom = pyscf_mol.offset_ao_by_atom()

    # things coming from mean field calculation.
    mo_occ = pyscf_mf.mo_occ
    
    # if occupied coeffs aren't provided, get the ones from the mean field results.
    if C_occ is None: 
        C_occ = pyscf_mf.mo_coeff[:,mo_occ>0]
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
        return C_occ[:, []], C_occ[:, :]
    if active_atoms is None:
        return C_occ[:, :], C_occ[:, []]
    
    # debugging should print out same Mulliken Charges as standard mf.mulliken_pop().
    if debug: # initialize list to aggregate total electronic charge on atoms.
        q_on_atom = [0.0] * natoms
    
    for mo in range(C_occ.shape[1]):
        if debug: print('#### MO ', mo)

        rdm_mo = make_dm(C_occ[:, [mo]], mo_occ[mo])
        
        if debug: # if debugging, want to span all atoms.
            atoms = list(range(natoms))
        else: # otherwise, only need to span active ones.
            atoms = active_atoms
        
        for atom in atoms:
            offset = offset_ao_by_atom[atom, 2]
            extent = offset_ao_by_atom[atom, 3]
            
            overlap_atom = overlap[:, offset:extent]
            rdm_mo_atom = rdm_mo[:, offset:extent]
            
            q_atom_mo = np.einsum('ij,ij->', rdm_mo_atom, overlap_atom)
            if debug: 
                print(atom, q_atom_mo)
                q_on_atom[atom] += q_atom_mo
            
            if q_atom_mo > charge_threshold:
                if not debug: # if mo is active, no need to check further atoms. move to next mo.
                    active_mos.append(mo)
                    break
                else: # debugging requires a little more logic to avoid double counting mos.
                    if (mo not in active_mos) and (atom in active_atoms): 
                        active_mos.append(mo)
                    
                
    if debug: # final calculation and printout of Mulliken Charges for comparison.
        atomic_charges = pyscf_mol.atom_charges()
        print([[atomic_charges[atom] - q_on_atom[atom] for atom in range(natoms)]])

    # all mos not active are frozen
    frozen_mos = [i for i in range(C_occ.shape[1]) if i not in active_mos]
    
    return C_occ[:, active_mos], C_occ[:, frozen_mos]

def spade_partition(pyscf_mol, pyscf_mf, active_atoms=None, C_occ=None, debug=False):
    # things coming from molecule.
    offset_ao_by_atom = pyscf_mol.offset_ao_by_atom()

    # things coming from mean field calculation.
    mo_occ = pyscf_mf.mo_occ
    if C_occ is None: 
        C_occ = pyscf_mf.mo_coeff[:,mo_occ>0]
    overlap = pyscf_mf.get_ovlp()

    active_aos = []
    for atom in active_atoms:
        active_aos += list(range(offset_ao_by_atom[atom, 2], offset_ao_by_atom[atom, 3]))
    
    overlap_sqrt = linalg.fractional_matrix_power(overlap, 0.5)
    C_orthogonal_ao = (overlap_sqrt @ C_occ)[active_aos, :]
    u, s, v = np.linalg.svd(C_orthogonal_ao, full_matrices=True)
    
    if len(s)==1:
        n_act_mos = 1
    else:
        ds = [-(s[i + 1] - s[i]) for i in range(len(s)-1)]
        n_act_mos = np.argpartition(ds, -1)[-1]+1

    C_A = C_occ @ v.T[:,:n_act_mos]
    C_B = C_occ @ v.T[:,n_act_mos:]
        
    return C_A, C_B

def embedding_procedure(pyscf_mol,
                        init_mf,
                        xc=None,
                        corr_meth=None,
                        active_atoms=None,
                        charge_threshold=0.4,
                        localize=True,
                        mu=10**6,
                        distribute_mos=mulliken_partition,
                        trunc_lambda=None,
                        debug_dist=False,
                        molden=None):
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
    den_mat_AB = make_dm(C_occ, mo_occ[mo_occ>0]) 
    den_mat_A = make_dm(C_occ_A, mo_occ[:C_occ_A.shape[1]])
    den_mat_B = make_dm(C_occ_B, mo_occ[:C_occ_B.shape[1]])

    # get full and subsystem potential terms and make potentials
    v_AB = init_mf.get_veff(mol=mol, dm=den_mat_AB)
    v_A = init_mf.get_veff(mol=mol, dm=den_mat_A)
    v_B = init_mf.get_veff(mol=mol, dm=den_mat_B)

    # get electronic energies of parts
    energy_elec_AB, energy_coulomb_AB = init_mf.energy_elec(dm=den_mat_AB, vhf=v_AB)
    energy_elec_A, energy_coulomb_A = init_mf.energy_elec(dm=den_mat_A, vhf=v_A)
    energy_elec_B, energy_coulomb_B = init_mf.energy_elec(dm=den_mat_B, vhf=v_B)
    energy_nonadd = (energy_coulomb_AB - energy_coulomb_A - energy_coulomb_B)
    energy_nuc = init_mf.energy_nuc()
    
    # build embedding potential
    hcore = init_mf.get_hcore()
    v_embed = v_AB - v_A
    projector = np.dot(np.dot(ovlp, den_mat_B), ovlp)
    hcore_A_in_B = hcore + v_embed + (mu * projector)
    n_act_elecs = int(sum(mo_occ[:C_occ_A.shape[1]]))
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
    
            if (mol.bas_atom(shell) not in active_atoms): # shells on active atoms are automatically kept
                for ao in aos_in_shell:
                    if ((den_mat_A[ao, ao] * ovlp[ao, ao]) > trunc_lambda): break
                else:
                    continue # if nothing trips the break, these AOs aren't kept and we move on
    
            include[shell] = True
            active_aos += aos_in_shell
            
        print(len(active_aos))
    
        if len(active_aos)==mol.nao:
            print("No AOs Truncated")
        else:            
            # make truncated basis set
            print(' Making Truncated Basis Set')
            trunc_basis = deepcopy(flattened_basis)
            for ia in range(mol.natm):
                symbol = mol.atom_symbol(ia)
                shell_ids = mol.atom_shell_ids(ia)

                # Keep on the AOs in shells that were not screened
                trunc_basis[symbol] = [trunc_basis[symbol][i] for i,shell in enumerate(shell_ids) if include[shell]]
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
            tden_mat_A = den_mat_A[np.ix_(active_aos, active_aos)]
            projector = projector[np.ix_(active_aos, active_aos)]
            v_embed = v_embed[np.ix_(active_aos, active_aos)]
            hcore = hcore[np.ix_(active_aos, active_aos)]
            tovlp = ovlp[np.ix_(active_aos, active_aos)]
            hcore_A_in_B = hcore_A_in_B[np.ix_(active_aos, active_aos)]
            pure_dA = 2 * purify(tden_mat_A / 2, tovlp)

            # make initial guess
            tveff = tinit_mf.get_veff(dm=pure_dA)
            h_eff = hcore_A_in_B + tveff
            e_mos, c_mos = tinit_mf.eig(h_eff, tovlp)
            occ_mos = tinit_mf.get_occ(e_mos, c_mos)
            guess_d = make_dm(c_mos[:,occ_mos>0], occ_mos[occ_mos>0])

            # truncated initial method (self embedded)
            tinit_mf.get_hcore = lambda *args: hcore_A_in_B
            tinit_mf.kernel(guess_d)

            # Overwrite previous values
            den_mat_A = tinit_mf.make_rdm1()
            v_A = tinit_mf.get_veff(dm=den_mat_A)
            energy_elec_A, energy_coulomb_A = tinit_mf.energy_elec(dm=den_mat_A, vhf=v_A, h1e=hcore)
            print(energy_elec_A)
        
    # make embedding mean field object
    if (xc is None) or (xc.lower()=='rhf'):
        mf_embed = scf.RHF(mol)
    else:
        if corr_meth:
            print("***Attempting to perform correlated WF method on top of DFT***")
        mf_embed = dft.RKS(mol)
        mf_embed.xc = xc
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
    if trunc_lambda:
        energy_projector = mu * np.dot(den_mat_A_inB - den_mat_A, projector).trace()
    else:
        energy_projector = mu * np.dot(den_mat_A_inB, projector).trace()

    # first-order correction
    first_order_correction = np.dot(den_mat_A_inB - den_mat_A, v_embed).trace()

    # recombined energy with embedded part
    embed_energy = energy_elec_A_inB - energy_elec_A + energy_elec_AB
    
    rv = (embed_energy + energy_nuc, first_order_correction, energy_projector, )
    
    # correlated WF method
    if corr_meth:
        if corr_meth.lower()=='mp2':
            embed_corr = mp.MP2(mf_embed)
            embed_corr.kernel()
            rv = rv + (embed_corr.e_corr,)
            
        elif corr_meth.lower() in ['ccsd', 'ccsd(t)']:
            embed_corr = cc.CCSD(mf_embed)
            embed_corr.kernel()
            rv = rv + (embed_corr.emp2,)
            rv = rv + (embed_corr.e_corr - embed_corr.emp2,)
            
        if corr_meth.lower()=='ccsd(t)':
            embed_corr_et = embed_corr.ccsd_t()
            rv = rv + (embed_corr_et,)
        
    return rv


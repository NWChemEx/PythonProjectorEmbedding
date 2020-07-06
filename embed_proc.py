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
                        debug_dist=False):
    # Manby-like embedding procedure. Kinda rigid at the moment.
    mol = pyscf_mol.copy()
    tot_energy_AB = init_mf.e_tot
    
    # get intermediates
    ovlp = init_mf.get_ovlp()
    hcore = init_mf.get_hcore()
    mo_occ = init_mf.mo_occ
    C_occ = init_mf.mo_coeff[:, mo_occ>0]
    if localize:
        C_occ = lo.PM(mol, C_occ).kernel()
        
    # split mos
    init_mf.charge_threshold = charge_threshold
    C_occ_A, C_occ_B = distribute_mos(mol, init_mf,
                                      active_atoms=active_atoms,
                                      C_occ=C_occ,
                                      debug=debug_dist)

    # make full and subsystem densities
    den_mat_AB = make_dm(C_occ, mo_occ[mo_occ>0]) 
    den_mat_A = make_dm(C_occ_A, mo_occ[:C_occ_A.shape[1]])
    den_mat_B = den_mat_AB - den_mat_A

    # get full and subsystem potential terms and new make potentials
    v_AB = init_mf.get_veff(mol=mol, dm=den_mat_AB)
    v_A = init_mf.get_veff(mol=mol, dm=den_mat_A)
    
    v_embed = v_AB - v_A
    projector = np.dot(np.dot(ovlp, den_mat_B), ovlp)
    
    # get electronic energy for A
    energy_A, _ = init_mf.energy_elec(dm=den_mat_A, vhf=v_A, h1e=hcore + v_embed)
    
    # build embedding potential
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
            
        print("Active AOs:", len(active_aos), "/", mol.nao)
    
        if len(active_aos) == mol.nao:
            print("No AOs Truncated")
            trunc_lambda = None
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
            hcore_A_in_B = hcore_A_in_B[np.ix_(active_aos, active_aos)]
            tden_mat_A = den_mat_A[np.ix_(active_aos, active_aos)]
            tovlp = ovlp[np.ix_(active_aos, active_aos)]
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
            energy_A, _ = tinit_mf.energy_elec(dm=den_mat_A, vhf=v_A, h1e=hcore_A_in_B)
        
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
    den_mat_A_inB = mf_embed.make_rdm1()

    # make potential for embedded result
    v_A_inB = mf_embed.get_veff(mol, den_mat_A_inB)
    
    # get electronic energy for embedded part
    energy_A_inB, _ = mf_embed.energy_elec(dm=den_mat_A_inB, vhf=v_A_inB)

    # recombined energy with embedded part
    embed_energy = energy_A_inB - energy_A + tot_energy_AB
    
    rv = (embed_energy, )
    
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


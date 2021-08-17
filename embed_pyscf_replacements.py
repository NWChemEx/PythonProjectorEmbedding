import numpy as np
from pyscf.lib import logger
from pyscf.lib import NPArrayWithTag

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None or (isinstance(vhf, NPArrayWithTag) and getattr(vhf, 'ecoul', None) is None):
        vhf = mf.get_veff(mf.mol, dm)

    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    if isinstance(h1e, np.ndarray) and h1e.ndim == 2:
        h1e = np.array((h1e, h1e))
    e1 = np.einsum('aij,aji->', h1e, dm)

    if hasattr(vhf, 'ecoul'):
        e2 = vhf.ecoul + vhf.exc
        
        mf.scf_summary['e1'] = e1.real
        mf.scf_summary['coul'] = vhf.ecoul.real
        mf.scf_summary['exc'] = vhf.exc.real
        logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
    else:
        if isinstance(vhf, np.ndarray) and vhf.ndim == 2:
            vhf = np.array((vhf, vhf))
        e2 = np.einsum('aij,aji->', vhf, dm) * 0.5

        mf.scf_summary['e1'] = e1.real
        mf.scf_summary['e2'] = e2.real
        logger.debug(mf, 'E1 = %s  Ecoul = %s', e1, e2.real)

    return (e1+e2).real, e2


import numpy as np
from pyscf.lib import logger, NPArrayWithTag


def _check_dims_then_einsum(mat, dm):
    if mat.ndim == 2 and dm.ndim == 2:
        return np.einsum("ij,ji->", mat, dm)
    elif mat.ndim == 2 and dm.ndim == 3:
        return np.einsum("ij,aji->", mat, dm)
    elif mat.ndim == 3 and dm.ndim == 2:
        return np.einsum("aij,ji->", mat, dm * 0.5)
    elif mat.ndim == 3 and dm.ndim == 3:
        return np.einsum("aij,aji->", mat, dm)
    else:
        raise RuntimeError(f"Unexpected dims: {mat.ndim}, {dm.ndim}")


def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None:
        dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None or (
        isinstance(vhf, NPArrayWithTag) and getattr(vhf, "ecoul", None) is None
    ):
        vhf = mf.get_veff(mf.mol, dm)

    e1 = _check_dims_then_einsum(h1e, dm)

    if hasattr(vhf, "ecoul"):
        e2 = vhf.ecoul + vhf.exc

        mf.scf_summary["e1"] = e1.real
        mf.scf_summary["coul"] = vhf.ecoul.real
        mf.scf_summary["exc"] = vhf.exc.real
        logger.debug(
            mf, "PATCHED E1 = %s  Ecoul = %s  Exc = %s", e1, vhf.ecoul, vhf.exc
        )
    else:
        e2 = _check_dims_then_einsum(vhf, dm) * 0.5

        mf.scf_summary["e1"] = e1.real
        mf.scf_summary["e2"] = e2.real
        logger.debug(mf, "PATCHED E1 = %s  Ecoul = %s", e1, e2.real)

    return (e1 + e2).real, e2

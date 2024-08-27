# from IPython import get_ipython
from tqdm import tqdm as tqdm_base


def tqdm_clear(*args, **kwargs):
    getattr(tqdm_base, '_instances', {}).clear()


def tqdm(*args, **kwargs):
    """Decorator of tqdm, to avoid some errors if tqdm 
    terminated unexpectedly

    Returns
    -------
    an decorated `tqdm` class
    """

    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

base_str = tqdm.__doc__ if tqdm.__doc__ else ""
init_str = tqdm_base.__init__.__doc__ if tqdm_base.__init__.__doc__ else ""
tqdm.__doc__ = base_str + init_str

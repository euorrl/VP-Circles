import numpy as np


def vp_circle(pop: np.ndarray, f: float) -> dict:
    """
    Compute the Valeriepieris Circle (VP circle) on a 2D population raster.

    Parameters
    ----------
    pop : np.ndarray, shape (H, W)
        2D population raster (non-negative). pop[i, j] is the population (or weight)
        at row i, column j.
    f : float
        Target population fraction in (0, 1]. The circle must cover at least
        f * pop.sum() of the total population.

    Returns
    -------
    result : dict
        A dictionary with:
        - "radius" : int
            The minimum integer radius (in pixel/grid units) such that there exists
            at least one center whose disk contains >= target population.
        - "centers" : list[tuple[int, int]]
            All centers (row, col) that achieve the target population at this radius.
            There can be multiple optimal centers.
        - "covered" : float
            The maximum covered population among all centers at the optimal radius.
        - "target" : float
            The target population threshold, i.e., f * total_population.
    """
    # -------- Input validation --------
    pop = np.asarray(pop)

    if pop.ndim != 2:
        raise ValueError(f"`pop` must be a 2D array, got shape {pop.shape}")

    if np.any(pop < 0):
        raise ValueError("`pop` must be non-negative (found values < 0).")
    
    if not (0 < f <= 1):
        raise ValueError("f must be in (0, 1]")

    pop = pop.astype(float, copy=False)

    total = float(np.sum(pop))
    target = f * total

    # Edge case: empty population raster
    if total == 0.0:
        return {"radius": 0, "centers": [], "covered": 0.0, "target": 0.0}

    # TODO (next step): implement VP circle search for minimal radius and centers.
    # For now, raise an error to indicate the core algorithm is not implemented yet.
    raise NotImplementedError("vp_circle core computation is not implemented yet.")

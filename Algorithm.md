# Algorithm Overview

## Problem definition

Consider a 2D spatial raster in which each cell represents a non-negative population (or mass) value.  
Let the total population be the sum of all cell values, and let `f ∈ (0, 1)` be a target fraction.

The Valeriepieris Circle (VP circle) problem is to find the **smallest possible radius** `r*` such that there exists at least one circle of radius `r*` whose interior contains at least a fraction `f` of the total population.

Because the data are defined on a discrete grid, the circle center is restricted to grid cell centers, and the population inside a circle is defined as the sum of values over all cells whose centers lie within the circle.

In general, there may be multiple grid locations that achieve the same minimal radius.  
The output of the VP circle problem therefore consists of:
- the optimal radius `r*`, and
- a set of grid locations (centers) for which a circle of radius `r*` contains at least a fraction `f` of the total population.

## High-level algorithm structure

Directly solving the VP circle problem by evaluating all possible radii and all possible centers is computationally infeasible for large spatial rasters.  
The main challenge lies in the size of the search space: a large number of potential center locations combined with a continuous range of possible radii.

Although the feasibility of a radius is monotonic (larger radii can only increase the amount of population covered), the optimal radius cannot be obtained analytically.  
An efficient solution therefore requires a strategy that rapidly narrows both the radius range and the set of candidate centers, while minimizing expensive population aggregation operations.

To address this, the algorithm adopts a **coarse-to-fine search strategy**, progressively refining the solution in three stages, each serving a distinct role:

1. **Stage I: Exponential halving**  
   Starting from a large initial radius that is guaranteed to be feasible, the radius is repeatedly reduced by a factor of two.  
   This stage quickly identifies a coarse interval that brackets the optimal radius.

2. **Stage II: Binary search with pruning**  
   Within the bracketing interval, a binary search is performed to further narrow the radius range.  
   At the same time, grid locations that cannot possibly be optimal centers are discarded, progressively shrinking the candidate set.

3. **Stage III: Exact discrete refinement**  
   Once the radius interval is sufficiently small or the number of candidate centers is limited, an exact search is performed.  
   This final stage exploits the discrete nature of the grid to determine the minimal achievable radius precisely.

Together, these three stages allow the algorithm to efficiently locate the exact VP circle solution without exhaustively evaluating all possible center–radius combinations.


## Stage I & II: Continuous-radius search

Stages I and II jointly search over a continuous radius while progressively reducing the set of candidate centers.  
Both stages rely on the same feasibility test and differ mainly in how the radius is updated and how candidate centers are pruned.

At any point during these stages, the algorithm maintains the following state:
- a radius value or a radius interval,
- a boolean mask indicating feasible center locations at the current radius, and
- the number of feasible centers.

### Feasibility evaluation and population aggregation

For a given radius `r`, a grid location is considered **feasible** if a circle of radius `r` centered at that location contains at least a fraction `f` of the total population.

Evaluating this condition independently for every grid cell would be computationally prohibitive.  
To avoid this, the algorithm constructs a **disk kernel** corresponding to radius `r`, whose support consists of all grid offsets lying within the circle.

Convolving the population raster with this disk kernel yields, in a single operation, the population sum within radius `r` for **all possible center locations**.  
This population aggregation is implemented efficiently using FFT-based convolution.

The feasibility evaluation therefore produces:
- a raster of population sums for all centers, and
- a boolean feasibility mask obtained by thresholding these sums against `f · (total population)`.

### Stage I: Exponential halving

Stage I starts from a sufficiently large initial radius, chosen for example as half of the grid diagonal length.  
The purpose of this choice is **not** to make all center locations feasible, but to guarantee that **at least one feasible center exists**: a circle centered near the grid center with this radius covers the entire raster, so the enclosed population equals the total population and satisfies any `f ∈ (0, 1)`.

From this initial value, the radius is repeatedly reduced by a factor of two.  
After each halving step, feasibility is re-evaluated using the convolution-based aggregation described above.

At each step, one of the following outcomes occurs:
- **The feasible-center set remains non-empty**: at least one center is still feasible, indicating that the radius may be reduced further and halving continues.
- **The feasible-center set becomes empty**: no center is feasible at the current radius, so the optimal radius is bracketed between the current radius and the previous feasible one, and the algorithm transitions to Stage II.
- **The number of feasible centers becomes sufficiently small**: further continuous-radius refinement offers diminishing returns, and the algorithm transitions directly to Stage III.

The third case is motivated by the cost structure of the algorithm: Stage III performs exact refinement **per candidate center**, so reducing the candidate set early makes the final stage computationally tractable.

If the second case occurs, Stage I outputs:
- a lower and upper bound on the optimal radius, and
- the feasibility mask at the last feasible radius, which serves as the initial candidate-center set for Stage II.

### Stage II: Binary search with pruning

Stage II refines the bracketing interval obtained from Stage I using binary search.  
At each iteration, feasibility is evaluated at the midpoint radius of the current interval.

If at least one feasible center exists at the midpoint radius, the upper bound of the interval is updated, and the corresponding feasibility mask becomes the new candidate set.  
If no feasible center exists, the lower bound is updated, while the candidate set from the most recent feasible radius is retained.

This procedure simultaneously:
- narrows the radius interval, and
- prunes the candidate centers, since only locations that remain feasible at smaller radii are preserved.

Stage II terminates when **either**:
- the radius interval becomes sufficiently small, **or**
- the number of candidate centers falls below a predefined threshold.

Upon termination, Stage II passes to Stage III:
- a reduced set of candidate centers, and
- a (possibly tight) upper bound on the optimal radius.

Either of these improvements benefits the exact refinement stage: fewer candidate centers reduce the number of per-center refinements, while a tighter radius bound limits the range of discrete radius levels that must be considered.


## Stage III: Exact discrete refinement

Stages I and II reduce the VP circle problem to a small set of candidate centers and provide a safe upper bound on the optimal radius.  
Stage III uses this information to compute the **exact minimal achievable radius** by exploiting the discrete geometry of the grid.

### Discrete radius levels induced by the grid

On a regular integer grid, the set of cells included in a circular neighborhood changes only when the squared distance

`d^2 = (Δx)^2 + (Δy)^2`

crosses a new value realizable by integer offsets. As a consequence, the population contained within a circle does not vary continuously with the radius, but only at a finite set of **discrete squared-distance levels**.

Let

`s_max = floor(r_upper^2)`

where `r_upper` denotes the radius upper bound obtained from the continuous search stages. Stage III enumerates all integer offsets `(Δy, Δx)` satisfying

`(Δx)^2 + (Δy)^2 ≤ s_max`

and computes the corresponding squared distances `d^2`. These squared-distance values form a finite, strictly increasing sequence of discrete levels, each of which induces a candidate radius

`r = sqrt(d^2)`.

Because the set of grid cells included in a circular neighborhood changes only at these squared-distance levels, it is sufficient to consider radii induced by the enumerated distances.

### Per-center exact aggregation

For each candidate center, Stage III groups population contributions according to their squared-distance levels. Each grid cell contributes its population mass to the level associated with its squared offset distance from the center.

This grouping transforms the original two-dimensional aggregation problem into a one-dimensional problem over the discrete level sequence. By computing a prefix sum over the level-indexed population contributions, the algorithm obtains the cumulative population enclosed as the radius increases.

The smallest level at which the cumulative population reaches the target fraction `f` determines the exact minimal squared radius `s*` for that center, and hence the exact minimal radius

`r* = sqrt(s*)`.

### Selection of the global optimum

After exact refinement has been performed for all candidate centers, the global optimal radius is obtained as:

`r* = min_over_centers(r*_center)`.

All centers achieving this minimal radius are retained as optimal VP circle centers.


## Computational complexity and scalability

This section analyzes the computational complexity of different approaches to computing a VP circle and explains how the stage-wise strategy adopted in this library improves scalability.

Throughout this section, we use the following notation:

- `N = H × W` denotes the total number of grid cells;
- `R_max` denotes the maximum relevant radius scale;
- `f` is the target population fraction, and `t = f · total` is the target population;
- `M` denotes the number of candidate centers entering Stage III;
- `r*` denotes the optimal VP radius.

---

### Radius scale and raster size

For an `H × W` regular grid, the maximum relevant radius is typically on the order of half the grid diagonal:

    R_max ≈ 0.5 · sqrt(H^2 + W^2)

When `H` and `W` are of the same order, this implies:

    R_max = O(√N)
    R_max^2 = O(N)
    R_max^3 = O(N^(3/2))

This relationship allows all complexities to be restated purely in terms of `N`, making differences in scalability explicit.

---

### 1) Brute-force approaches

#### (A) Per-center, per-radius direct summation

The most straightforward method enumerates every grid cell as a candidate center and explicitly sums the population within the disk for all candidate radii.

- A single feasibility check at radius `r` costs O(r²), proportional to the number of cells inside the disk.
- For one center, enumerating radii `r = 1..R_max` yields:

    Σ r² = O(R_max³)

- Repeating this for all `N` centers gives:

    O(N · R_max³)

Substituting `R_max = O(√N)` results in:

    O(N^(5/2))

This complexity grows too rapidly to be practical for large rasters.

---

#### (B) Per-center binary search with direct summation

A common refinement is to perform a binary search over the radius for each center.

- Each feasibility test still costs O(r²);
- Binary search requires O(log R_max) tests.

Thus, for one center:

    O(R_max² · log R_max)

and for all centers:

    O(N · R_max² · log R_max)

Using `R_max² = O(N)`, this becomes:

    O(N² · log N)

This improves over full enumeration but remains close to quadratic complexity.

---

### 2) Arthur (2024): moving centers with incremental updates

The algorithm described by Arthur (2024) does not perform a full binary search independently at every grid location. Instead, it follows this strategy:

1. For an initial center, perform a binary search to find the minimal feasible radius `minR`;
2. Move the center to a neighboring grid cell and evaluate feasibility at the same radius `minR`;
3. Only if the population at the new center satisfies `P ≥ t` is a new binary search performed to reduce `minR`;
4. Otherwise, the algorithm moves on to the next center.

The key algorithmic optimization is that when the center shifts by one grid cell, population changes occur only near the disk boundary. Therefore, a single shift update costs only a boundary-level amount:

    O(r)

rather than O(r²).

A conservative upper bound on the algorithm’s complexity can thus be written as:

    O(N · R_max · log R_max)

Substituting `R_max = O(√N)` yields:

    O(N^(3/2) · log N)

Compared to brute-force methods, this reduces the exponent by exploiting incremental updates, but the complexity still grows faster than linear in `N`.

---

### 3) Stage-wise algorithm used in this library

This library adopts a fundamentally different computational structure by decomposing the search into three stages:

- Stages I and II compute global feasibility maps via convolution, eliminating large numbers of centers at once;
- Stage III performs exact discrete refinement only on a small set of remaining candidate centers.

This design shifts most of the computational effort away from per-center processing.

---

#### Stages I & II: global feasibility via FFT convolution

For a fixed radius `r`, a feasibility test consists of:

- constructing a disk kernel;
- convolving the population raster with the kernel using FFT;
- thresholding the result to obtain the feasible-center mask.

A single FFT-based convolution costs:

    O(N · log N)

In this library, Stages I (exponential halving) and II (binary search) stop once the radius interval reaches grid resolution (`eps = 1`). The total number of feasibility probes therefore satisfies:

    O(log R_max)

Hence, the combined cost of Stages I and II is:

    O(N · log N · log R_max)

Using `log R_max = O(log N)`, this becomes:

    O(N · (log N)²)

which is close to linear in practice.

---

#### Stage III: exact discrete refinement on a small candidate set

Stage III operates on:

- `M` candidate centers;
- a squared-radius upper bound `S = floor(r_upper²)`.

On an integer grid, the set of cells inside a disk changes only at squared-distance levels:

    d² = (Δx)² + (Δy)²

Enumerating all integer offsets satisfying `(Δx)² + (Δy)² ≤ S` yields O(S) such offsets.

Thus, the cost of Stage III is:

    O(M · S)

In the worst case, `r_upper ≈ R_max` implies `S = O(N)`. However, the purpose of Stages I and II is precisely to reduce both `M` and `r_upper`, making this term much smaller in practice.

---

### Overall comparison and conclusion

Combining all stages, the overall complexity of the proposed method can be summarized as:

    O(N · (log N)² + M · S)

For comparison:

- Brute-force enumeration: O(N^(5/2))
- Per-center binary search: O(N² · log N)
- Arthur (2024): O(N^(3/2) · log N)
- This library: O(N · (log N)² + M · S)

By replacing repeated per-center computations with a small number of global feasibility probes and restricting exact calculations to a heavily pruned candidate set, the proposed approach achieves substantially improved scalability with respect to raster size.

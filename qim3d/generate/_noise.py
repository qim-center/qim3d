"""3D gradient noise for synthetic volumes, using only NumPy.

Voxel coordinates are processed in batches so temporary coordinate, gradient,
and interpolation arrays cover only a batch, rather than the entire volume.

Algorithm references (this is a NumPy implementation, not a verbatim code port):

* Ken Perlin, "Improved Noise" reference implementation (2002):
  https://mrl.cs.nyu.edu/~perlin/noise/
  The cube-corner contributions and quintic interpolation follow this method;
  our seeded permutation and gradient-index mapping differ from the reference.
* Stefan Gustavson, "Simplex noise demystified" (2005):
  https://itn-web.it.liu.se/~stegu76/simplexnoise/simplexnoise.pdf
  Describes the gradient directions, 3D skew/unskew transform, tetrahedron
  selection, and distance-weighted gradient contributions.
* weswigham/simplex, ``Noise3D``:
  https://github.com/weswigham/simplex/blob/master/c/src/simplex.c
  Uses the same 0.5 support constant, fourth-power falloff, and amplitude factor
  32 as our Simplex implementation. We use a seeded permutation table instead
  of its fixed table, and sort coordinates instead of branching to select corners.
* Gustavson and McEwan, "Tiling Simplex Noise and Flow Noise in Two and Three
  Dimensions" (2022), section 7:
  https://www.jcgt.org/published/0011/01/02/paper-lowres.pdf
  Explains why reducing the support constant from 0.6 to 0.5 removes boundary
  discontinuities. We retain the fourth power rather than their cubic falloff.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Twelve equally long directions pointing to the edge midpoints of a cube.
_GRADIENTS = np.array(
    [
        (1, 1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (-1, 0, 1),
        (1, 0, -1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, -1, 1),
        (0, 1, -1),
        (0, -1, -1),
    ],
    dtype=np.float64,
)


def _fade(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quintic blend with zero first and second derivatives at 0 and 1."""
    return t**3 * (t * (t * 6 - 15) + 10)


def _gradient_indices(
    corner_x: NDArray[np.int64],
    corner_y: NDArray[np.int64],
    corner_z: NDArray[np.int64],
    permutation: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Hash wrapped integer corner coordinates into the gradient table.

    Coordinates must be wrapped modulo 256 before use (a corner offset of 1
    may then be added). The duplicated permutation handles lookup overflow.
    """
    return permutation[corner_x + permutation[corner_y + permutation[corner_z]]] % len(
        _GRADIENTS
    )


def _gradient_dot(
    permutation: NDArray[np.int64],
    corner_positions: NDArray[np.int64],
    displacements: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate each corner's linear ramp: gradient dot displacement."""
    # Bitwise AND with 255 is modulo 256 for these integer coordinates.
    corner_x, corner_y, corner_z = (corner_positions & 255).T
    indices = _gradient_indices(corner_x, corner_y, corner_z, permutation)
    return np.einsum("ij,ij->i", _GRADIENTS[indices], displacements)


def _perlin(
    points: NDArray[np.float64], permutation: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Blend the eight surrounding cube corners' gradient contributions.

    Each row of ``points`` is evaluated independently. For each corner, compute
    gradient dot displacement, multiply by the three axis blend weights, and
    sum. See Perlin's reference linked in the module docstring.
    """
    cell_origin = np.floor(points).astype(np.int64)
    local_position = points - cell_origin
    blend_weights = _fade(local_position)
    local_x, local_y, local_z = local_position.T
    fade_x, fade_y, fade_z = blend_weights.T

    # Wrap grid coordinates modulo 256 only for the permutation-table lookup.
    cell_x, cell_y, cell_z = (cell_origin & 255).T
    noise_values = np.zeros(len(points))
    for corner_x in (0, 1):
        weight_x = fade_x if corner_x else 1 - fade_x
        for corner_y in (0, 1):
            weight_xy = weight_x * (fade_y if corner_y else 1 - fade_y)
            for corner_z in (0, 1):
                indices = _gradient_indices(
                    cell_x + corner_x,
                    cell_y + corner_y,
                    cell_z + corner_z,
                    permutation,
                )
                gradients = _GRADIENTS[indices]

                # Displacement from this corner to each sample in the cube.
                dx = local_x - corner_x
                dy = local_y - corner_y
                dz = local_z - corner_z
                corner_contribution = (
                    gradients[:, 0] * dx + gradients[:, 1] * dy + gradients[:, 2] * dz
                )
                corner_weight = weight_xy * (fade_z if corner_z else 1 - fade_z)
                noise_values += corner_weight * corner_contribution
    return noise_values


def _simplex(
    points: NDArray[np.float64], permutation: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Sum four tetrahedron corners' distance-weighted gradient contributions.

    Each row of ``points`` is evaluated independently. This follows the 3D
    construction described by Gustavson, with the falloff
    ``max(0.5 - distance_squared, 0)**4`` and amplitude factor 32 used by
    weswigham/simplex's Noise3D. See the module references.
    """
    # Skew to a grid where floor() locates a cube split into six tetrahedra.
    # The 3D skew factor F3 is 1/3; its inverse uses G3 = 1/6.
    skew_amount = points.sum(axis=1, keepdims=True) / 3
    cell_origin = np.floor(points + skew_amount).astype(np.int64)
    unskew_amount = cell_origin.sum(axis=1, keepdims=True) / 6
    local_position = points - cell_origin + unskew_amount

    # Descending coordinate order selects the tetrahedron. For x >= y >= z,
    # its corner offsets are 000, 100, 110, 111. Stable sorting resolves ties.
    axis_order = np.argsort(-local_position, axis=1, kind="stable")
    first_corner = np.zeros_like(cell_origin)
    np.put_along_axis(first_corner, axis_order[:, :1], 1, axis=1)
    second_corner = first_corner.copy()
    np.put_along_axis(second_corner, axis_order[:, 1:2], 1, axis=1)
    corner_offsets = (
        np.zeros_like(cell_origin),
        first_corner,
        second_corner,
        np.ones_like(cell_origin),
    )

    noise_values = np.zeros(len(points))
    for corner_number, corner_offset in enumerate(corner_offsets):
        # The offsets contain 0, 1, 2, then 3 ones. Unskewing therefore adds
        # corner_number / 6 to each component of the sample's displacement.
        displacement = local_position - corner_offset + corner_number / 6
        distance_squared = np.einsum("ij,ij->i", displacement, displacement)
        # Limit support so departing corners contribute zero at cell boundaries.
        attenuation = np.maximum(0.5 - distance_squared, 0)
        corner_contribution = _gradient_dot(
            permutation, cell_origin + corner_offset, displacement
        )
        noise_values += attenuation**4 * corner_contribution
    return 32 * noise_values


def _noise_field(
    shape: tuple[int, int, int],
    scale: float,
    noise_type: Literal["perlin", "simplex"],
    seed: int | None,
    *,
    chunk_size: int = 65536,
) -> NDArray[np.float64]:
    """Sample one seeded 3D field in batches, retaining global voxel coordinates.

    ``scale`` multiplies voxel coordinates to set the spatial frequency. Smaller
    values give broader features. Only this scale is sampled; layers at different
    scales are not combined.

    Points are evaluated independently using a shared permutation table.
    The default batch size balances per-call overhead and temporary memory use.
    """
    rng = np.random.default_rng(seed)
    # Duplicate the shuffled table so nested lookups can overflow by 256.
    permutation = np.tile(rng.permutation(256), 2)
    evaluate = {"perlin": _perlin, "simplex": _simplex}[noise_type]
    result = np.empty(shape, dtype=np.float64)
    flat = result.reshape(-1)
    for start in range(0, flat.size, chunk_size):
        stop = min(start + chunk_size, flat.size)
        points = (
            np.column_stack(np.unravel_index(np.arange(start, stop), shape)) * scale
        )
        flat[start:stop] = evaluate(points, permutation)
    return result

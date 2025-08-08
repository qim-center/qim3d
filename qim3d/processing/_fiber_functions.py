import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from _fiberBundleClass import FiberBundle
import qim3d

## ------------------------------------------------------------------ G E N E R A T O R S ------------------------------------------------------------------ ##
def generate_random_lines(N, steps=128, step_size=1.0, initial_positions=None):
    lines = []
    
    if initial_positions is None:
        # Random initial positions if not provided
        initial_positions = [np.array([np.random.uniform(0, steps),
                                       np.random.uniform(0, steps),
                                       0.0]) for _ in range(N)]
    
    for start in initial_positions:
            line = [start.copy()]
            current = start.copy()
            for _ in range(1, steps):
                dx, dy = np.random.uniform(-step_size, step_size, size=2)
                dz = 1.0 # Constant step in z
                current = current + np.array([dx, dy, dz])
                line.append(current.copy())
            lines.append(np.array(line))
    
    return lines

def generate_slicewise_lines(volume, axis=2):
    """
    Compute slice-wise centroids of connected components in a 3D binary volume.
    
    Parameters:
    - volume: 3D numpy array (binary segmentation)
    - axis: Axis along which to slice (0, 1, or 2)
    
    Returns:
    List[np.ndarray]: List of arrays, each containing slice-wise centroid coordinates [x, y, z] for each component.
    """
    
    # Label connected components in 3D
    labeled_volume, num_features = label(volume)
    
    # Prepare output list
    fibers = []

    # Iterate over each connected component
    for label_id in range(1, num_features + 1):
        component_mask = (labeled_volume == label_id)
        line = []
        
        # Iterate over slices along the specified axis
        for slice_index in range(volume.shape[axis]):
            # Create slicing object
            slicer = [slice(None)] * 3
            slicer[axis] = slice_index
            slice_mask = component_mask[tuple(slicer)]
            
            if np.any(slice_mask):
                # Compute centroid in the slice
                centroid = center_of_mass(slice_mask)
                
                # Adjust centroid to full volume coordinates
                coords = list(centroid)
                coords.insert(axis, slice_index)
                line.append(coords)
                
        # Convert to numpy array and append to result list
        fibers.append(np.array(line))
        
    return fibers


## ------------------------------------------------------------------ O P E R A T I O N S ------------------------------------------------------------------ ##

## --------- Smoothing operations -------------- ##
def smooth_savgol(line, filter_size=3, poly_order=2):
    dim = line.shape[1]
    window_length = int(filter_size)
    
    return np.array([savgol_filter(line[:,i], window_length, poly_order) for i in range(dim)]).T

def smooth_moving(line, filter_size=3):
    dim = line.shape[1]
    window_length = int(filter_size)
    
    return np.array([np.convolve(line[:,i], np.ones(window_length)/window_length, mode='valid') for i in range(dim)]).T

def smooth_fibers(fibers, method='savgol', filter_size=3):
    #nFib = len(fibers)
    fibers_smooth = []
    if method == 'savgol':
        for fib in fibers:
            fibers_smooth.append( smooth_savgol(fib, filter_size=filter_size, poly_order=2) )
    elif method == 'moving':
        for fib in fibers:
            fibers_smooth.append( smooth_moving(fib, filter_size=filter_size) )
    else:
        raise ValueError("Unknown method: {}".format(method))

    return fibers_smooth


## --------- Resampling operations -------------- ##

def resample_equidistant(line, sectioning_distance=1):
    dim = line.shape[1]

    # Calculate accumulated length along line
    accum_dist = np.concatenate(([0], np.cumsum(np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1)))))
    
    # Query points
    qp = np.arange(0, accum_dist[-1], sectioning_distance)
    
    # Interpolate new line
    return np.array([np.interp(qp, accum_dist, line[:,i]) for i in range(dim)]).T

def resample_fibers(fibers, sectioning_distance=1):
    fibers_resampled = []
    for fib in fibers:
        fibers_resampled.append( resample_equidistant(fib, sectioning_distance) )

    return fibers_resampled
   
## --------- Other operations -------------- ##

# Skew-symmetric cross-product matrix
def ssc(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# Rotation matrix from vector A to B
def rotation_matrix(A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    v = np.cross(A, B)
    c = np.dot(A, B)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    vx = ssc(v)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

def compute_naive_directions(line):

    # Local direction vector (normalized)
    dir_vec = np.diff(line, axis=0, append=line[-1:])
    d_vecs = dir_vec / np.linalg.norm(dir_vec, axis=1, keepdims=True)

    return d_vecs

def compute_direction_vectors(line, epsilon=1e-10, illdefined_value=0.0):
    """
    Compute direction, radial, and normal vectors, and local curvature radius
    from a resampled centerline using vectorized operations.
    Note that the first and last point of the line is not fully defined.

    Parameters:
    - line (ndarray): [n x 3] array of resampled centerline points
    - epsilon (float): tolerance level, in case of almost co-linear points
    - illdefined_value (float or np.nan): the value to use for ill-defined cases

    Returns:
    - d_vec (ndarray): [n x 3] direction unit vectors
    - r_vec (ndarray): [n x 3] radial unit vectors
    - n_vec (ndarray): [n x 3] normal unit vectors
    - c_rad (ndarray): [n] curvature radii
    """
    # Number of points and selected pointsets
    n = line.shape[0]
    P1 = line[0:-2,:]
    P2 = line[1:-1,:]
    P3 = line[2:,:]

    v1 = P2 - P1 # In-going vectors
    v2 = P3 - P2 # Out-going vectors
    
    cpp = np.cross(v1, v2) # pairwise cross products
    cpp_norm = np.linalg.norm(cpp, axis=1) # normalized

    # Potential degenerate case: three co-linear points do not span a plane!
    # Avoid division by zero by setting small epsilon for degenerate cases
    cpp_norm_safe = np.where(cpp_norm < epsilon, epsilon, cpp_norm)

    # Normal vectors
    n_vec = cpp / cpp_norm_safe[:, np.newaxis]

    # Circumscribed circle radius
    # The maths: https://en.wikipedia.org/wiki/Circumscribed_circle
    a = np.linalg.norm(P1 - P2, axis=1)
    b = np.linalg.norm(P2 - P3, axis=1)
    c = np.linalg.norm(P3 - P1, axis=1)
    c_rad = a * b * c / (2 * cpp_norm_safe)

    # Circumcenter coefficients
    # Using einsum for pairwise dot-products
    coeff1 = b**2 * np.einsum('ij,ij->i', P1 - P2, P1 - P3) / (2 * cpp_norm_safe**2)
    coeff2 = c**2 * np.einsum('ij,ij->i', P2 - P1, P2 - P3) / (2 * cpp_norm_safe**2)
    coeff3 = a**2 * np.einsum('ij,ij->i', P3 - P1, P3 - P2) / (2 * cpp_norm_safe**2)
    
    # Circum-center and radial vector
    PC = coeff1[:, np.newaxis] * P1 + coeff2[:, np.newaxis] * P2 + coeff3[:, np.newaxis] * P3
    r_vec = PC - P2
    r_vec /= np.linalg.norm(r_vec, axis=1)[:, np.newaxis]

    # Direction vectors
    d_vec = np.cross(r_vec, n_vec)

    # Handle degenerate / ill-defined cases
    #degenerate_value = 0.0 #alternative "np.nan"
    degenerate_mask = cpp_norm < epsilon
    if np.any(degenerate_mask):
        avg_vec = (v1 + v2)[degenerate_mask]
        avg_vec /= np.linalg.norm(avg_vec, axis=1)[:, np.newaxis]
        d_vec[degenerate_mask] = avg_vec
        r_vec[degenerate_mask] = illdefined_value
        n_vec[degenerate_mask] = illdefined_value
        c_rad[degenerate_mask] = illdefined_value

    # Expand to match original length by copying first and last entries
    d_vec = np.vstack([d_vec[0,:], d_vec, d_vec[-1,:]])
    r_vec = np.vstack([r_vec[0,:], r_vec, r_vec[-1,:]])
    n_vec = np.vstack([n_vec[0,:], n_vec, n_vec[-1,:]])
    c_rad = np.concatenate([[c_rad[0]], c_rad, [c_rad[-1]]])

    return d_vec, r_vec, n_vec, c_rad


def generate_spokes(line, n_radial=6, radius=2.0, radial_resolution=1, direction_vectors=None):
    """
    Generate spoke points from a centerline
    
    Parameters:
    - line (ndarray): [nP x 3] array representing the centerline points
    - n_radial (int): Number of spokes (angular resolution)
    - radius (float): Radius of the cylinder spokes
    - radial_resolution (int): Number of samples along each spoke (1 = tip only)
    - direction_vectors (ndarray, optional): [nP x 3] direction unit vectors for each point of the centerline
    
    Returns:
    - spoke_points (ndarray): [nN x 3 x radial_resolution] mesh node coordinates
    """
    n_points = line.shape[0]
    
    # Generate generic x,y spokes
    # Could be optional function input to avoid recomputing
    angles = np.linspace(0, 2 * np.pi, n_radial + 1)[:-1]  # remove duplicate at 2π
    base_spokes = np.stack([np.sin(angles), np.cos(angles), np.zeros_like(angles)], axis=1)
    
    # Generate radial distances
    radial_distances = np.linspace(0, radius, radial_resolution+1)
    radial_distances = radial_distances[1:] #omit displacement of 0
    radial_distances.shape
    
    # Prepare direction vectors
    if direction_vectors is not None:
        if direction_vectors.shape[0] != n_points:
            raise ValueError("Input direction-vectors have wrong size")
        norms = np.linalg.norm(direction_vectors, axis=1)
        if np.any(np.abs(norms - 1) > 1e-3):
            raise ValueError("Input direction-vectors are not unit vectors")
    else:
        direction_vectors = compute_direction_vectors(line, epsilon=1e-10, illdefined_value=0.0)[0] # Use only the forward direction vectors here
    
    # Generate spoke points
    spoke_points = []
    for i in range(n_points):
        rot_m = rotation_matrix(direction_vectors[i, :], np.array([0, 0, 1]))
        rotated_spokes = base_spokes @ rot_m.T  # [n_radial x 3]
        samples = np.array([r * rotated_spokes for r in radial_distances])  # [radial_resolution x n_radial x 3]
        samples = samples.transpose(1, 2, 0)  # [n_radial x 3 x radial_resolution]
        samples += line[i, :].reshape(1, 3, 1)
        spoke_points.append(samples)
    spoke_points = np.concatenate(spoke_points, axis=0)  # [nN x 3 x radial_resolution]

    if spoke_points.shape[2] == 1:
        spoke_points = np.squeeze(spoke_points)
    
    return spoke_points

def regular_cylinder_faces(n_points, n_radial=6):
    """
    Generates the triangular faces for a cylinder-like object, characterized
    by a centerline of n_points and a chosen radial resolution (n_radial)
    
    Parameters:
    - n_points (int): Number of slices/layers/points in the centerline of the 'cylinder'-object
    - n_radial (int): The angular resolution
    
    Returns:
    - tri (ndarray): Array of triangle indices with shape (nF, 3)
    """
    tri = []
    
    for i in range(n_points - 1):  # Loop for each 'slice'
        for j in range(n_radial - 1):  # Loop along 'spokes'
            tri.append([(i)*n_radial + j, (i)*n_radial + j + 1, (i+1)*n_radial + j + 1])
            tri.append([(i+1)*n_radial + j + 1, (i+1)*n_radial + j, (i)*n_radial + j])
        
        # Wrapping start and end
        tri.append([(i+1)*n_radial - 1, i*n_radial, (i+1)*n_radial])
        tri.append([(i+1)*n_radial, (i+2)*n_radial - 1, (i+1)*n_radial - 1])
    
    return np.array(tri)

def regular_cylinder_meshing(fibers, n_radial=6, radius=2.0):

    # Compute vertices and faces of the mesh
    nodes = []
    tri = []
    total_count = 0
    for fib in fibers:

        # The vertices
        nP = np.shape(fib)[0] # "Length" of the fiber.
        verts = generate_spokes(fib, n_radial=n_radial, radius=radius, direction_vectors=None)
        nodes.append(verts)

        # The faces
        faces = regular_cylinder_faces(n_points=nP, n_radial=n_radial)
        tri.append( faces + total_count )

        # Update
        total_count = total_count + np.shape(verts)[0] # Update total vertex count
    
    # Finalize
    nodes = np.vstack(nodes)
    tri = np.vstack(tri)

    return nodes, tri

## ------------------------------------------------------------------ O P E R A T I O N S : V O L U M E T R I C ------------------------------------------------------------------ ##

def unfold_fiber(fiberBundle, fiber_id, radius_max, n_spokes, n_samples):

    # Assert that fiberBundle is a FiberClass

    # Load volume
    vol = qim3d.io.load(fiberBundle.volume_path, virtual_stack=True)
    vol = np.transpose(vol, [2, 1, 0])
    
    # Set-up interpolator
    x = np.arange(start=0, stop=fiberBundle.volume_dimensions[0], step=1)
    y = np.arange(start=0, stop=fiberBundle.volume_dimensions[1], step=1)
    z = np.arange(start=0, stop=fiberBundle.volume_dimensions[2], step=1)
    F_interp = RegularGridInterpolator((x,y,z), vol.astype('float16'), method='linear', bounds_error=False, fill_value=0.0)

    # Generate spokes
    line = fiberBundle.centrelines[fiber_id]
    n_line = line.shape[0]
    spoke_points = generate_spokes(line, n_radial=n_spokes, radius=radius_max, radial_resolution=n_samples)

    # Sample subvolume
    unfolded_volume = np.zeros((n_line, n_spokes, n_samples))
    for i in range(0, n_samples):
        val = F_interp(spoke_points[:,:,i]) # sample intensities at the i'th samples on all spokes
        unfolded_volume[:,:,i] = val.reshape(n_line, n_spokes)

    return unfolded_volume

    
## ------------------------------------------------------------------ V I S U A L I Z A T I O N S ------------------------------------------------------------------ ##

def spagetti_plot(axis, fibers, format_string='-', title_string='spagetti-plot'):

    # Assuming 3D case
    
    # Plot each line
    for line in fibers:
        axis.plot(line[:, 0], line[:, 1], line[:, 2], format_string)

    # Decorate
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    axis.set_title(title_string)
    axis.set_aspect('equal')

    return

def endpoints_plot(axis, fibers, start_color='r', end_color='b', title_string='endpoint-plot'):

     # Assuming 3D case
    
    nFib = len(fibers)
    endpoints_1 = np.zeros(shape=(nFib, 3))
    endpoints_2 = np.zeros(shape=(nFib, 3))
    
    # Plot each line and store endpoints
    for j, line in enumerate(fibers):
        axis.plot(line[:, 0], line[:, 1], line[:, 2], '-', color=[0.5, 0.5, 0.5, 0.3])
        endpoints_1[j,:] = line[0,:]
        endpoints_2[j,:] = line[-1,:]

    # Plot endpoints
    axis.plot(endpoints_1[:,0], endpoints_1[:,1], endpoints_1[:,2], '.', color=start_color)
    axis.plot(endpoints_2[:,0], endpoints_2[:,1], endpoints_2[:,2], '.', color=end_color)

    # Decorate
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    axis.set_title(title_string)
    axis.set_aspect('equal')

def mesh_plot(axis, fibers, title_string='mesh-plot', radius=2.0, radial_resolution=6, backend='matplotlib'):

    # Works only in 3D case
    nodes, tri = regular_cylinder_meshing(fibers, n_radial=radial_resolution, radius=radius)
    
    # Store them matplotlib Poly3DColletion for simple display
    mesh = Poly3DCollection(nodes[tri], alpha=1.0, facecolors='r', edgecolors='r', shade=True)

    axis.add_collection3d(mesh)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    axis.set_title(title_string)
    axis.set_aspect('equal')
    #axis.azim = azim
    #axis.elev = elev

## ------------------------------------------------------------------ M E A S U R E M E N T S ------------------------------------------------------------------ ##

## --------- Quantification operations -------------- ##
def measure_length(line, voxel_size=1.0):
    
    return voxel_size * np.sum(np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1)))
    
def measure_tortuosity(line, voxel_size=1.0):
    direct_length = voxel_size * np.linalg.norm(line[-1,:] - line[0,:], 2)
    total_length = measure_length(line, voxel_size=voxel_size)

    return direct_length / total_length

def measure_curvature(line):

    line_radii = compute_direction_vectors(line, epsilon=1e-10, illdefined_value=np.nan)[3]
    line_curvature = 1 / line_radii

    # Handle co-linear cases by setting curvature to zero
    line_curvature[np.isnan(line_radii)] = 0.0
    
    return line_curvature
    
def measure_fiber_statistics(fibers, stat='length', voxel_size=1.0):

    nFib = len(fibers)
    stat_array = np.zeros(shape=(nFib,1))
    
    if stat == 'length':
        for j, fib in enumerate(fibers):
            stat_array[j] = measure_length(fib, voxel_size=voxel_size)
    elif stat == 'tortuosity':
        for j, fib in enumerate(fibers):
            stat_array[j] = measure_tortuosity(fib, voxel_size=voxel_size)
    else:
        raise ValueError("Unknown statistic: {}".format(stat))
        
    return stat_array
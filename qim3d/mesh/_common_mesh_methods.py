import io

import numpy as np
import scipy
import scipy.ndimage
from pygel3d import hmesh
from pyvista import UnstructuredGrid, CellType, PolyData
from pytetwild import tetrahedralize, tetrahedralize_pv
import pyvista as pv

from qim3d.utils import log

class VolumeMesh(UnstructuredGrid):

    # def tetrahedralize(self, optimize:bool = True, edge_length_fac:float = 0.05):
    #     return VolumeMesh(tetrahedralize_pv(self, edge_length_fac, optimize))

    def export_to_comsol(self, 
               filename:str = 'mesh1.mphtxt'):
        
        if not filename.endswith('.mphtxt'):
            raise ValueError(f"Filename needs to have extension '.mphtxt. Your filename is {filename}")
        
        # These to can be arguments in the future if the feature is required
        tags = ('mesh1', )
        types = ('obj', )

        vertices_str = io.StringIO()
        np.savetxt(vertices_str, self.points.astype(np.float64, copy=False), fmt = '%.12f')

        tetras_str = io.StringIO()
        tetras = self.cells.reshape((-1, 5))[:, 1:]
        np.savetxt(tetras_str, tetras, fmt = '%d')

        with open(filename, "w") as f:
            writeline = lambda s: f.write(str(s) + '\n')
            # Some lines require number of characters before the actual string
            write_num_line = lambda s: writeline(str(len(s)) + ' ' + s)
            start_object = lambda obj_num: writeline(f'#--------------- Object {obj_num} ---------------\n\n0 0 1')

            #####################################
            #           HEADER
            #####################################
            writeline("# Created by COMSOL Multiphysics.\n\n# Major & minor version\n0 1")
            writeline(len(tags))
            for tag in tags:
                write_num_line(tag)
            writeline(len(types))
            for type in types:
                write_num_line(type)

            #####################################
            #           VERTICES
            #####################################
            start_object(0)
            write_num_line('Mesh') #class
            writeline(4) #version
            writeline(3) #sdim

            writeline(self.points.shape[0])
            writeline(0) #lowest mesh vertex index

            writeline(vertices_str.getvalue())

            #####################################
            #           TETRAS
            #####################################
            writeline(1) # number of element types
            write_num_line('tet')
            writeline(4) # number of vertices per element (tetrahedra)
            writeline(tetras.shape[0])
            writeline(tetras_str.getvalue())

            writeline(0) # number of geometric entity indices
   
class SurfaceMesh(PolyData):
    def __new__(cls, mesh):
        if isinstance(mesh, hmesh.Manifold):

            # get faces
            faces= np.ones((0, 3))
            for face in mesh.faces():
                new_ver = mesh.circulate_face(face)
                new_ver = np.expand_dims(np.array(new_ver), axis = 0)
                faces = np.append(faces, new_ver, axis = 0)
            return super().__new__(cls, mesh.points(), faces)

        elif isinstance(mesh, PolyData):
            return super().__new__(cls, mesh.points, mesh.faces)
        
    def tetrahedralize(self, optimize:bool = True, edge_length_fac:float = 0.05):
        return VolumeMesh(tetrahedralize_pv(self, edge_length_fac, optimize))



def from_volume(
    volume: np.ndarray, 
    mesh_precision: float = 1.0, 
    backend:str = 'pyvista', 
    method:str = 'marching_cubes', 
    return_pygel3D:bool = False,
    **kwargs: any
) -> SurfaceMesh | hmesh.Manifold:
    """
    Converts a 3D binary or grayscale volume into a polygon mesh (isosurface extraction).

    This function transforms voxel-based data into a vector-based surface representation (triangular mesh). This process, often called polygonization or tessellation, is a necessary step for 3D printing (exporting to STL), finite element analysis (FEA), or surface-based geometric measurements. It utilizes the [`volumetric_isocontour`](https://www2.compute.dtu.dk/projects/GEL/PyGEL/pygel3d/hmesh.html#volumetric_isocontour) function from PyGEL3D to generate a high-quality manifold.

    Args:
        volume (np.ndarray): A 3D numpy array representing a volume.
        mesh_precision (float, optional): Scaling factor for adjusting the resolution of the mesh.
                                          Default is 1.0 (no scaling).
        backend (str, optional): What python package is used to compute mesh from volume. 
            It is either 'pyvista' or 'pygel'. Default is 'pyvista'
        method (str, optional): Only applies if ˚backend = pyvista˚. What method is used to compute mesh from volume. 
            It can be either 'marching_cubes' or 'flying_edges'. Default is 'marching_cubes
        return_pygel3D (bool, optional): If set to True, returns pygel3d.hmesh.Manifold. If False, returns qim3d.mesh.SurfaceMesh.
            Default is False.
        **kwargs: Additional arguments to pass to the Pygel3D volumetric_isocontour function.

    Raises:
        ValueError: If the input is not 3D, is empty, or if `mesh_precision` is outside the (0, 1] range.

    Returns:
        mesh (hmesh.Manifold):
            The generated mesh object containing vertices, edges, and faces.

    Example:
        Convert a 3D numpy array to a Pygel3D mesh object:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.volume()

        # Visualize the generated blob
        qim3d.viz.volumetric(synthetic_blob)
        ```
        ![pygel3d_visualization_vol](../../assets/screenshots/viz-pygel_mesh_vol.png){width='300', length='200'}

        ```python
        # Convert the 3D numpy array to a Pygel3D mesh object
        mesh = qim3d.mesh.from_volume(synthetic_blob, mesh_precision=0.5)

        # Visualize the generated mesh
        qim3d.viz.mesh(mesh)
        ```
        ![pygel3d_visualization_mesh](../../assets/screenshots/viz-pygel_mesh.png){width='300', length='200'}
    """

    if volume.ndim != 3:
        msg = 'The input volume must be a 3D numpy array.'
        raise ValueError(msg)

    if volume.size == 0:
        msg = 'The input volume must not be empty.'
        raise ValueError(msg)

    if not (0 < mesh_precision <= 1):
        msg = 'The mesh precision must be between 0 and 1.'
        raise ValueError(msg)
    
    if backend not in ('pyvista', 'pygel'):
        msg = f"Backend has to be either 'pyvista' or 'pygel'. Yours is {backend}"
        raise ValueError(msg)

    # Apply scaling to adjust mesh resolution
    volume = scipy.ndimage.zoom(volume, zoom=mesh_precision, order=0)

    if backend == 'pyvista':
        grid = pv.ImageData(dimensions=volume.shape)
        mesh = grid.contour([1], volume.flatten(order="F"), method=method)

    elif backend == 'pygel':
        mesh = hmesh.volumetric_isocontour(volume, **kwargs)
        if return_pygel3D:
            return mesh

    return SurfaceMesh(mesh)

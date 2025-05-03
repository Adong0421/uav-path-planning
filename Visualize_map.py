import open3d as o3d
import numpy as np

def create_voxel_grid_from_glb(glb_file, voxel_size):
    mesh = o3d.io.read_triangle_mesh(glb_file)
    min_bound = np.min(mesh.vertices, axis=0)
    max_bound = np.max(mesh.vertices, axis=0)
    min_bound = np.squeeze(min_bound).astype(np.float64)
    max_bound = np.squeeze(max_bound).astype(np.float64)
    print('min_bound', min_bound)
    print('max_bound', max_bound)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    return voxel_grid
glb_file = r"D:\vpn\export.glb"
voxel_size = 2
voxel_grid = create_voxel_grid_from_glb(glb_file, voxel_size)
o3d.io.write_voxel_grid('voxel_grid_gc.ply', voxel_grid)
o3d.visualization.draw_geometries([voxel_grid])

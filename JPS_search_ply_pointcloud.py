import numpy as np
import heapq
import open3d as o3d
import time
import random


def read_glb_file(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    return vertices


def create_3d_grid(coordinates, cell_size):
    grid_size = np.ceil((np.max(coordinates, axis=0) - np.min(coordinates, axis=0)) / cell_size).astype(int)
    grid = np.zeros(grid_size, dtype=np.int8)
    origin = np.min(coordinates, axis=0)
    for x, y, z in coordinates:
        ix, iy, iz = int(np.floor((x - origin[0]) / cell_size[0])), \
                     int(np.floor((y - origin[1]) / cell_size[1])), \
                     int(np.floor((z - origin[2]) / cell_size[2]))
        if 0 <= ix < grid_size[0] and 0 <= iy < grid_size[1] and 0 <= iz < grid_size[2]:
            grid[ix, iy, iz] = 1
    return grid, grid_size


def convert_index_to_coordinate(index, cell_size, origin):
    x = origin[0] + (index[0] + 0.5) * cell_size[0]
    y = origin[1] + (index[1] + 0.5) * cell_size[1]
    z = origin[2] + (index[2] + 0.5) * cell_size[2]
    return x, y, z


def heuristic(a, b, heuristic_type='weighted'):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(a[2] - b[2])
    if heuristic_type == 'weighted':
        # 增加曼哈顿距离的权重，减少欧几里得距离的权重
        manhattan = dx + dy + dz
        euclidean = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return 0.8 * manhattan + 0.2 * euclidean
    elif heuristic_type == 'diagonal':
        # 对角线距离启发式函数
        return max(dx, dy, dz)


def jps(graph, start, end, heuristic_type='weighted', z_min=16, z_max=40):
    global open_list, closed_list, g_scores, f_scores, came_from  # 定义为全局变量
    open_list = []
    closed_list = set()
    g_scores = {tuple(start): 0}
    f_scores = {tuple(start): heuristic(start, end, heuristic_type)}
    came_from = {}
    heapq.heappush(open_list, (0 + heuristic(start, end, heuristic_type), 0, tuple(start)))
    while open_list:
        _, _, current = heapq.heappop(open_list)
        if tuple(current) == end:
            path = []
            current_tuple = current
            while current_tuple in came_from:
                path.append(current_tuple)
                current_tuple = came_from[current_tuple]
            return path[::-1]
        closed_list.add(tuple(current))
        neighbors = get_jump_points(graph, current, came_from.get(tuple(current)), z_min, z_max)
        for neighbor in neighbors:
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple in closed_list:
                continue
            tentative_g_score = g_scores[tuple(current)] + 1
            if tentative_g_score < g_scores.get(neighbor_tuple, np.inf):
                came_from[neighbor_tuple] = current
                g_scores[neighbor_tuple] = tentative_g_score
                f_scores[neighbor_tuple] = tentative_g_score + heuristic(neighbor, end, heuristic_type)
                heapq.heappush(open_list, (f_scores[neighbor_tuple], id(neighbor_tuple), neighbor_tuple))
    for node, parent in came_from.items():
        print(f"Node {node} came from {parent}")
    return []


def get_jump_points(graph, current, parent=None, z_min=16, z_max=40):
    def is_passable(coord):
        x, y, z = coord
        # 增加 z 轴范围检查
        if not (0 <= x < graph.shape[0] and 0 <= y < graph.shape[1] and z_min <= z < z_max):
            return False
        # 检查相邻单元格是否为障碍物
        adjacent = [(x + dx, y + dy, z + dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if
                    (dx, dy, dz) != (0, 0, 0)]
        for adj in adjacent:
            ax, ay, az = adj
            if 0 <= ax < graph.shape[0] and 0 <= ay < graph.shape[1] and z_min <= az < z_max and graph[ax, ay, az] == 1:
                return False
        return graph[x, y, z] == 0

    def jump(x, y, z, dx, dy, dz):
        while True:
            x, y, z = x + dx, y + dy, z + dz
            # 增加 z 轴范围检查
            if not is_passable((x, y, z)):
                return None
            if dx != 0 and dy != 0 and dz != 0:
                if not is_passable((x, y, z - dz)) or not is_passable((x, y - dy, z)) or not is_passable((x - dx, y, z)):
                    return (x, y, z)
            elif dx != 0 and dy != 0:
                if not is_passable((x, y, z - 1)) or not is_passable((x, y - dy, z)):
                    return (x, y, z)
            elif dx != 0 and dz != 0:
                if not is_passable((x, y - 1, z)) or not is_passable((x - dx, y, z)):
                    return (x, y, z)
            elif dy != 0 and dz != 0:
                if not is_passable((x - 1, y, z)) or not is_passable((x, y - dy, z)):
                    return (x, y, z)
            return (x, y, z)

    neighbors = []
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
                  (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
                  (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
                  (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                  (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                  (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]

    for dx, dy, dz in directions:
        jump_point = jump(current[0], current[1], current[2], dx, dy, dz)
        if jump_point:
            neighbors.append(jump_point)

    return neighbors


def create_path_point_cloud(path_coordinates, color=[1, 0, 0]):
    path_cloud = o3d.geometry.PointCloud()
    path_cloud.points = o3d.utility.Vector3dVector(np.array(path_coordinates))
    path_cloud.colors = o3d.utility.Vector3dVector(np.array([color for _ in path_coordinates]))
    return path_cloud


start_time = time.time()
filename = r"D:\vpn\export.glb"
coordinates = read_glb_file(filename)
# 减小网格的边长，增加网格的分辨率
cell_size_meters = 2  # 调整网格大小以匹配你的数据
# 创建3D网格
grid, grid_size = create_3d_grid(coordinates, (cell_size_meters, cell_size_meters, cell_size_meters))
np.save('grid.npy', grid)
np.save('grid_size.npy', grid_size)
# 可以尝试修改起点和终点来获取不同的路径
start = (0, 0, 16)  # 定义起点
goal = (1, 1, 38)  # 定义终点，需要根据实际情况调整
# 尝试不同的启发式函数类型
path = jps(grid, start, goal, heuristic_type='weighted', z_min=16, z_max=40)
print("Found path:", path)
print(f"Path length: {len(path)}")
if path:
    path_coordinates = [convert_index_to_coordinate(index, (cell_size_meters, cell_size_meters, cell_size_meters),
                                                    np.min(coordinates, axis=0)) for index in path]
    print("Path coordinates:", path_coordinates)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
    mesh = o3d.io.read_triangle_mesh(filename)
    path_cloud = create_path_point_cloud(path_coordinates, color=[1, 0, 0])
    o3d.visualization.draw_geometries([mesh, path_cloud])
else:
    print("No path found.")
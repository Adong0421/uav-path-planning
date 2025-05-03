import numpy as np
import heapq
import open3d as o3d
import time


def parse_coordinates_from_text(file_path):
    """解析文本文件中的坐标点，返回一个字典，其中键是区域名称，值是起始和终止坐标的元组"""
    region_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):
            region_file = lines[i].strip()
            start_coords = tuple(map(int, lines[i + 1].split(':')[1].strip().split(',')))
            end_coords = tuple(map(int, lines[i + 2].split(':')[1].strip().split(',')))
            region_data[region_file] = (start_coords, end_coords)
    print(region_data)
    return region_data


def read_glb_file(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    return vertices


def create_3d_grid(coordinates, cell_size):
    # 计算每个方向上的网格数量
    grid_size = np.ceil((np.max(coordinates, axis=0) - np.min(coordinates, axis=0)) / cell_size).astype(int)
    # 创建网格
    grid = np.zeros(grid_size, dtype=np.int8)
    # 使用 GLB 文件中的最小坐标作为原点位置
    origin = np.min(coordinates, axis=0)
    # 向量化操作来标记占据的网格单元
    # 首先计算每个坐标对应的网格索引
    indices = np.floor((coordinates - origin) / cell_size).astype(int)
    # 使用 numpy 的高级索引来标记网格单元
    valid_indices = np.all((indices >= 0) & (indices < grid_size), axis=1)
    grid[indices[valid_indices, 0], indices[valid_indices, 1], indices[valid_indices, 2]] = 1
    return grid, grid_size


def convert_index_to_coordinate(index, cell_size, origin):
    x = origin[0] + (index[0] + 0.5) * cell_size[0]
    y = origin[1] + (index[1] + 0.5) * cell_size[1]
    z = origin[2] + (index[2] + 0.5) * cell_size[2]
    return x, y, z


def heuristic(a, b, heuristic_type='manhattan'):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(a[2] - b[2])
    if heuristic_type == 'manhattan':
        return dx + dy + dz
    elif heuristic_type == 'euclidean':
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    elif heuristic_type == 'diagonal':
        return max(dx, dy, dz)
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")


def jps(graph, start, end, heuristic_type='manhattan'):
    open_list = []
    closed_list = set()
    # 使用 numpy 数组存储分数，使用数组索引作为节点标识
    g_scores = np.full(graph.shape, np.inf)
    f_scores = np.full(graph.shape, np.inf)
    came_from = {}
    g_scores[tuple(start)] = 0
    f_scores[tuple(start)] = heuristic(start, end, heuristic_type)
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
        neighbors = get_jump_points(graph, current, came_from.get(tuple(current)))
        for neighbor in neighbors:
            neighbor_tuple = tuple(neighbor)
            tentative_g_score = g_scores[current] + 1
            if tentative_g_score < g_scores[neighbor_tuple]:
                came_from[neighbor_tuple] = current
                g_scores[neighbor_tuple] = tentative_g_score
                f_scores[neighbor_tuple] = tentative_g_score + heuristic(neighbor, end, heuristic_type)
                heapq.heappush(open_list, (f_scores[neighbor_tuple], id(neighbor_tuple), neighbor_tuple))
    return []


def get_jump_points(graph, current, parent=None):
    def is_passable(coord):
        x, y, z = coord
        return 0 <= x < graph.shape[0] and 0 <= y < graph.shape[1] and 0 <= z < graph.shape[2] and graph[x, y, z] == 0

    def jump(x, y, z, dx, dy, dz):
        while True:
            x, y, z = x + dx, y + dy, z + dz
            if not is_passable((x, y, z)):
                return None
            if dx!= 0 and dy!= 0 and dz!= 0:
                if not is_passable((x, y, z - dz)) or not is_passable((x, y - dy, z)) or not is_passable((x - dx, y, z)):
                    return (x, y, z)
            elif dx!= 0 and dy!= 0:
                if not is_passable((x, y, z - 1)) or not is_passable((x, y - dy, z)):
                    return (x, y, z)
            elif dx!= 0 and dz!= 0:
                if not is_passable((x, y - 1, z)) or not is_passable((x - dx, y, z)):
                    return (x, y, z)
            elif dy!= 0 and dz!= 0:
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


def main(file_path):
    region_data = parse_coordinates_from_text(file_path)
    regions = list(region_data.keys())
    num_regions = len(regions)
    print(len(regions))
    path_lengths = np.zeros((num_regions + 1, num_regions + 1), dtype=object)  # 路径长度矩阵
    print(path_lengths.shape)

    path_lengths[1:, 0] = regions
    path_lengths[0, 1:] = regions

    start = (0, 0, 0)
    for i in range(len(regions) - 1):

        region1 = regions[i]
        start1, end1 = region_data[region1]


        path01 = jps(grid, start, start1, heuristic_type='diagonal')
        path_coordinates = [convert_index_to_coordinate(index, (cell_size_meters, cell_size_meters, cell_size_meters), np.min(coordinates, axis=0)) for index in
                              path01]
        print("Path coordinates from", start, "start to", region1, "end:", path_coordinates)
        print(f"Path length: {len(path01)}")
        print("")

            # 保存路径到文本文件
        save_path_to_file(path_coordinates, f"path_{start}_start_to_{region1}_end.txt")


        path02 = jps(grid, start, end1, heuristic_type='diagonal')
        path_coordinates = [convert_index_to_coordinate(index, (cell_size_meters, cell_size_meters, cell_size_meters), np.min(coordinates, axis=0)) for index in
                              path02]
        print("Path coordinates from", region1, "start to", start, "end:", path_coordinates)
        print(f"Path length: {len(path02)}")
        print("")
        # 保存路径到文本文件
        save_path_to_file(path_coordinates, f"path_{region1}_start_to_{start}_end.txt")



# 保存路径到文本文件
def save_path_to_file(path_coordinates, file_path):
    with open(file_path, "w") as f:
        for coordinates in path_coordinates:
            f.write(f"{coordinates}\n")


# 保存矩阵到文件
def save_matrix_to_file(matrix, file_path):
    with open(file_path, "w") as f:
        for row in matrix:
            f.write('\t'.join(str(item) for item in row))
            f.write('\n')


# 使用示例
file_path = 'SG.txt'
# 使用示例
start_time = time.time()
filename = r"D:\vpn\export.glb"
coordinates = read_glb_file(filename)
# 直接确定网格的边长（以米为单位）
cell_size_meters = 2  # 调整网格大小以匹配你的数据
# 创建 3D 网格
grid, grid_size = create_3d_grid(coordinates, (cell_size_meters, cell_size_meters, cell_size_meters))
# 转换路径中的网格索引坐标为原始地理坐标
# 输出 xyz 方向上的网格数量
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time)
main(file_path)
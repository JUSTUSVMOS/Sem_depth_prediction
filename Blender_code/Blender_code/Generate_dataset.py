import bpy
import bmesh
import random
import os

# Set the output folder path
output_folder_3D = "D:/TSMC/Blender/GeneratedModels"
output_folder_depth = "D:/TSMC/Blender/Modelsdepth"
output_folder_tri = "D:/TSMC/Blender/3D_triangles/"
os.makedirs(output_folder_3D, exist_ok=True)
os.makedirs(output_folder_depth, exist_ok=True)
os.makedirs(output_folder_tri, exist_ok=True)

# Parameters
num_models = 1
plane_x = 1000.0
plane_y = 750.0
height_min=5
height_max=40
gap = 1.0
shrink_distance = -2
inset_amount = 0.005

def clear_scene():
    """删除现有的所有网格对象。"""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
def create_plane(length, width):
    """创建一个指定长宽的矩形平面。"""
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "RectangularPlane"
    plane.scale.x = length
    plane.scale.y = width
    return plane

def subdivide_plane(plane, num_cuts):
    """对平面进行等分切割。"""
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(plane.data)
    
    for face in bm.faces:
        face.select = face.normal.z > 0.99  # 选择朝上的面
    bmesh.update_edit_mesh(plane.data)

    bpy.ops.mesh.subdivide(number_cuts=num_cuts)

     # 选择平行于X轴的边
    bpy.ops.mesh.select_all(action='DESELECT')  # 取消选择所有元素
    for edge in bm.edges:
        v1, v2 = edge.verts
        if abs(v1.co.y - v2.co.y) < 1e-6:  # 判断Y坐标是否几乎相等
            edge.select = True

     # 删除选中的平行X轴边
    bpy.ops.mesh.dissolve_edges()
    for face in bm.faces:
        face.select = True
    bmesh.update_edit_mesh(plane.data)
    return bmesh.from_edit_mesh(plane.data)

def select_non_adjacent_faces(bm, num_faces):
    """从非相邻的面中随机选择指定数量的面。"""
    extrude_faces = [face for face in bm.faces if face.normal.z > 0.99 and face.select]
    selected_faces = []

    # 遍历以选择非相邻的面
    while len(selected_faces) < num_faces and extrude_faces:
        face = random.choice(extrude_faces)
        
        # 检查是否与已选择的面相邻
        is_adjacent = any(
            set(face.edges) & set(selected_face.edges) for selected_face in selected_faces
        )
        
        if not is_adjacent:
            selected_faces.append(face)
        
        # 移除已考虑的面
        extrude_faces.remove(face)
    
    return selected_faces

def select_random_faces(bm, num_faces, height_min, height_max):
    """随机选择指定数量的非相邻面，分别拉伸不同高度。"""
    
    # 获取非相邻的随机面
    random_faces = select_non_adjacent_faces(bm, num_faces)
    
    if random_faces:
        # 生成滿足高度差大於 gap 的高度列表
        heights = [random.uniform(height_min, height_max)]
        while len(heights) < num_faces:
            new_height = random.uniform(height_min, height_max)
            if all(abs(new_height - h) > (gap-shrink_distance) for h in heights):
                heights.append(new_height)

        print(heights)
        # 拉伸每个非相邻的随机面，并分配高度
        for face, height in zip(random_faces, heights):
            # 清空选择状态
            for f in bm.faces:
                f.select = False
            face.select = True
            bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0, 0, height)})

    # 更新编辑网格
    bmesh.update_edit_mesh(bpy.context.object.data)
    
def extrude_selected_faces(height_min, height_max):
    """对选中的面执行 Extrude 操作，并向上随机移动。"""
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={
        "value": (0, 0, random.uniform(height_min, height_max))
    })
    
#复制 Z > 0 的面，并沿其法线方向移动指定的距离。
def duplicate_and_move_faces(obj, distance, inset_amount, shrink_distance):
    bpy.ops.mesh.select_all(action='DESELECT')
    
    # 仅选择 Z > 0 的面
    for face in bm.faces:
        face.select = False  # 清除所有选择
        if all(vert.co.z > 0 for vert in face.verts):  # 仅选择 Z > 0 的面
            face.select = True
     # 缩小选中的面 (Inset Faces)
    bpy.ops.mesh.inset(thickness=inset_amount)
    
    # 将缩小的面沿法线方向抬高 (Shrink/Fatten)
    bpy.ops.transform.shrink_fatten(value=shrink_distance)
    bpy.ops.mesh.duplicate()
    copied_faces = [face for face in bm.faces if face.select]
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, distance)}
    )
    
def export_model(output_folder, model_name, frame_index):
    """导出当前场景模型。"""
    filepath = os.path.join(output_folder, f"{model_name}_{frame_index:04}.obj")
    bpy.ops.export_scene.obj(filepath=filepath)
    print(f"Exported model {frame_index} to {filepath}")

def setup_depth_output_tree(output_folder_depth, frame_index):
    # 开启节点树，并清空现有节点
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()

    # 设置图像格式
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.view_layer.use_pass_normal = False
    bpy.context.view_layer.use_pass_z = True

    # 创建节点
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    map_range = tree.nodes.new(type="CompositorNodeMapRange")
    invert_depth = tree.nodes.new(type="CompositorNodeInvert")
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    
    depth_file_output.label = 'Depth Output'
    invert_depth.label = 'Invert Depth'

    # 设置 map_range 的输入范围
    cam = bpy.context.scene.objects['Camera']
    cam.location = (0.0, 0.0, 500.0)  # 假设相机位置Z轴远点
    map_range.inputs['From Min'].default_value = -cam.location.z
    map_range.inputs['From Max'].default_value = 0.0  # 调整视深度为所需值
    
    # 設置輸出範圍（調整對比度）
    map_range.inputs['To Min'].default_value = 0.0  # 對應最低亮度（黑色）
    map_range.inputs['To Max'].default_value = 5.0  # 對應最高亮度（白色）

    # 连接节点
    links = tree.links
    links.new(render_layers.outputs['Depth'], invert_depth.inputs[1])
    links.new(invert_depth.outputs[0], map_range.inputs[0])
    links.new(map_range.outputs[0], depth_file_output.inputs[0])

    # 设置输出路径
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].path = os.path.join(output_folder_depth, "depth_")
    #bpy.context.scene.frame_set(frame_index)

    return depth_file_output

def configure_camera(scene):
    cam = scene.objects['Camera']
    cam.rotation_euler = (0, 0, 0)
    cam.data.clip_end = 2000.0
    cam.data.angle = 10 * (3.1415926 / 180.0)
    cam.data.lens = 18.0

def render_frame(output_folder_depth, frame_index):
    scene = bpy.context.scene
    configure_camera(scene)
    
    # 设置分辨率
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 750
    scene.render.resolution_percentage = 100
    
    # 创建深度输出节点并渲染
    depth_file_output = setup_depth_output_tree(output_folder_depth, frame_index)
    
    # 设置当前帧并渲染
    scene.frame_set(frame_index)
    bpy.ops.render.render()

def export_triangulated_mesh(obj, output_file, gap, num_faces, mat_in=0, mat_out=-123):
    # 准备三角化物体
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # 使用 Blender 內建的三角化函數
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # 更新物体的 mesh 数据
    bm.to_mesh(mesh)
    bm.free()

    # 计算每个面的中心点 Z 坐标
    face_z_coords = [(face, sum((obj.matrix_world @ obj.data.vertices[v].co).z for v in face.vertices[:]) / 3) for face in obj.data.polygons]

    # 計算每個面中心點的 Z 坐標，只保留法向量平行於 Z 軸的面
    face_max_z = [
        (face, sum((obj.matrix_world @ obj.data.vertices[v].co).z for v in face.vertices[:]) / 3)
        for face in obj.data.polygons
        if abs(face.normal.z) > 0.99  # 篩選出法向量接近平行 Z 軸的面
    ]
    
    # 去除重複 Z 值，並根據 Z 值由高到低排序
    face_max_z = sorted(set(z for _, z in face_max_z), reverse=True)

    # 選擇高度：使用第 1、3、5... 高的 Z 坐標，盡量選出 num_faces 個不同的高度
    selected_heights = [face_max_z[i] for i in range(0, len(face_max_z), 2)][:num_faces]
    print(selected_heights)

    with open(output_file, "w") as file:
        for face, z_avg in face_z_coords:
            verts_in_face = face.vertices[:]
            
            if len(verts_in_face) == 3:
                # 取得三個點的世界坐標
                v1 = obj.matrix_world @ obj.data.vertices[verts_in_face[0]].co
                v2 = obj.matrix_world @ obj.data.vertices[verts_in_face[1]].co
                v3 = obj.matrix_world @ obj.data.vertices[verts_in_face[2]].co
                
                # 計算法線並驗證右手定則
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = edge1.cross(edge2)

                # 確保法線朝向正 Z 軸
                if normal.z < 0:
                    v2, v3 = v3, v2  # 交換 v2 和 v3 以確保法線方向正確

                # 分配材質參數
                vertex_z_coords = [v1.z, v2.z, v3.z]
                if any(z in selected_heights for z in vertex_z_coords):  # 若任何一個點的 Z 坐標在選擇的高度中
                    mat_in, mat_out = 1, -123
                elif all(z == vertex_z_coords[0] for z in vertex_z_coords) and (vertex_z_coords[0]+gap) in selected_heights:  # 若所有點的 Z 坐標相等且在選擇的高度中
                    mat_in, mat_out = 0, 1
                else:  # 其他情況
                    mat_in, mat_out = 0, -123
                
                # 格式化并写入行
                line = f"{mat_in:>6} {mat_out:>5} {v1.x:>7.2f} {v1.y:>7.2f} {v1.z:>7.2f} "
                line += f"{v2.x:>7.2f} {v2.y:>7.2f} {v2.z:>7.2f} "
                line += f"{v3.x:>7.2f} {v3.y:>7.2f} {v3.z:>7.2f}\n"
                file.write(line)
        
        # 追加额外的三角形数据
        additional_triangles = get_additional_triangles()
        file.write(additional_triangles.strip())

# 获取附加的三角形数据
def get_additional_triangles():
    return """
-125 -125  -500  -375    200   500  -375    200   500   375    200
-125 -125  -500  -375    200   500   375    200  -500   375    200
-122 -122   500  -375 -10000   500   375 -10000   500   375    200
-122 -122   500  -375 -10000   500   375    200   500  -375    200
-122 -122  -500   375 -10000  -500  -375 -10000  -500   375    200
-122 -122  -500   375    200  -500  -375 -10000  -500  -375    200
-122 -122  -500  -375 -10000   500  -375 -10000   500  -375    200
-122 -122  -500  -375 -10000   500  -375    200  -500  -375    200
-122 -122   500   375 -10000  -500   375 -10000   500   375    200
-122 -122   500   375    200  -500   375 -10000  -500   375    200
-127 -127   500  -375 -10000  -500  -375 -10000   500   375 -10000
-127 -127   500   375 -10000  -500  -375 -10000  -500   375 -10000
    """

# 删除切割线后，过滤 Z=0 的面
def dissolve_and_filter_faces(obj, z_threshold=0.01):
    
    # 选中所有元素
    bpy.ops.mesh.select_all(action='SELECT')
    
    # 删除切割线，保留必要的形状
    bpy.ops.mesh.dissolve_limited(angle_limit=0.01)
    
    # 获取 bmesh 数据
    bm = bmesh.from_edit_mesh(obj.data)
    
    # 取消选中 Z=0 的面
    #for face in bm.faces:
        #face_center_z = face.calc_center_median().z
        #if abs(face_center_z) < z_threshold:
            #face.select = False  # 取消选中
    
    # 更新 bmesh 数据
    bmesh.update_edit_mesh(obj.data)

def move_quad_faces_along_normal(obj, faces, distance):
    """根据四个顶点的平均法向量移动指定的面"""
    for face in faces:
        verts = [v for v in face.verts]
        if len(verts) != 4:
            continue  # 确保是四边形面

        # 获取四个顶点的世界坐标
        v1 = obj.matrix_world @ verts[0].co
        v2 = obj.matrix_world @ verts[1].co
        v3 = obj.matrix_world @ verts[2].co
        v4 = obj.matrix_world @ verts[3].co

        # 使用每三个顶点计算法向量并取平均
        normal1 = (v2 - v1).cross(v3 - v1).normalized()
        normal2 = (v3 - v2).cross(v4 - v2).normalized()
        normal3 = (v4 - v3).cross(v1 - v3).normalized()
        normal4 = (v1 - v4).cross(v2 - v4).normalized()

        # 平均法向量
        average_normal = (normal1 + normal2 + normal3 + normal4).normalized()

        # 按法向量方向移动四边形面
        translation = average_normal * distance
        for vert in verts:
            vert.co += obj.matrix_world.inverted() @ translation

# Generate 100 models
for i in range(num_models):  # 调整循环次数
    num_cuts = random.randint(5, 12) # 隨機選擇切割次數
    num_faces = random.randint(1, 3)   # 隨機選擇幾個高度
    clear_scene()
    plane = create_plane(plane_x, plane_y)
    bm = subdivide_plane(plane, num_cuts )
    select_random_faces(bm, num_faces, height_min, height_max)
    #extrude_selected_faces(height_min, height_max)
    dissolve_and_filter_faces(plane)
    duplicate_and_move_faces(plane, gap, inset_amount, shrink_distance)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    export_model(output_folder_3D, "model", i)

    #triangles
    obj = bpy.context.active_object  # 获取当前选中的物体
    output_file = f"{output_folder_tri}triangles_{i:04}.tri"
    export_triangulated_mesh(obj, output_file, gap, num_faces)
    
    #Depth
    render_frame(output_folder_depth, i)

print("Finished generating and exporting 100 models.")

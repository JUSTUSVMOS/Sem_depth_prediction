import bpy
import os
import time

# 設定參數
DEPTH_IMAGE_DIR = "D:/TSMC/Blender/demodepth/"
OUTPUT_DIR = "D:/TSMC/Blender/Restore_3D/"
WIDTH, HEIGHT = 1000, 750
RES_X, RES_Y = 200, 150
CAM_Z = 500.0
DISPLACE_STRENGTH = 100

# 確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 清除場景中所有 Mesh 物件
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# 取得所有 PNG 深度圖檔案
depth_images = [f for f in os.listdir(DEPTH_IMAGE_DIR) if f.endswith(".png")]

for img_name in depth_images:
    img_path = os.path.join(DEPTH_IMAGE_DIR, img_name)
    obj_name = os.path.splitext(img_name)[0]  # 取得不含副檔名的名稱
    output_obj_path = os.path.join(OUTPUT_DIR, f"{obj_name}.obj")
    
    print(f"處理: {img_name} -> {output_obj_path}")
    
    # 創建網格物件
    mesh = bpy.data.meshes.new(name=obj_name)
    plane = bpy.data.objects.new(obj_name, mesh)
    bpy.context.collection.objects.link(plane)
    
    # 生成細分網格
    vertices = [(x * WIDTH / RES_X - WIDTH / 2, 
                 y * HEIGHT / RES_Y - HEIGHT / 2, 
                 0) for y in range(RES_Y + 1) for x in range(RES_X + 1)]
    faces = [(y * (RES_X + 1) + x, 
              y * (RES_X + 1) + x + 1, 
              (y + 1) * (RES_X + 1) + x + 1, 
              (y + 1) * (RES_X + 1) + x) for y in range(RES_Y) for x in range(RES_X)]
    
    # 設置 Mesh
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    
    # 設置相機
    cam = bpy.context.scene.camera
    if cam is None:
        cam_data = bpy.data.cameras.new(name="Camera")
        cam = bpy.data.objects.new(name="Camera", object_data=cam_data)
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
    cam.location.z = CAM_Z
    
    # 強制切換到相機視角
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = {'area': area, 'region': region, 'space_data': area.spaces.active}
                    bpy.ops.view3d.view_camera(override)
                    break
    
    # UV 投影
    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)
    
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        view_area = next((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'), None)
        if view_area:
            for region in view_area.regions:
                if region.type == 'WINDOW':
                    override = {'area': view_area, 'region': region, 'space_data': view_area.spaces.active}
                    bpy.ops.uv.project_from_view(override, scale_to_bounds=True)
                    break
        else:
            print("Error: No 3D View found, unable to project UV.")
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')  # 確保回到物件模式
    
    # 添加位移修改器
    mod = plane.modifiers.new(name="Displace", type='DISPLACE')
    texture = bpy.data.textures.new(name=f"Texture_{obj_name}", type='IMAGE')
    texture.image = bpy.data.images.load(img_path)
    
    mod.texture = texture
    mod.texture_coords = 'UV'
    mod.strength = DISPLACE_STRENGTH
    mod.mid_level = 0
    mod.direction = 'Z'
    
    # 確保 UI 不會卡住
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    time.sleep(0.1)
    
    # 匯出 OBJ
    bpy.ops.export_scene.obj(filepath=output_obj_path, use_selection=True, use_materials=False)
    
    # 刪除生成的物件
    bpy.data.objects.remove(plane, do_unlink=True)
    bpy.data.meshes.remove(mesh, do_unlink=True)
    bpy.data.textures.remove(texture, do_unlink=True)
    

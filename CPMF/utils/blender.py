"""
We used a 60 camera setup to capture the 3D model of the object from different angles.
This code should be imported in blender to get the png and .tiff files.
The code also will place in 60 camera with backlight to give some shadow to the object.
Note: 1. Blender version should be 3.2 and the default import file of ply should be overwritten 
according to this 
video :https://www.youtube.com/watch?v=-OMV2LrTwVw&ab_channel=MichaelProstka
git: https://github.com/TombstoneTumbleweedArt/import-ply-as-verts/blob/916bbd32af17916ee7cd6c9a6256abc8f62b68a6/README.md

2. The video also demonstrates the right voxelization and texturing of the object in blender after importing the ply file.
"""


"""
Script1: Place 60 cameras in an atomic config around the object
"""
"""
import bpy
from math import radians, sin, cos, pi
from mathutils import Vector

# Create a master empty object to parent all cameras and lights
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
master_empty = bpy.context.object
master_empty.name = 'Master_Empty'

def add_camera(location, name, parent):
    # Create camera
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = name
    
    # Point the camera towards the center (0, 0, 0)
    look_at = Vector((0, 0, 0)) - Vector(location)
    look_at.normalize()
    
    # Calculate the camera rotation
    rot_quat = look_at.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Set the parent of the camera
    camera.parent = parent

def add_area_light(location, name, parent):
    # Create area light
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = 0  # Set light energy

    # Point the light towards the center (0, 0, 0)
    look_at = Vector((0, 0, 0)) - Vector(location)
    look_at.normalize()
    
    # Calculate the light rotation
    rot_quat = look_at.to_track_quat('-Z', 'Y')
    light.rotation_euler = rot_quat.to_euler()

    # Set the parent of the light
    light.parent = parent

# Parameters for camera and light placement
elements_per_ring = 15
radius = 30  # Distance from center
total_elements = 0  # Keep track of the total number of cameras and lights

# Helper function to calculate the position for different rings
def calculate_position(phi, ring_angle, offset_angle=0):
    x = radius * cos(ring_angle) * sin(phi + offset_angle)
    y = radius * sin(ring_angle) * sin(phi + offset_angle)
    z = radius * cos(phi)
    return x, y, z

# Define the angles for the rings
ring_angles = {
    'Vertical': 0,
    'Horizontal': pi / 2,
    '45Degree': pi / 4,
    '135Degree': 3 * pi / 4  # 45 degrees to Y-axis in the negative X direction
}

# Place cameras and lights for each ring
for ring_name, ring_angle in ring_angles.items():
    offset_angle = 0 if ring_name in ['Vertical', 'Horizontal'] else pi  # Offset for the last two rings
    for elem in range(elements_per_ring):
        # Azimuthal angle (phi): 360-degree intervals divided by the number of elements
        phi = radians(360 / elements_per_ring * elem)
        
        # Calculate position
        x, y, z = calculate_position(phi, ring_angle, offset_angle)
        
        # Add camera and light to the scene and parent them to the master empty
        element_name = f"{str(total_elements).zfill(2)}"
        add_camera((x, y, z), f"Camera_{element_name}", master_empty)
        add_area_light((x, y, z), f"Light_{element_name}", master_empty)
        total_elements += 1

print("Four rings of cameras and lights placed successfully.")

"""

##################################################################################################

"""
Script2: Render the object from each camera and save the images
This script will take import a ply file and will save the .tiff and .png at the given location, moreover, there is also a commented out 
code to import the ground truth patch of the point cloud.
"""
"""
import bpy
import os
import numpy as np
import bpy_extras
import tifffile
import json
import PIL as Image
import gpu
from gpu_extras.batch import batch_for_shader


# Function to import a PLY file
def import_ply(filepath):
    bpy.ops.import_mesh.ply(filepath=filepath)


# Function to check if a point is visible from the camera
def is_visible(scene, cam, point):
    # Get the camera view matrix
    cam_matrix = cam.matrix_world.normalized().inverted()
    # Translate the point into camera space
    cam_point = cam_matrix @ point
    # Check if the point is in the view frustum (between the clipping planes)
    return (cam.data.clip_start < cam_point.length < cam.data.clip_end)


#old

def is_visible(scene, cam, point):
    direction = point - cam.location
    result = scene.ray_cast(scene.view_layers[0], cam.location, direction)
    if isinstance(result, tuple):
        hit, location, *_ = result
    else:
        hit, location, _, _, _ = result
    return hit and (location - point).length < 20000   # set high just to be sure to get more points in


# Function to get visible points from the point cloud
def get_visible_points(obj, cam, scene):
    visible_points = []
    depsgraph = bpy.context.evaluated_depsgraph_get()  # Get the depsgraph
    for vert in obj.data.vertices:
        world_coord = obj.matrix_world @ vert.co
        if is_visible(scene, cam, world_coord):
            visible_points.append(world_coord)
    return visible_points

# Function to project visible points to 2D and store XYZ coordinates
def project_points_to_image(points, cam, image_size=900):
    depth_image = np.zeros((image_size, image_size, 3), dtype=np.float32)

    for world_coord in points:
        # Calculate relative position (camera's location as origin)
        relative_pos = world_coord - cam.location
        x, y, z = relative_pos.x, relative_pos.y, relative_pos.z

        # Project to 2D using camera projection
        co_2d = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, cam, world_coord)
        image_x = int(co_2d.x * image_size)
        image_y = int((1 - co_2d.y) * image_size)

        # Update depth image
        if 0 <= image_x < image_size and 0 <= image_y < image_size:
            depth_image[image_y, image_x, 0] = x  # Set X value relative to camera
            depth_image[image_y, image_x, 1] = y  # Set Y value relative to camera
            depth_image[image_y, image_x, 2] = z  # Set Z value relative to camera

    return depth_image


# Function to set light visibility
def set_light_visibility(camera_name, visibility):
    light_name = camera_name.replace("Camera", "Light")
    light_object = bpy.data.objects.get(light_name)
    if light_object and light_object.type == 'LIGHT':
        light_object.hide_render = not visibility
        light_object.data.energy = 30000 if visibility else 0
        
        
def save_tiff_with_metadata(image, file_path):
    # Metadata in JSON format
    metadata_json = {
        'ImageWidth': 900,
        'ImageLength': 900,
        'BitsPerSample': (32, 32, 32),
        'Compression': 'None',
        'PhotometricInterpretation': 'RGB',
        'ImageDescription': '{"shape": [900, 900, 3]}',
        'SamplesPerPixel': 3,
        'RowsPerStrip': 900,
        'XResolution': (1, 1),
        'YResolution': (1, 1),
        'PlanarConfiguration': 'Contig',
        'ResolutionUnit': 'None',
        'Software': 'tifffile.py',
        'SampleFormat': 'IEEEFP',
    }
    metadata_str = json.dumps(metadata_json)

    # Save the TIFF file with metadata in the ImageDescription tag
    tifffile.imwrite(file_path, image, photometric='rgb', description=metadata_str)
    
def get_intrinsic_camera_matrix(cam, image_size):
    # Assuming square pixels (aspect ratio = 1)
    aspect_ratio = 1
    # Get the focal length in pixel units
    f_pix = cam.data.lens * image_size / cam.data.sensor_width
    # Principal point coordinates (assuming they are at the center of the image)
    cx = image_size / 2
    cy = image_size / 2

    intrinsic_matrix = np.array([
        [f_pix, 0, cx],
        [0, f_pix * aspect_ratio, cy],
        [0, 0, 1]
    ])
    return intrinsic_matrix


def set_object_visibility(condition, visibility):
    for obj in bpy.data.objects:
        if condition(obj.name):
            obj.hide_set(not visibility)
            
def delete_object(obj_name):
    # Check if the object exists in the scene
    if obj_name in bpy.data.objects:
        # Get the object
        obj = bpy.data.objects[obj_name]
        # Select the object (required for deletion)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        # Delete the object
        bpy.ops.object.delete()
        




           
    
# Directories for saving images
save_dir_png = r"C:\Users\abhin\Downloads\Test\Real_AD-3D\shell\train\good\rgb"
save_dir_tiff = r"C:\Users\abhin\Downloads\Test\Real_AD-3D\shell\train\good\xyz"
#save_dir_png_marked_only = r"C:\Users\abhin\Downloads\Test\Real_AD-3D\gemstone\train\good\gt"


os.makedirs(save_dir_png, exist_ok=True)
os.makedirs(save_dir_tiff, exist_ok=True)

# Set render settings for PNG
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGB'
bpy.context.scene.render.resolution_x = 900
bpy.context.scene.render.resolution_y = 900
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.film_transparent = False
bpy.data.worlds['World'].color = (0, 0, 0)  # Black color


# Import PLY file
#ply_filepath = r"C:\Users\abhin\Downloads\Test\output_folder_groundtruth_diamond\unmarked\535_sink_cut_unmarked.ply"
#ply_filepath = r"C:\Users\abhin\Downloads\Real3D-AD-PLY\Real3D-AD-PLY\diamond\40_template.ply"
ply_filepath = r"C:\Users\abhin\Downloads\Real3D-AD-PLY\Real3D-AD-PLY\shell\63_template.ply"
#import_ply(ply_filepath)

filename = "63_template"


# Get the mesh object name from the PLY file g
mesh_object_name = os.path.splitext(os.path.basename(ply_filepath))[0]
mesh_object = bpy.data.objects.get(mesh_object_name)


bpy.data.worlds['World'].use_nodes = False
for obj in bpy.data.objects:
    print(obj.name, obj.type)


patch_filename = r"C:\Users\abhin\Downloads\Test\output_folder_groundtruth_diamond\marked\535_sink_cut_marked.ply"
#import_ply(patch_filename)




# Process each camera
for camera in bpy.data.objects:
    if camera.type == 'CAMERA' and camera.name.startswith("Camera_"):
            
        intrinsic_matrix = get_intrinsic_camera_matrix(camera, 900)  # Assuming image size is 900x900
        print(f"Intrinsic Matrix for {camera.name}:")
        print(intrinsic_matrix)
            
                    
        for light_object in bpy.data.objects:
            if light_object.type == 'LIGHT':
                light_object.hide_render = True
                light_object.data.energy = 0

        set_light_visibility(camera.name, True)
        bpy.context.scene.camera = camera
        bpy.ops.render.render(write_still=True)
        image_path_png = os.path.join(save_dir_png,f"{filename}_"+ f"{camera.name}.png")
        bpy.data.images['Render Result'].save_render(filepath=image_path_png)
            
        if mesh_object and mesh_object.type == 'MESH':
            print("HEYY")
            scene = bpy.context.scene
            visible_points = get_visible_points(mesh_object, camera, scene)
            tiff_image = project_points_to_image(visible_points, camera)
                
            # Save TIFF with metadata
            image_path_tiff = os.path.join(save_dir_tiff,f"{filename}_"+ f"{camera.name}.tiff")
            save_tiff_with_metadata(tiff_image.astype(np.float32), image_path_tiff)

        set_light_visibility(camera.name, False)
        print(f"Saved PNG: {image_path_png}")
        print(f"Saved TIFF: {image_path_tiff}")
            

            
print("All images have been rendered and saved!!!")     



""""""
#just for the anomoly marked
for camera in bpy.data.objects:
    if camera.type == 'CAMERA' and camera.name.startswith("Camera_43"):
             
        for light_object in bpy.data.objects:
            if light_object.type == 'LIGHT':
                light_object.hide_render = True
                light_object.data.energy = 0

        set_light_visibility(camera.name, True)
        bpy.context.scene.camera = camera
        bpy.ops.render.render(write_still=True)
        image_path_png = os.path.join(save_dir_png_marked_only,f"{filename}_"+ f"{camera.name}.png")
        bpy.data.images['Render Result'].save_render(filepath=image_path_png)
        print(f"Saved PNG Ground truth: {image_path_png}")



"""""
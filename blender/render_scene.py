import os, math
import bpy
import bmesh
import random
import numpy as np
from mathutils import Vector, Matrix


def createMetaball(n=10, r0=4.0, r1=2.5, render_resolution=0.05):
    origin = (0, 0, 0)
    metaball = bpy.data.metaballs.new("MetaBall")
    obj = bpy.data.objects.new("MetaBallObject", metaball)
    bpy.context.collection.objects.link(obj)

    metaball.resolution = 0.2
    metaball.render_resolution = render_resolution

    for i in range(n):
        location = Vector(origin) + Vector(random.uniform(-r0, r0) for i in range(3))

        element = metaball.elements.new()
        element.co = location
        element.radius = r1 * (random.random() * 0.25 + 0.85)

    return obj


def build_sphere():
    # Add sky sphere
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=50, location=(0, 0, 0), segments=250)
    sphere = bpy.context.selected_objects[0]
    material = bpy.data.materials.new(name="Sky_Noise_Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    # # Clear default nodes
    for node in material.node_tree.nodes:
        material.node_tree.nodes.remove(node)

    # Add shader nodes
    shader_output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse_shader = nodes.new(type="ShaderNodeBsdfDiffuse")
    texture_coord = nodes.new(type="ShaderNodeTexCoord")
    noise_texture = nodes.new(type="ShaderNodeTexNoise")
    mapping_node = nodes.new(type="ShaderNodeMapping")  # New node
    bright_contrast = nodes.new(type="ShaderNodeBrightContrast")

    bright_contrast.inputs["Bright"].default_value = 1.0  # Change as needed
    bright_contrast.inputs["Contrast"].default_value = 15.0  # Change as needed
    mapping_node.inputs["Scale"].default_value = (
        0.2,
        0.2,
        0.2,
    )  # (X, Y, Z) scaling factors

    # Connect nodes together
    links = material.node_tree.links
    link0 = links.new
    link0(
        texture_coord.outputs["Object"], mapping_node.inputs["Vector"]
    )  # New connection
    link0(
        mapping_node.outputs["Vector"], noise_texture.inputs["Vector"]
    )  # Updated connection

    link0(
        noise_texture.outputs["Color"], bright_contrast.inputs[0]
    )  # Updated connection
    link0(bright_contrast.outputs[0], diffuse_shader.inputs["Color"])  # New connection
    link0(diffuse_shader.outputs["BSDF"], shader_output.inputs["Surface"])
    link0(mapping_node.outputs["Vector"], noise_texture.inputs["Vector"])

    sphere.data.materials.append(material)


def build_cube(n=10, r0=4.0, r1=2.5, render_resolution=0.05):
    # Add cube to scene
    # bpy.ops.mesh.primitive_cube_add()
    # cube = bpy.context.selected_objects[0]
    # cube.location = (0.0, 0.0, 0.0)

    cube = createMetaball(n=n, r0=r0, r1=r1, render_resolution=render_resolution)
    # Create a new material
    material = bpy.data.materials.new(name="Perlin_Noise_Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    # # Clear default nodes
    for node in material.node_tree.nodes:
        material.node_tree.nodes.remove(node)

    # Add shader nodes
    shader_output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse_shader = nodes.new(type="ShaderNodeBsdfDiffuse")
    texture_coord = nodes.new(type="ShaderNodeTexCoord")
    noise_texture = nodes.new(type="ShaderNodeTexNoise")
    mapping_node = nodes.new(type="ShaderNodeMapping")  # New node
    bright_contrast = nodes.new(type="ShaderNodeBrightContrast")

    bright_contrast.inputs["Bright"].default_value = 1.0  # Change as needed
    bright_contrast.inputs["Contrast"].default_value = 15.0  # Change as needed
    mapping_node.inputs["Scale"].default_value = (
        1.0,
        1.0,
        1.0,
    )  # (X, Y, Z) scaling factors

    # Connect nodes together
    links = material.node_tree.links
    link0 = links.new
    link0(
        texture_coord.outputs["Object"], mapping_node.inputs["Vector"]
    )  # New connection
    link0(
        mapping_node.outputs["Vector"], noise_texture.inputs["Vector"]
    )  # Updated connection

    link0(
        noise_texture.outputs["Color"], bright_contrast.inputs[0]
    )  # Updated connection
    link0(bright_contrast.outputs[0], diffuse_shader.inputs["Color"])  # New connection
    link0(diffuse_shader.outputs["BSDF"], shader_output.inputs["Surface"])
    link0(mapping_node.outputs["Vector"], noise_texture.inputs["Vector"])

    cube.data.materials.append(material)


def render_scene_examples(
    start_example, n_views, n=10, r0=4.0, r1=2.5, render_resolution=0.05
):
    # Set up rendering
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    render.engine = "BLENDER_EEVEE"
    render.image_settings.color_mode = "RGB"  # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = "8"
    render.image_settings.file_format = "PNG"
    render.resolution_x = 128
    render.resolution_y = 128
    render.resolution_percentage = 100
    render.film_transparent = True

    scene.use_nodes = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_direct = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_indirect = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create input render layer node
    render_layers = nodes.new("CompositorNodeRLayers")

    # Create depth output nodes
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = "Depth Output"
    depth_file_output.base_path = ""
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = "OPEN_EXR"
    depth_file_output.format.color_depth = "16"
    links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])

    # # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = "MULTIPLY"
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = "ADD"
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "Normal Output"
    normal_file_output.base_path = ""
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = "PNG"
    links.new(bias_node.outputs[0], normal_file_output.inputs[0])

    # # Create albedo output nodes
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = "Albedo Output"
    albedo_file_output.base_path = ""
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = "RGB"
    albedo_file_output.format.color_depth = "8"
    links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])

    # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()

    bpy.ops.object.select_all(action="DESELECT")

    ################ START SCENE LOGIC #################
    # build_sphere()
    build_cube(n, r0, r1, render_resolution)
    ################ END SCENE LOGIC ###################

    # obj = bpy.context.selected_objects[0]
    # context.view_layer.objects.active = obj

    # Make light just directional, disable shadows.
    light = bpy.data.lights["Light"]
    light.type = "SUN"
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 2.0

    # Add another light source so stuff facing away from light is not completely dark
    bpy.ops.object.light_add(type="SUN")
    light2 = bpy.data.lights["Sun"]
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 0.015
    bpy.data.objects["Sun"].rotation_euler = bpy.data.objects["Light"].rotation_euler
    bpy.data.objects["Sun"].rotation_euler[0] += 180

    # Place camera
    cam = scene.objects["Camera"]
    cam.location = (0, 10, 5)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam.data.clip_end = 10000.0

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty

    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty

    view_count = n_views
    stepsize = 360.0 / view_count

    fp = os.path.join("../data", "example")

    for i in range(0, view_count):
        render_file_path = fp + "_" + str(i + start_example)
        scene.render.filepath = render_file_path
        depth_file_output.file_slots[0].path = render_file_path + "_depth"
        normal_file_output.file_slots[0].path = render_file_path + "_normal"
        albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
        bpy.ops.render.render(write_still=True)

        scene.render.filepath = render_file_path + "_delta"

        x, y, z = (
            (1 if random.random() > 0.5 else -1) * 1.0 / 8.0,
            (1 if random.random() > 0.5 else -1) * 1.0 / 8.0,
            (1 if random.random() > 0.5 else -1) * 1.0 / 8.0,
        )

        cam.location[0] += x * 2.0
        cam.location[1] += y * 2.0
        cam.location[2] += z * 2.0
        cam_empty.rotation_euler[0] += x
        cam_empty.rotation_euler[1] += y
        cam_empty.rotation_euler[2] += z
        depth_file_output.file_slots[0].path = render_file_path + "_depth_delta"
        normal_file_output.file_slots[0].path = render_file_path + "_normal_delta"
        albedo_file_output.file_slots[0].path = render_file_path + "_albedo_delta"
        bpy.ops.render.render(write_still=True)
        cam_empty.rotation_euler[0] -= x
        cam_empty.rotation_euler[1] -= y
        cam_empty.rotation_euler[2] -= z
        cam.location[0] -= x * 2.0
        cam.location[1] -= y * 2.0
        cam.location[2] -= z * 2.0
        cam_empty.rotation_euler[2] += math.radians(stepsize)


example = 0
for i in range(0, 100):
    bpy.ops.wm.read_factory_settings()
    n = 10 + int(20 * random.random())
    r0 = 4.0
    r1 = 2.5
    render_resolution = 0.25
    render_scene_examples(example, 10, n, r0, r1, render_resolution)
    example += 10

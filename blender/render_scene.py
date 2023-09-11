import os, math
import bpy
import random
from mathutils import Vector


def createMetaball(origin=(0, 0, 0), n=30, r0=4, r1=2.5):
    metaball = bpy.data.metaballs.new("MetaBall")
    obj = bpy.data.objects.new("MetaBallObject", metaball)
    bpy.context.collection.objects.link(obj)

    metaball.resolution = 0.2
    metaball.render_resolution = 0.05

    for i in range(n):
        location = Vector(origin) + Vector(random.uniform(-r0, r0) for i in range(3))

        element = metaball.elements.new()
        element.co = location
        element.radius = r1

    return obj


# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = "BLENDER_EEVEE"
render.image_settings.color_mode = "RGB"  # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = "8"
render.image_settings.file_format = "PNG"
render.resolution_x = 256
render.resolution_y = 256
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True

# Clear default nodes
for n in nodes:
    nodes.remove(n)

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
# alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
# links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
# links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

# albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
# albedo_file_output.label = "Albedo Output"
# albedo_file_output.base_path = ""
# albedo_file_output.file_slots[0].use_node_format = True
# albedo_file_output.format.file_format = "PNG"
# albedo_file_output.format.color_mode = "RGB"
# albedo_file_output.format.color_depth = "8"
# links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])


# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()


bpy.ops.object.select_all(action="DESELECT")

################ START SCENE LOGIC #################
# Add cube to scene
bpy.ops.mesh.primitive_cube_add()
cube = bpy.context.selected_objects[0]
cube.location = (0.0, 0.0, 0.0)
createMetaball()

################ END SCENE LOGIC ###################

obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

# Possibly disable specular shading
for slot in obj.material_slots:
    node = slot.material.node_tree.nodes["Principled BSDF"]
    node.inputs["Specular"].default_value = 0.05


# Make light just directional, disable shadows.
light = bpy.data.lights["Light"]
light.type = "SUN"
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 10.0

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

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

view_count = 10
stepsize = 360.0 / view_count
rotation_mode = "XYZ"

model_identifier = "cube"
fp = os.path.join("../data", model_identifier)

for i in range(0, view_count):
    print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

    render_file_path = fp + "_r_{0:03d}".format(int(i * stepsize))

    scene.render.filepath = render_file_path
    depth_file_output.file_slots[0].path = render_file_path + "_depth"
    normal_file_output.file_slots[0].path = render_file_path + "_normal"
    # albedo_file_output.file_slots[0].path = render_file_path + "_albedo"

    bpy.ops.render.render(write_still=True)  # render still
    pixels = bpy.data.images["Viewer Node"]
    print(pixels)
    cam_empty.rotation_euler[2] += math.radians(stepsize)

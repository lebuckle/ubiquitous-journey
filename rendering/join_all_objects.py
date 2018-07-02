# Select all objects in a scene and combine them
# Used for voxel models imported into blender to combine into one mesh
# This makes processing much faster
import bpy

counter = 0
for ob in bpy.context.scene.objects:
    counter = counter + 1
    # Show progress
    print("Counter: {}" .format(counter))
    if ob.type == 'MESH':
        ob.select = True
        bpy.context.scene.objects.active = ob
    else:
        ob.select = False
bpy.ops.object.join()
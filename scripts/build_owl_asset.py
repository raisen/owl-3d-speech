#!/usr/bin/env python3
"""
Procedural 3D Rigged Owl Generator for Blender Headless Pipeline.
================================================================

This script builds a complete, stylized 3D owl asset from scratch using Blender's
Python API (bpy). It generates:
  - Full anatomy: Torso, chest bib, facial disc, golden irises, pupils, gloss cornea,
    hooked upper beak, articulated lower mandible, ear tufts (plumicorns), folded wings,
    tail feathers, and perched wooden branch.
  - Armature Rig: Root -> Spine -> Chest -> Neck -> Head, with Beak_Lower, Beak_Upper,
    Eyelid_L, Eyelid_R, EarTuft_L, EarTuft_R bones.
  - PBR Materials: Mottled feathers, radiant golden amber eyes, jet black pupils,
    keratin beak, and textured bark.
  - Vertex groups & Armature skinning.
  - Clean glTF / GLB export for downstream animation or web display via <model-viewer>.

Usage:
  blender -b -P scripts/build_owl_asset.py -- [options]
  blender -b -P scripts/build_owl_asset.py -- --output assets/owl_base.glb
"""

import sys
import os
import math
import argparse

# Check if running inside Blender
try:
    import bpy
    import bmesh
    from mathutils import Vector, Euler, Matrix
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def parse_args():
    """Parse command line arguments passed after '--' in Blender."""
    argv = sys.argv
    if "--" in argv:
        args_to_parse = argv[argv.index("--") + 1:]
    else:
        args_to_parse = []

    parser = argparse.ArgumentParser(description="Procedural 3D Rigged Owl Asset Generator")
    parser.add_argument("--output", "-o", type=str, default="assets/owl_base.glb",
                        help="Output path for the generated GLB file")
    parser.add_argument("--subdivisions", type=int, default=2,
                        help="Subdivision level for organic geometry smoothness")
    return parser.parse_args(args_to_parse)


def clean_scene():
    """Clear all default objects, collections, and materials."""
    if not IN_BLENDER:
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Remove any leftover objects or orphan data
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.armatures:
        bpy.data.armatures.remove(block)


def create_pbr_material(name, base_color, roughness=0.7, metallic=0.0, emission=None, emission_strength=1.0):
    """Create a standard PBR Principled BSDF material compatible across Blender 3.x and 4.x."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf:
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Color socket
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = base_color

    # Roughness socket
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = roughness

    # Metallic socket
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic

    # Emission handling across versions
    if emission:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = emission
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = emission
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength

    return mat


def setup_materials():
    """Setup rich PBR materials for the owl anatomy."""
    materials = {}

    # 1. Main feather body (warm tawny brown / barred owl)
    materials["Body"] = create_pbr_material(
        "Mat_Owl_Body",
        base_color=(0.28, 0.18, 0.11, 1.0),
        roughness=0.88,
        metallic=0.0
    )

    # 2. Chest Bib / Breast feathers (soft cream buff)
    materials["ChestBib"] = create_pbr_material(
        "Mat_Owl_ChestBib",
        base_color=(0.88, 0.81, 0.70, 1.0),
        roughness=0.90,
        metallic=0.0
    )

    # 3. Facial Disc (radiant parchment white with subtle contrast)
    materials["FacialDisc"] = create_pbr_material(
        "Mat_Owl_FacialDisc",
        base_color=(0.94, 0.90, 0.84, 1.0),
        roughness=0.82,
        metallic=0.0
    )

    # 4. Facial Disc Rim (dark framing ring)
    materials["FacialDiscRim"] = create_pbr_material(
        "Mat_Owl_FacialDiscRim",
        base_color=(0.14, 0.09, 0.06, 1.0),
        roughness=0.85,
        metallic=0.0
    )

    # 5. Beak Keratin (dark charcoal horn with specular polish)
    materials["Beak"] = create_pbr_material(
        "Mat_Owl_Beak",
        base_color=(0.12, 0.11, 0.10, 1.0),
        roughness=0.32,
        metallic=0.05
    )

    # 6. Eye Golden Iris (radiant Strigiformes golden honey)
    materials["Iris"] = create_pbr_material(
        "Mat_Owl_Iris",
        base_color=(1.00, 0.65, 0.05, 1.0),
        roughness=0.15,
        metallic=0.10,
        emission=(1.00, 0.60, 0.04, 1.0),
        emission_strength=0.18
    )

    # 7. Eye Pupil (deep velvet jet black)
    materials["Pupil"] = create_pbr_material(
        "Mat_Owl_Pupil",
        base_color=(0.015, 0.015, 0.015, 1.0),
        roughness=0.05,
        metallic=0.0
    )

    # 8. Cornea Gloss (wet reflective corneal surface)
    materials["Cornea"] = create_pbr_material(
        "Mat_Owl_Cornea",
        base_color=(0.95, 0.95, 1.0, 1.0),
        roughness=0.02,
        metallic=0.0
    )

    # 9. Wings & Flight Feathers (deep chocolate brown with barring)
    materials["Wings"] = create_pbr_material(
        "Mat_Owl_Wings",
        base_color=(0.20, 0.13, 0.08, 1.0),
        roughness=0.85,
        metallic=0.0
    )

    # 10. Perched Wooden Branch (rough natural bark)
    materials["Branch"] = create_pbr_material(
        "Mat_Wood_Branch",
        base_color=(0.22, 0.15, 0.10, 1.0),
        roughness=0.95,
        metallic=0.0
    )

    # 11. Claws / Talons (slate gray keratin)
    materials["Talons"] = create_pbr_material(
        "Mat_Owl_Talons",
        base_color=(0.10, 0.10, 0.10, 1.0),
        roughness=0.40,
        metallic=0.05
    )

    return materials


def assign_vertex_group(obj, group_name, weight=1.0):
    """Create a vertex group on the object and assign all current vertices."""
    vg = obj.vertex_groups.new(name=group_name)
    all_indices = [v.index for v in obj.data.vertices]
    if all_indices:
        vg.add(all_indices, weight, 'REPLACE')


def create_mesh_part(name, primitive_func, material, group_name, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), **prim_kwargs):
    """Helper to create a mesh object, set transform, assign material and vertex group."""
    primitive_func(**prim_kwargs)
    obj = bpy.context.active_object
    obj.name = name
    obj.location = location
    obj.rotation_euler = rotation
    obj.scale = scale
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    if material:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    if group_name:
        assign_vertex_group(obj, group_name, weight=1.0)

    # Smooth shading
    for poly in obj.data.polygons:
        poly.use_smooth = True

    return obj


def build_owl_geometry(materials):
    """Build all owl meshes with anatomical proportions and vertex groups."""
    parts = []

    # 1. Torso / Body: Egg-like oval volume centered at z=0.85
    body = create_mesh_part(
        name="Owl_Body",
        primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
        material=materials["Body"],
        group_name="Spine",
        location=(0, 0, 0.85),
        scale=(0.42, 0.38, 0.58),
        segments=32,
        ring_count=18
    )
    # Split vertex weights: lower vertices to Spine, upper to Chest
    vg_chest = body.vertex_groups.new(name="Chest")
    vg_spine = body.vertex_groups["Spine"]
    for v in body.data.vertices:
        # Blend weights smoothly based on z height
        z = v.co.z
        if z > 0.85:
            factor = min(1.0, (z - 0.85) / 0.40)
            vg_chest.add([v.index], factor, 'REPLACE')
            vg_spine.add([v.index], 1.0 - factor, 'REPLACE')
        else:
            vg_spine.add([v.index], 1.0, 'REPLACE')
            vg_chest.add([v.index], 0.0, 'REPLACE')
    parts.append(body)

    # 2. Chest Bib / Breast: Protruding soft front patch facing -Y
    chest_bib = create_mesh_part(
        name="Owl_ChestBib",
        primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
        material=materials["ChestBib"],
        group_name="Chest",
        location=(0, -0.20, 0.88),
        scale=(0.32, 0.18, 0.40),
        segments=24,
        ring_count=16
    )
    parts.append(chest_bib)

    # 3. Head: Spherical owl head perched on neck at z=1.52
    head = create_mesh_part(
        name="Owl_Head",
        primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
        material=materials["Body"],
        group_name="Head",
        location=(0, -0.02, 1.52),
        scale=(0.38, 0.35, 0.35),
        segments=32,
        ring_count=20
    )
    parts.append(head)

    # 4. Facial Discs (Left & Right halves forming the heart-shaped Strigiform mask)
    disc_l = create_mesh_part(
        name="Owl_FacialDisc_L",
        primitive_func=bpy.ops.mesh.primitive_circle_add,
        material=materials["FacialDisc"],
        group_name="Head",
        location=(0.14, -0.32, 1.55),
        rotation=(math.radians(82), math.radians(-6), math.radians(12)),
        scale=(0.19, 0.22, 0.04),
        fill_type='NGON',
        vertices=24
    )
    disc_r = create_mesh_part(
        name="Owl_FacialDisc_R",
        primitive_func=bpy.ops.mesh.primitive_circle_add,
        material=materials["FacialDisc"],
        group_name="Head",
        location=(-0.14, -0.32, 1.55),
        rotation=(math.radians(82), math.radians(6), math.radians(-12)),
        scale=(0.19, 0.22, 0.04),
        fill_type='NGON',
        vertices=24
    )
    parts.extend([disc_l, disc_r])

    # 5. Facial Disc Rim Rings (Dark framing rim)
    rim_l = create_mesh_part(
        name="Owl_FacialDiscRim_L",
        primitive_func=bpy.ops.mesh.primitive_torus_add,
        material=materials["FacialDiscRim"],
        group_name="Head",
        location=(0.14, -0.32, 1.55),
        rotation=(math.radians(82), math.radians(-6), math.radians(12)),
        scale=(0.19, 0.22, 0.04),
        major_radius=1.0,
        minor_radius=0.08,
        major_segments=24,
        minor_segments=8
    )
    rim_r = create_mesh_part(
        name="Owl_FacialDiscRim_R",
        primitive_func=bpy.ops.mesh.primitive_torus_add,
        material=materials["FacialDiscRim"],
        group_name="Head",
        location=(-0.14, -0.32, 1.55),
        rotation=(math.radians(82), math.radians(6), math.radians(-12)),
        scale=(0.19, 0.22, 0.04),
        major_radius=1.0,
        minor_radius=0.08,
        major_segments=24,
        minor_segments=8
    )
    parts.extend([rim_l, rim_r])

    # 6. Eyes: Golden Irises + Jet Black Pupils + High-Gloss Corneas
    for side, sign in [("L", 1), ("R", -1)]:
        eye_center = (sign * 0.14, -0.33, 1.55)

        # Golden Iris disc
        iris = create_mesh_part(
            name=f"Owl_Iris_{side}",
            primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
            material=materials["Iris"],
            group_name="Head",
            location=eye_center,
            rotation=(math.radians(84), 0, sign * math.radians(10)),
            scale=(0.082, 0.05, 0.082),
            segments=24,
            ring_count=12
        )

        # Center Pupil
        pupil = create_mesh_part(
            name=f"Owl_Pupil_{side}",
            primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
            material=materials["Pupil"],
            group_name="Head",
            location=(eye_center[0], eye_center[1] - 0.015, eye_center[2]),
            rotation=(math.radians(84), 0, sign * math.radians(10)),
            scale=(0.046, 0.04, 0.046),
            segments=20,
            ring_count=10
        )

        # Cornea gloss bubble
        cornea = create_mesh_part(
            name=f"Owl_Cornea_{side}",
            primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
            material=materials["Cornea"],
            group_name="Head",
            location=(eye_center[0], eye_center[1] - 0.01, eye_center[2]),
            scale=(0.088, 0.06, 0.088),
            segments=24,
            ring_count=12
        )
        parts.extend([iris, pupil, cornea])

    # 7. Eyelids: Left and Right blink caps
    eyelid_l = create_mesh_part(
        name="Owl_Eyelid_L",
        primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
        material=materials["Body"],
        group_name="Eyelid_L",
        location=(0.14, -0.32, 1.58),
        scale=(0.095, 0.07, 0.065),
        segments=20,
        ring_count=12
    )
    eyelid_r = create_mesh_part(
        name="Owl_Eyelid_R",
        primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
        material=materials["Body"],
        group_name="Eyelid_R",
        location=(-0.14, -0.32, 1.58),
        scale=(0.095, 0.07, 0.065),
        segments=20,
        ring_count=12
    )
    parts.extend([eyelid_l, eyelid_r])

    # 8. Beak Upper: Downward curved raptor hook
    beak_upper = create_mesh_part(
        name="Owl_Beak_Upper",
        primitive_func=bpy.ops.mesh.primitive_cone_add,
        material=materials["Beak"],
        group_name="Beak_Upper",
        location=(0, -0.38, 1.48),
        rotation=(math.radians(-70), 0, 0),
        scale=(0.055, 0.065, 0.12),
        vertices=16
    )
    parts.append(beak_upper)

    # 9. Beak Lower: Articulated Mandible (pivots cleanly on Beak_Lower bone!)
    beak_lower = create_mesh_part(
        name="Owl_Beak_Lower",
        primitive_func=bpy.ops.mesh.primitive_cone_add,
        material=materials["Beak"],
        group_name="Beak_Lower",
        location=(0, -0.35, 1.41),
        rotation=(math.radians(-110), 0, 0),
        scale=(0.045, 0.055, 0.09),
        vertices=16
    )
    parts.append(beak_lower)

    # 10. Ear Tufts (Plumicorns): Left & Right flared feather horns
    ear_l = create_mesh_part(
        name="Owl_EarTuft_L",
        primitive_func=bpy.ops.mesh.primitive_cone_add,
        material=materials["Body"],
        group_name="EarTuft_L",
        location=(0.20, -0.06, 1.86),
        rotation=(math.radians(18), math.radians(22), math.radians(-15)),
        scale=(0.05, 0.035, 0.18),
        vertices=12
    )
    ear_r = create_mesh_part(
        name="Owl_EarTuft_R",
        primitive_func=bpy.ops.mesh.primitive_cone_add,
        material=materials["Body"],
        group_name="EarTuft_R",
        location=(-0.20, -0.06, 1.86),
        rotation=(math.radians(18), math.radians(-22), math.radians(15)),
        scale=(0.05, 0.035, 0.18),
        vertices=12
    )
    parts.extend([ear_l, ear_r])

    # 11. Wings: Folded side wings along torso
    wing_l = create_mesh_part(
        name="Owl_Wing_L",
        primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
        material=materials["Wings"],
        group_name="Chest",
        location=(0.38, 0.02, 0.82),
        rotation=(math.radians(12), math.radians(-8), math.radians(-10)),
        scale=(0.10, 0.28, 0.50),
        segments=24,
        ring_count=14
    )
    wing_r = create_mesh_part(
        name="Owl_Wing_R",
        primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
        material=materials["Wings"],
        group_name="Chest",
        location=(-0.38, 0.02, 0.82),
        rotation=(math.radians(12), math.radians(8), math.radians(10)),
        scale=(0.10, 0.28, 0.50),
        segments=24,
        ring_count=14
    )
    parts.extend([wing_l, wing_r])

    # 12. Tail: Downward folded rear wedge
    tail = create_mesh_part(
        name="Owl_Tail",
        primitive_func=bpy.ops.mesh.primitive_cone_add,
        material=materials["Wings"],
        group_name="Spine",
        location=(0, 0.28, 0.44),
        rotation=(math.radians(35), 0, 0),
        scale=(0.20, 0.08, 0.35),
        vertices=16
    )
    parts.append(tail)

    # 13. Claws / Talons perched on branch
    for sign in [1, -1]:
        claw_center = (sign * 0.16, -0.08, 0.28)
        claw = create_mesh_part(
            name=f"Owl_Talon_{'L' if sign > 0 else 'R'}",
            primitive_func=bpy.ops.mesh.primitive_uv_sphere_add,
            material=materials["Talons"],
            group_name="Root",
            location=claw_center,
            rotation=(math.radians(45), 0, 0),
            scale=(0.09, 0.12, 0.06),
            segments=16,
            ring_count=10
        )
        parts.append(claw)

    # 14. Perched Wooden Branch
    branch = create_mesh_part(
        name="Owl_Perch_Branch",
        primitive_func=bpy.ops.mesh.primitive_cylinder_add,
        material=materials["Branch"],
        group_name="Root",
        location=(0, -0.08, 0.22),
        rotation=(0, math.radians(90), 0),
        scale=(0.09, 0.09, 0.90),
        vertices=24
    )
    parts.append(branch)

    return parts


def build_armature():
    """Create complete hierarchical armature rig with precise pivots for speech & strigiform biomechanics."""
    arm_data = bpy.data.armatures.new("Owl_Armature_Data")
    arm_obj = bpy.data.objects.new("Owl_Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    # Enter Edit Mode to create bones
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_data.edit_bones

    # 1. Root
    bone_root = edit_bones.new("Root")
    bone_root.head = Vector((0.0, 0.0, 0.0))
    bone_root.tail = Vector((0.0, 0.0, 0.30))

    # 2. Spine (Mid body)
    bone_spine = edit_bones.new("Spine")
    bone_spine.head = Vector((0.0, 0.0, 0.30))
    bone_spine.tail = Vector((0.0, 0.0, 0.85))
    bone_spine.parent = bone_root

    # 3. Chest (Upper body / breath expansion)
    bone_chest = edit_bones.new("Chest")
    bone_chest.head = Vector((0.0, 0.0, 0.85))
    bone_chest.tail = Vector((0.0, 0.0, 1.30))
    bone_chest.parent = bone_spine

    # 4. Neck (Strigiform pivot base)
    bone_neck = edit_bones.new("Neck")
    bone_neck.head = Vector((0.0, 0.0, 1.30))
    bone_neck.tail = Vector((0.0, -0.02, 1.48))
    bone_neck.parent = bone_chest

    # 5. Head (Saccadic yaw/pitch snaps, rolls, nodding)
    bone_head = edit_bones.new("Head")
    bone_head.head = Vector((0.0, -0.02, 1.48))
    bone_head.tail = Vector((0.0, -0.02, 1.85))
    bone_head.parent = bone_neck

    # 6. Upper Beak
    bone_beak_u = edit_bones.new("Beak_Upper")
    bone_beak_u.head = Vector((0.0, -0.28, 1.50))
    bone_beak_u.tail = Vector((0.0, -0.42, 1.46))
    bone_beak_u.parent = bone_head

    # 7. Lower Beak (Mandible - speech viseme driver)
    bone_beak_l = edit_bones.new("Beak_Lower")
    bone_beak_l.head = Vector((0.0, -0.26, 1.44))
    bone_beak_l.tail = Vector((0.0, -0.40, 1.40))
    bone_beak_l.parent = bone_head

    # 8. Eyelids (Blinking drivers)
    bone_eye_l = edit_bones.new("Eyelid_L")
    bone_eye_l.head = Vector((0.14, -0.30, 1.58))
    bone_eye_l.tail = Vector((0.14, -0.38, 1.58))
    bone_eye_l.parent = bone_head

    bone_eye_r = edit_bones.new("Eyelid_R")
    bone_eye_r.head = Vector((-0.14, -0.30, 1.58))
    bone_eye_r.tail = Vector((-0.14, -0.38, 1.58))
    bone_eye_r.parent = bone_head

    # 9. Ear Tufts (Plumicorns - emotive expressive twitches)
    bone_tuft_l = edit_bones.new("EarTuft_L")
    bone_tuft_l.head = Vector((0.20, -0.06, 1.80))
    bone_tuft_l.tail = Vector((0.28, 0.02, 2.05))
    bone_tuft_l.parent = bone_head

    bone_tuft_r = edit_bones.new("EarTuft_R")
    bone_tuft_r.head = Vector((-0.20, -0.06, 1.80))
    bone_tuft_r.tail = Vector((-0.28, 0.02, 2.05))
    bone_tuft_r.parent = bone_head

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


def join_and_skin(mesh_parts, arm_obj):
    """Combine all geometry into a single clean character mesh parented to the armature."""
    # Ensure all objects are in OBJECT mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Select all mesh parts
    bpy.ops.object.select_all(action='DESELECT')
    for p in mesh_parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = mesh_parts[0]

    # Join meshes
    bpy.ops.object.join()
    unified_mesh = bpy.context.active_object
    unified_mesh.name = "Owl_Mesh"

    # Add Armature Modifier
    arm_mod = unified_mesh.modifiers.new(name="Armature", type='ARMATURE')
    arm_mod.object = arm_obj
    arm_mod.use_vertex_groups = True

    # Parent mesh to Armature
    unified_mesh.parent = arm_obj

    return unified_mesh


def export_glb(output_path, arm_obj, mesh_obj):
    """Export the rigged owl asset to GLB."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Select both Armature and Mesh
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    print(f"[*] Exporting rigged 3D Owl GLB to: {output_path}")

    # glTF export kwargs with compatibility
    export_kwargs = {
        "filepath": output_path,
        "export_format": 'GLB',
        "use_selection": True,
        "export_apply": False,
        "export_animations": True,
        "export_skins": True,
        "export_materials": 'EXPORT',
        "export_lights": False,
        "export_cameras": False,
    }

    try:
        bpy.ops.export_scene.gltf(**export_kwargs)
        print(f"[+] Successfully exported GLB: {output_path} ({os.path.getsize(output_path):,} bytes)")
    except Exception as e:
        print(f"[!] Warning: gltf export failed with default flags: {e}. Retrying simplified...")
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')


def main():
    if not IN_BLENDER:
        print("[!] ERROR: This script must be run inside Blender:")
        print("    blender -b -P scripts/build_owl_asset.py -- --output assets/owl_base.glb")
        sys.exit(1)

    args = parse_args()
    print("[*] Starting Procedural 3D Rigged Owl Asset Generation...")

    clean_scene()
    materials = setup_materials()
    print(f"[*] Configured {len(materials)} PBR materials.")

    mesh_parts = build_owl_geometry(materials)
    print(f"[*] Built {len(mesh_parts)} anatomical mesh elements.")

    armature = build_armature()
    print(f"[*] Assembled 11-bone Strigiform Armature Rig.")

    owl_mesh = join_and_skin(mesh_parts, armature)
    print(f"[*] Bound and skinned mesh ({len(owl_mesh.data.vertices)} vertices).")

    export_glb(args.output, armature, owl_mesh)
    print("[+] Procedural 3D Owl Asset Generation Complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Blender Headless Speech Animator & Renderer for 3D Rigged Owl.
=============================================================

This script runs inside headless Blender (`blender -b -P scripts/animate_and_render.py -- [args]`).
It loads the rigged 3D owl asset, animates biomechanically authentic strigiform motions,
stages cinematic 3-point lighting and 85mm portrait camera, and renders the final video.

Animation Mechanics:
  1. Viseme-Driven Lower Mandible (Beak_Lower):
     - Viseme angle mapping: X=0°, A=0°, B=-7°, C=-16°, D=-28°, E=-13°, F=-12°, G=-8°, H=-20°.
     - Dynamic scaling with audio RMS envelope: Angle = Base * (0.4 + 0.6 * RMS_norm).
  2. Strigiform Saccadic Head Snaps:
     - Fast 2-3 frame rapid gaze shifts followed by rock-steady stabilization holds (1.5-3.5s).
     - Yaw (+/-18° to +/-35°) and Pitch micro-adjustments.
  3. Inquisitive Roll Tilts:
     - Characteristic head rolls (+/-15° to +/-22°) during phrase pauses and clause breaks.
  4. Speech Emphasis Nods:
     - Dynamic down-pitches (-4° to -6°) triggered on vocal stress peaks (high RMS).
  5. Blinking (Eyelid_L, Eyelid_R):
     - Rapid 2-frame close, 1-frame hold, 3-frame ease open every 3-5 seconds and post-saccade.
  6. Organic Respiration (Chest & Spine):
     - Subtle 3.2s sinusoidal breathing expansion and gentle posture sway.
"""

import sys
import os
import math
import json
import random
import shutil
import argparse
import subprocess

try:
    import bpy
    from mathutils import Vector, Euler, Matrix
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


# Viseme base pitch angles in degrees (negative tilts mandible downward to open)
VISEME_ANGLES = {
    "X": 0.0,    # Closed / resting
    "A": -8.0,   # Slight mouth shape
    "B": -18.0,  # Consonants
    "C": -36.0,  # Moderate open vowels
    "D": -52.0,  # Wide open vowels (AA, AO, AW)
    "E": -28.0,  # Mid vowels
    "F": -22.0,  # Fricatives
    "G": -16.0,  # Palatal
    "H": -45.0,  # Wide open breath (W, OO)
}


def parse_args():
    argv = sys.argv
    if "--" in argv:
        args_to_parse = argv[argv.index("--") + 1:]
    else:
        args_to_parse = []

    parser = argparse.ArgumentParser(description="Blender Headless Speech Animator & Renderer")
    parser.add_argument("--glb", type=str, default="assets/owl_base.glb",
                        help="Path to rigged owl GLB asset")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to speech WAV audio file")
    parser.add_argument("--visemes", type=str, required=True,
                        help="Path to Rhubarb visemes JSON file")
    parser.add_argument("--envelope", type=str, default=None,
                        help="Path to audio RMS envelope JSON file")
    parser.add_argument("--output", type=str, default="outputs/owl_speech.mp4",
                        help="Path to output rendered MP4 file")
    parser.add_argument("--engine", type=str, default="CYCLES",
                        choices=["CYCLES", "BLENDER_EEVEE", "EEVEE", "WORKBENCH"],
                        help="Render engine to use")
    parser.add_argument("--device", type=str, default="GPU",
                        choices=["GPU", "CPU"],
                        help="Compute device for Cycles")
    parser.add_argument("--samples", type=int, default=32,
                        help="Number of render samples per frame")
    parser.add_argument("--resolution", type=str, default="1080x1080",
                        help="Output resolution WIDTHxHEIGHT (e.g. 1080x1080, 1280x720)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Target frame rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for procedural saccadic motions")
    return parser.parse_args(args_to_parse)


def clean_scene():
    """Clear default scene objects."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        bpy.data.armatures.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)


def import_owl_model(glb_path):
    """Import the rigged owl asset and locate armature and pose bones."""
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"Model GLB not found at: {glb_path}")

    print(f"[*] Importing owl asset: {glb_path}")
    bpy.ops.import_scene.gltf(filepath=glb_path)

    arm_obj = None
    mesh_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            arm_obj = obj
        elif obj.type == 'MESH':
            mesh_obj = obj

    if not arm_obj:
        raise RuntimeError("No Armature found in imported GLB file.")

    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    # Set all pose bones to XYZ Euler rotation mode for clean curve editing
    for pbone in arm_obj.pose.bones:
        pbone.rotation_mode = 'XYZ'

    print(f"[+] Loaded Armature: '{arm_obj.name}' with {len(arm_obj.pose.bones)} pose bones.")
    return arm_obj, mesh_obj


def load_visemes_and_envelope(visemes_path, envelope_path):
    """Load mouth cues and optional RMS envelope."""
    with open(visemes_path, "r", encoding="utf-8") as f:
        visemes_data = json.load(f)

    mouth_cues = visemes_data.get("mouthCues", [])
    duration = visemes_data.get("metadata", {}).get("duration", 0.0)

    envelope_map = {}
    if envelope_path and os.path.exists(envelope_path):
        with open(envelope_path, "r", encoding="utf-8") as f:
            env_data = json.load(f)
            for item in env_data:
                envelope_map[round(item.get("time", 0.0), 2)] = item.get("rms", 0.0)

    # If duration wasn't in metadata, derive from last mouth cue
    if duration <= 0.0 and mouth_cues:
        duration = mouth_cues[-1].get("end", 0.0)

    return mouth_cues, envelope_map, duration


def get_rms_at_time(envelope_map, t):
    """Query normalized RMS at time t with nearest neighbor lookup."""
    if not envelope_map:
        return 0.5
    rounded_t = round(t, 2)
    return envelope_map.get(rounded_t, 0.4)


def animate_owl(arm_obj, mouth_cues, envelope_map, duration, fps=30.0, seed=42):
    """Generate all F-curve animations on the armature pose bones."""
    random.seed(seed)
    total_frames = max(1, int(math.ceil(duration * fps)))
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.render.fps = int(fps)

    # Clear any previous animation data to prevent quaternion track conflicts
    if arm_obj.animation_data:
        arm_obj.animation_data_clear()

    # Ensure all pose bones are strictly in XYZ Euler rotation mode so keyframing rotation_euler works
    for bone in arm_obj.pose.bones:
        bone.rotation_mode = 'XYZ'

    pose_bones = arm_obj.pose.bones
    b_beak = pose_bones.get("Beak_Lower")
    b_head = pose_bones.get("Head")
    b_neck = pose_bones.get("Neck")
    b_chest = pose_bones.get("Chest")
    b_spine = pose_bones.get("Spine")
    b_eye_l = pose_bones.get("Eyelid_L")
    b_eye_r = pose_bones.get("Eyelid_R")
    b_tuft_l = pose_bones.get("EarTuft_L")
    b_tuft_r = pose_bones.get("EarTuft_R")

    print(f"[*] Animating {total_frames} frames ({duration:.2f}s at {fps} fps)...")

    # --------------------------------------------------------------------------
    # 1. Beak Lower Viseme Animation (Speech)
    # --------------------------------------------------------------------------
    if b_beak:
        # Precompute target angle per frame
        for frame in range(1, total_frames + 1):
            t = (frame - 1) / fps

            # Find active mouth cue
            active_val = "X"
            for cue in mouth_cues:
                if cue["start"] <= t <= cue["end"]:
                    active_val = cue["value"]
                    break

            base_deg = VISEME_ANGLES.get(active_val, 0.0)
            rms = get_rms_at_time(envelope_map, t)

            # Dynamic RMS scaling: Angle = Base * (0.60 + 0.40 * RMS_norm)
            scaled_deg = base_deg * (0.60 + 0.40 * rms)
            pitch_rad = math.radians(scaled_deg)

            b_beak.rotation_euler = (pitch_rad, 0.0, 0.0)
            b_beak.keyframe_insert(data_path="rotation_euler", frame=frame)

    # --------------------------------------------------------------------------
    # 2. Respiration & Body Micro-sway (Spine & Chest)
    # --------------------------------------------------------------------------
    for frame in range(1, total_frames + 1):
        t = (frame - 1) / fps
        breath_phase = 2.0 * math.pi * t / 3.2

        if b_spine:
            spine_pitch = math.sin(breath_phase) * math.radians(0.7)
            b_spine.rotation_euler = (spine_pitch, 0.0, 0.0)
            b_spine.keyframe_insert(data_path="rotation_euler", frame=frame)

        if b_chest:
            chest_pitch = math.sin(breath_phase + 0.3) * math.radians(0.9)
            chest_scale = 1.0 + 0.008 * math.sin(breath_phase)
            b_chest.rotation_euler = (chest_pitch, 0.0, 0.0)
            b_chest.scale = (chest_scale, chest_scale, chest_scale)
            b_chest.keyframe_insert(data_path="rotation_euler", frame=frame)
            b_chest.keyframe_insert(data_path="scale", frame=frame)

    # --------------------------------------------------------------------------
    # 3. Strigiform Saccadic Head Snaps & Inquisitive Tilts (Head & Neck)
    # --------------------------------------------------------------------------
    if b_head:
        saccade_targets = []
        cur_t = 0.0
        cur_yaw = 0.0
        cur_pitch = 0.0
        cur_roll = 0.0

        # Initial rest frame
        saccade_targets.append((1, 0.0, 0.0, 0.0))

        frame = 1
        while frame < total_frames:
            # Hold steady for 1.6 to 3.2 seconds (Strigiform lock)
            hold_sec = random.uniform(1.6, 3.2)
            hold_frames = int(hold_sec * fps)
            next_frame = min(total_frames, frame + hold_frames)

            # Hold current pose
            saccade_targets.append((next_frame, cur_pitch, cur_roll, cur_yaw))

            if next_frame >= total_frames:
                break

            # Fast 2-frame saccadic snap
            snap_frames = random.choice([2, 3])
            snap_end = min(total_frames, next_frame + snap_frames)

            # New gaze direction
            target_yaw = math.radians(random.choice([
                0.0, 0.0,  # Center bias
                random.uniform(8.0, 22.0),   # Look Left
                random.uniform(-22.0, -8.0), # Look Right
                random.uniform(25.0, 38.0),  # Wide Left
                random.uniform(-38.0, -25.0) # Wide Right
            ]))
            target_pitch = math.radians(random.uniform(-4.0, 5.0))

            # Inquisitive roll tilt: trigger occasionally, especially when turning
            if random.random() < 0.40:
                target_roll = math.radians(random.choice([-1, 1]) * random.uniform(14.0, 22.0))
            else:
                target_roll = 0.0

            cur_pitch = target_pitch
            cur_roll = target_roll
            cur_yaw = target_yaw

            saccade_targets.append((snap_end, cur_pitch, cur_roll, cur_yaw))
            frame = snap_end

        # Insert saccade keyframes
        for f_idx, pitch, roll, yaw in saccade_targets:
            b_head.rotation_euler = (pitch, roll, yaw)
            b_head.keyframe_insert(data_path="rotation_euler", frame=f_idx)

        # Overlay speech emphasis down-nods on high RMS frames
        for frame in range(1, total_frames + 1):
            t = (frame - 1) / fps
            rms = get_rms_at_time(envelope_map, t)
            if rms > 0.65 and (frame % 8 == 0):
                # Apply rhythmic downward nod (-4° to -6°)
                nod_deg = -5.0 * (rms - 0.4)
                nod_rad = math.radians(nod_deg)
                # Sample current evaluated pitch and add nod
                b_head.rotation_euler[0] += nod_rad
                b_head.keyframe_insert(data_path="rotation_euler", index=0, frame=frame)

    # --------------------------------------------------------------------------
    # 4. Blinking (Eyelid_L & Eyelid_R)
    # --------------------------------------------------------------------------
    blink_bones = [b for b in [b_eye_l, b_eye_r] if b is not None]
    if blink_bones:
        blink_frame = 20
        while blink_frame < total_frames - 10:
            # Neutral open
            for b in blink_bones:
                b.scale = (1.0, 1.0, 1.0)
                b.rotation_euler = (0.0, 0.0, 0.0)
                b.keyframe_insert(data_path="scale", frame=blink_frame)
                b.keyframe_insert(data_path="rotation_euler", frame=blink_frame)

            # Snap closed (2 frames)
            close_frame = blink_frame + 2
            for b in blink_bones:
                b.scale = (1.0, 1.0, 0.05)
                b.rotation_euler = (math.radians(-65.0), 0.0, 0.0)
                b.keyframe_insert(data_path="scale", frame=close_frame)
                b.keyframe_insert(data_path="rotation_euler", frame=close_frame)

            # Hold closed 1 frame
            hold_frame = close_frame + 1
            for b in blink_bones:
                b.keyframe_insert(data_path="scale", frame=hold_frame)
                b.keyframe_insert(data_path="rotation_euler", frame=hold_frame)

            # Ease open (3 frames)
            open_frame = hold_frame + 3
            for b in blink_bones:
                b.scale = (1.0, 1.0, 1.0)
                b.rotation_euler = (0.0, 0.0, 0.0)
                b.keyframe_insert(data_path="scale", frame=open_frame)
                b.keyframe_insert(data_path="rotation_euler", frame=open_frame)

            # Next blink in 3.5 to 5.5 seconds
            blink_interval = int(random.uniform(3.5, 5.5) * fps)
            blink_frame = open_frame + blink_interval

    # --------------------------------------------------------------------------
    # 5. Ear Tufts Flare (Plumicorns)
    # --------------------------------------------------------------------------
    for side, tuft in [("L", b_tuft_l), ("R", b_tuft_r)]:
        if not tuft:
            continue
        sign = 1 if side == "L" else -1
        for frame in range(1, total_frames + 1, 15):
            t = (frame - 1) / fps
            rms = get_rms_at_time(envelope_map, t)
            flare = math.radians(sign * (3.0 + 7.0 * rms))
            tuft.rotation_euler = (0.0, flare, 0.0)
            tuft.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Set interpolation mode to BEZIER for smooth organic transitions
    if arm_obj.animation_data and arm_obj.animation_data.action:
        for fcurve in arm_obj.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'BEZIER'
                kf.easing = 'AUTO'

    print("[+] Armature procedural animation generated successfully.")


def setup_lighting_and_camera(resolution="1080x1080"):
    """Set up cinematic 3-point portrait lighting and 85mm camera."""
    scene = bpy.context.scene

    # Parse resolution
    try:
        width, height = [int(v) for v in resolution.lower().split("x")]
    except Exception:
        width, height = 1080, 1080

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    # 1. Camera: 85mm portrait telephoto
    cam_data = bpy.data.cameras.new("Portrait_Camera")
    cam_data.lens = 85.0
    cam_data.sensor_width = 36.0
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 2.45
    cam_data.dof.aperture_fstop = 3.5

    cam_obj = bpy.data.objects.new("Portrait_Camera", cam_data)
    cam_obj.location = (0.0, -2.45, 1.48)
    cam_obj.rotation_euler = (math.radians(88.0), 0.0, 0.0)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # 2. Key Light (Warm golden portrait light, 45° front-left)
    key_data = bpy.data.lights.new("Key_Light", 'AREA')
    key_data.energy = 420.0
    key_data.color = (1.0, 0.92, 0.82)
    key_data.size = 0.8
    key_obj = bpy.data.objects.new("Key_Light", key_data)
    key_obj.location = (-1.6, -1.8, 2.3)
    key_obj.rotation_euler = (math.radians(55.0), math.radians(-25.0), math.radians(-35.0))
    scene.collection.objects.link(key_obj)

    # 3. Fill Light (Cool skylight, 45° front-right)
    fill_data = bpy.data.lights.new("Fill_Light", 'AREA')
    fill_data.energy = 160.0
    fill_data.color = (0.82, 0.90, 1.0)
    fill_data.size = 1.2
    fill_obj = bpy.data.objects.new("Fill_Light", fill_data)
    fill_obj.location = (1.8, -1.6, 1.8)
    fill_obj.rotation_euler = (math.radians(50.0), math.radians(20.0), math.radians(45.0))
    scene.collection.objects.link(fill_obj)

    # 4. Rim / Hair Light (Crisp edge light behind owl)
    rim_data = bpy.data.lights.new("Rim_Light", 'SPOT')
    rim_data.energy = 550.0
    rim_data.color = (0.95, 0.95, 1.0)
    rim_data.spot_size = math.radians(55.0)
    rim_data.spot_blend = 0.3
    rim_obj = bpy.data.objects.new("Rim_Light", rim_data)
    rim_obj.location = (0.0, 1.8, 2.5)
    rim_obj.rotation_euler = (math.radians(-135.0), 0.0, 0.0)
    scene.collection.objects.link(rim_obj)

    # 5. Atmospheric dark world background
    world = bpy.data.worlds.new("Atmospheric_World")
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.02, 0.025, 0.035, 1.0)
        bg_node.inputs["Strength"].default_value = 0.6
    scene.world = world

    print(f"[+] Lighting & 85mm Camera configured at {width}x{height}.")


def configure_render_engine(engine="CYCLES", device="GPU", samples=32):
    """Configure render engine with GPU acceleration when available."""
    scene = bpy.context.scene

    # Map engine names
    engine_upper = engine.upper()
    if engine_upper in ["CYCLES"]:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = samples
        scene.cycles.preview_samples = min(16, samples)

        # Try to enable GPU compute
        if device.upper() == "GPU":
            try:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                # Find available device types: OPTIX > CUDA > METAL > NONE
                device_type = None
                for candidate in ['OPTIX', 'CUDA', 'METAL']:
                    try:
                        prefs.get_devices()
                        devices = prefs.devices
                        usable = [d for d in devices if d.type == candidate]
                        if usable:
                            device_type = candidate
                            break
                    except Exception:
                        pass

                if device_type:
                    prefs.compute_device_type = device_type
                    for d in prefs.devices:
                        d.use = (d.type == device_type)
                    scene.cycles.device = 'GPU'
                    print(f"[+] Cycles GPU Acceleration enabled: {device_type}")
                else:
                    scene.cycles.device = 'CPU'
                    print("[*] No GPU compute device detected. Using CPU for Cycles.")
            except Exception as e:
                scene.cycles.device = 'CPU'
                print(f"[*] Fallback to CPU: {e}")
        else:
            scene.cycles.device = 'CPU'

        # Denoising
        try:
            scene.cycles.use_denoising = True
        except Exception:
            pass

    elif engine_upper in ["EEVEE", "BLENDER_EEVEE"]:
        # Check for Blender 4.2+ EEVEE_NEXT or standard BLENDER_EEVEE
        available = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
        if 'BLENDER_EEVEE_NEXT' in available:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        elif 'BLENDER_EEVEE' in available:
            scene.render.engine = 'BLENDER_EEVEE'
        else:
            scene.render.engine = 'BLENDER_EEVEE'
        print(f"[+] EEVEE engine selected: {scene.render.engine}")

    elif engine_upper == "WORKBENCH":
        scene.render.engine = 'BLENDER_WORKBENCH'
        print("[+] Workbench engine selected.")


def render_and_mux(output_mp4, audio_wav, fps=30.0):
    """Render frame sequence and mux with audio using FFmpeg."""
    scene = bpy.context.scene
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(output_mp4)), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Set frame output to PNG
    scene.render.filepath = os.path.join(temp_dir, "frame_")
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    total_frames = scene.frame_end - scene.frame_start + 1
    print(f"[*] Rendering {total_frames} frames to {temp_dir}...")

    # Render animation frames
    bpy.ops.render.render(animation=True)
    print("[+] Frame rendering complete.")

    # Mux with FFmpeg
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)

    print(f"[*] Muxing frames with audio: {audio_wav} -> {output_mp4}")
    cmd = [
        ffmpeg_bin, "-y",
        "-r", str(fps),
        "-i", frame_pattern,
        "-i", audio_wav,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_mp4
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg muxing failed: {res.stderr}")

    # Clean up temp frames
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"[!] Note: temp frames cleanup notice: {e}")

    file_size = os.path.getsize(output_mp4)
    print(f"[✓] Render & Mux Complete: {output_mp4} ({file_size:,} bytes)")


def main():
    if not IN_BLENDER:
        print("[!] ERROR: This script must be run inside Blender:")
        print("    blender -b -P scripts/animate_and_render.py -- [options]")
        sys.exit(1)

    args = parse_args()
    print("=" * 70)
    print("OWL 3D SPEECH — HEADLESS ANIMATOR & RENDERER")
    print("=" * 70)

    clean_scene()
    arm_obj, mesh_obj = import_owl_model(args.glb)
    mouth_cues, envelope_map, duration = load_visemes_and_envelope(args.visemes, args.envelope)

    animate_owl(
        arm_obj=arm_obj,
        mouth_cues=mouth_cues,
        envelope_map=envelope_map,
        duration=duration,
        fps=args.fps,
        seed=args.seed
    )

    setup_lighting_and_camera(resolution=args.resolution)
    configure_render_engine(engine=args.engine, device=args.device, samples=args.samples)
    render_and_mux(args.output, args.audio, fps=args.fps)
    print("[+] Speech animation pipeline finished successfully!")


if __name__ == "__main__":
    main()

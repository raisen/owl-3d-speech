#!/usr/bin/env python3
"""
Owl 3D Speech — Master Pipeline Orchestrator CLI.
================================================

Executes the complete end-to-end 3D speech animation workflow:
  1. Procedural 3D Rigged Owl Asset Generation (if not already built).
  2. Neural Voice Synthesis (Edge-TTS) & Viseme Alignment (Rhubarb / Algorithmic).
  3. Headless Blender Animation & GPU Cycles/EEVEE Rendering.
  4. Video & Audio Muxing to web-optimized MP4.

Usage:
  # Using poem preset:
  python3 run_pipeline.py --poem the_owl --voice ryan --engine CYCLES

  # Custom text:
  python3 run_pipeline.py --text "Wisdom comes not from words, but from the silence between them." --voice christopher

  # Using custom audio:
  python3 run_pipeline.py --audio my_audio.wav --output outputs/custom_speech.mp4

  # Build 3D rigged GLB asset only:
  python3 run_pipeline.py --generate_asset_only
"""

import sys
import os
import time
import shutil
import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


POEM_PRESETS = {
    "the_owl": (
        "When cats run home and light is come,\n"
        "And dew is cold upon the ground,\n"
        "And the far-off stream is dumb,\n"
        "And the whirring sail goes round,\n"
        "Alone and warming his five wits,\n"
        "The white owl in the belfry sits."
    ),
    "if": (
        "If you can keep your head when all about you\n"
        "Are losing theirs and blaming it on you,\n"
        "If you can trust yourself when all men doubt you,\n"
        "But make allowance for their doubting too;\n"
        "If you can wait and not be tired by waiting,\n"
        "And yet don't look too good, nor talk too wise."
    ),
    "wise_old_owl": (
        "A wise old owl sat in an oak,\n"
        "The more he heard, the less he spoke;\n"
        "The less he spoke, the more he heard;\n"
        "Why aren't we all like that wise old bird?"
    ),
    "owl_and_pussycat": (
        "The Owl and the Pussy-cat went to sea\n"
        "In a beautiful pea-green boat,\n"
        "They took some honey, and plenty of money,\n"
        "Wrapped up in a five-pound note.\n"
        "The Owl looked up to the stars above,\n"
        "And sang to a small guitar."
    )
}


def find_blender(custom_path=None):
    """Find a usable Blender executable across platforms."""
    candidates = [
        custom_path,
        os.environ.get("BLENDER_BIN"),
        shutil.which("blender"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/opt/blender/blender"
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def find_rhubarb(custom_path=None):
    """Find Rhubarb Lip Sync binary if present."""
    candidates = [
        custom_path,
        os.environ.get("RHUBARB_BIN"),
        shutil.which("rhubarb"),
        str(PROJECT_ROOT / "bin" / "rhubarb"),
        "/usr/local/bin/rhubarb",
        "/opt/rhubarb/rhubarb"
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Owl 3D Speech — Master Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group_input = parser.add_mutually_exclusive_group()
    group_input.add_argument("--text", type=str, default=None,
                             help="Text prompt for the owl to speak")
    group_input.add_argument("--poem", type=str, choices=list(POEM_PRESETS.keys()),
                             default="the_owl",
                             help="Preset poem for the owl to recite")
    group_input.add_argument("--audio", type=str, default=None,
                             help="Path to pre-existing WAV audio file")

    parser.add_argument("--voice", type=str, default="ryan",
                        help="Voice preset (ryan, christopher, guy, sonia, jenny) or full Edge-TTS voice")
    parser.add_argument("--model", type=str, default=str(ASSETS_DIR / "owl_base.glb"),
                        help="Path to rigged 3D owl GLB model")
    parser.add_argument("--output", type=str, default=str(OUTPUTS_DIR / "owl_speech.mp4"),
                        help="Path to output rendered video file")
    parser.add_argument("--engine", type=str, default="CYCLES",
                        choices=["CYCLES", "BLENDER_EEVEE", "EEVEE", "WORKBENCH"],
                        help="Blender render engine")
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"],
                        help="Cycles compute device")
    parser.add_argument("--samples", type=int, default=32,
                        help="Render samples per frame")
    parser.add_argument("--resolution", type=str, default="1080x1080",
                        help="Output resolution WIDTHxHEIGHT")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video frame rate")
    parser.add_argument("--blender_bin", type=str, default=None,
                        help="Path to Blender binary")
    parser.add_argument("--rhubarb_bin", type=str, default=None,
                        help="Path to Rhubarb binary")
    parser.add_argument("--generate_asset_only", action="store_true",
                        help="Only build procedural owl 3D GLB model and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 76)
    print("      🦉 OWL 3D SPEECH — PROCEDURAL ANIMATION PIPELINE")
    print("=" * 76)

    # 1. Check Blender binary
    blender_bin = find_blender(args.blender_bin)
    if not blender_bin:
        print("[!] ERROR: Blender binary not found.")
        print("    Please ensure Blender is installed and in your PATH, or pass --blender_bin /path/to/blender.")
        sys.exit(1)
    print(f"[✓] Using Blender: {blender_bin}")

    # 2. Build 3D Owl GLB Asset if not present
    model_path = Path(args.model)
    if not model_path.exists() or args.generate_asset_only:
        print(f"[*] Procedural 3D Owl model not found at '{model_path}'. Building from scratch...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        build_script = SCRIPTS_DIR / "build_owl_asset.py"
        cmd_build = [
            blender_bin, "-b", "-P", str(build_script),
            "--", "--output", str(model_path)
        ]
        res = subprocess.run(cmd_build, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0 or not model_path.exists():
            print(f"[!] Asset generation failed:\n{res.stdout}")
            sys.exit(1)
        print(f"[✓] 3D Rigged Owl Asset built: {model_path} ({model_path.stat().st_size:,} bytes)")

    if args.generate_asset_only:
        print("[+] Asset generation complete (--generate_asset_only specified). Exiting.")
        return

    # 3. Audio & Viseme Processing
    rhubarb_bin = find_rhubarb(args.rhubarb_bin)
    if rhubarb_bin:
        print(f"[✓] Using Rhubarb Lip Sync: {rhubarb_bin}")
    else:
        print("[*] Rhubarb binary not found. Pipeline will automatically employ Algorithmic Aligner.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_audio = OUTPUTS_DIR / "speech.wav"
    temp_visemes = OUTPUTS_DIR / "visemes.json"
    temp_envelope = OUTPUTS_DIR / "envelope.json"

    # Determine speech text
    speech_text = args.text
    if not speech_text and not args.audio:
        speech_text = POEM_PRESETS.get(args.poem, POEM_PRESETS["the_owl"])

    # Import and run rhubarb_sync
    from scripts.rhubarb_sync import process_audio_pipeline
    print("\n" + "-" * 76)
    print("STAGE 1: Audio Synthesis & Viseme Alignment")
    print("-" * 76)

    sync_result = process_audio_pipeline(
        text=speech_text,
        voice=args.voice,
        audio_path=args.audio,
        output_audio=str(temp_audio),
        output_visemes=str(temp_visemes),
        output_envelope=str(temp_envelope),
        rhubarb_bin=rhubarb_bin or "rhubarb",
        fps=args.fps
    )

    # 4. Blender Headless Speech Animation & Rendering
    print("\n" + "-" * 76)
    print("STAGE 2: Blender Headless GPU Animation & Rendering")
    print("-" * 76)

    render_script = SCRIPTS_DIR / "animate_and_render.py"
    output_video = Path(args.output)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd_render = [
        blender_bin, "-b", "-P", str(render_script),
        "--",
        "--glb", str(model_path),
        "--audio", str(temp_audio),
        "--visemes", str(temp_visemes),
        "--envelope", str(temp_envelope),
        "--output", str(output_video),
        "--engine", args.engine,
        "--device", args.device,
        "--samples", str(args.samples),
        "--resolution", args.resolution,
        "--fps", str(args.fps)
    ]

    print(f"[*] Executing Blender render...")
    render_proc = subprocess.Popen(cmd_render, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in iter(render_proc.stdout.readline, ""):
        line_str = line.strip()
        if "Fra:" in line_str or "Rendered" in line_str or "[*]" in line_str or "[+]" in line_str or "[✓]" in line_str:
            print(f"    {line_str}")

    render_proc.wait()
    if render_proc.returncode != 0 or not output_video.exists():
        print(f"[!] Blender render failed with exit code: {render_proc.returncode}")
        sys.exit(1)

    elapsed = time.time() - start_time
    file_size_mb = output_video.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 76)
    print("                     🎉 PIPELINE COMPLETE!")
    print("=" * 76)
    print(f"  • Video Output:     {output_video}")
    print(f"  • File Size:        {file_size_mb:.2f} MB")
    print(f"  • Duration:         {sync_result['duration']:.2f}s ({int(sync_result['duration'] * args.fps)} frames)")
    print(f"  • Mouth Cues:       {sync_result['cues_count']}")
    print(f"  • Render Engine:    {args.engine} ({args.device})")
    print(f"  • Total Time:       {elapsed:.1f} seconds")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()

# owl-3d-speech — Agent Rules

This project provides a headless Blender and Rhubarb-driven 3D speech animation pipeline that generates, rigs, and animates a stylized 3D owl with authentic strigiform biomechanics, neural TTS speech, and interactive Google Colab execution.

## Core Rules & Architecture Guidelines

1. **Self-Contained Execution**:
   - All scripts (`scripts/build_owl_asset.py`, `scripts/rhubarb_sync.py`, `scripts/animate_and_render.py`) and orchestrator `run_pipeline.py` must run both locally and inside cloud GPU environments (Google Colab, headless Linux/macOS).
   - The Colab notebook `Owl_3D_Speech.ipynb` must remain fully self-contained, handling portable Blender tarball installation, Rhubarb binary setup, and Python dependencies automatically.

2. **Strigiform Biomechanical Accuracy**:
   - Owls have tubular, non-rotatable eyes fixed in cranial sockets. Never rotate eye bones or eyeballs for gaze shifts; always execute saccadic head rotations (yaw/pitch) with 2–3 frame transitions followed by 1.5–3.5s stabilization holds.
   - Blinks must synchronize with saccadic settling and periodic intervals.
   - Maintain inquisitive roll tilts ($\pm 15^\circ$ to $\pm 22^\circ$) on phrase pauses.
   - Scale mandible pitch dynamically: $\text{Angle} = \text{BaseAngle}(\text{viseme}) \times (0.4 + 0.6 \cdot \text{RMS}_{norm})$.

3. **FFmpeg & Video Encoding Standard**:
   - Final MP4 exports must strictly use `-c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest` to guarantee seamless web browser, Colab notebook, and cross-platform playback without codec incompatibility.

4. **Preserve Presets & Formats**:
   - Retain all preset voices and poem scripts in `assets/sample_poem.txt` and `run_pipeline.py`.

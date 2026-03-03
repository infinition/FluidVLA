import re

f_path = r"c:\DEV\Workspace\active\coding\_AI RESEARCH\FluidVLA\experiments\step2d_so101_urdf\so101_urdf_viewer.py"
with open(f_path, "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(
    r"USAGE :.*?--random_weights",
    r"""USAGE:
  # Full viewer mode (Isaac Sim GUI)
  python so101_urdf_viewer.py \
      --checkpoint ../../checkpoints/step2c_isaac/best.pt \
      --urdf /path/to/so101.urdf \
      --episodes 5 \
      --show_gui

  # Headless mode (log only, no GUI)
  python so101_urdf_viewer.py \
      --checkpoint ../../checkpoints/step2a/best.pt \
      --urdf /path/to/so101.urdf \
      --headless

  # Without checkpoint (random weights, pipeline test)
  python so101_urdf_viewer.py --urdf /path/to/so101.urdf --random_weights""",
    text,
    flags=re.DOTALL
)

with open(f_path, "w", encoding="utf-8") as f:
    f.write(text)

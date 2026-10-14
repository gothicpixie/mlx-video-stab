"""Write the synthetic test clip (CC0, procedural, with ground truth): python -I bench/make_synth.py out/synth.mp4 [T] [W] [H]"""
import sys
from mlxstab import synth, media

out = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 150
W = int(sys.argv[3]) if len(sys.argv) > 3 else 640; H = int(sys.argv[4]) if len(sys.argv) > 4 else 360
vid, _ = synth.render(T=T, W=W, H=H)
e = media.Encoder(out, W, H, 30, crf=10, preset="medium")
for f in vid: e.write(f)
e.close(); print("wrote", out, vid.shape)

import argparse
from pathlib import Path
import cv2
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    args = parser.parse_args()

    output = Path("samples/latest_trajectory.mp4")
    fps = 20
    scale = 8

    run = Path(f"checkpoints/run_{args.run_id}")
    checkpoint_path = max(
        run.glob("checkpoint_*_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1])
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    frames = (checkpoint["trajectories"][-1]["observation"] * 8).permute(0, 2, 3, 1).numpy()
    size = (frames.shape[2] * scale, frames.shape[1] * scale)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST))
    writer.release()
    print(f"Saved last trajectory from {checkpoint_path} to {output}")

if __name__ == "__main__":
    main()

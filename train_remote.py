import modal
import subprocess
import os
volume = modal.NetworkFileSystem.new()

stub = modal.Stub(
    "nanoGPT",
    image=modal.Image.micromamba()
    .apt_install(["git", "libgl1-mesa-dev", "xvfb", "libxrender1"])
    .pip_install(["pyvista", "torch", "scikit-image"])
    .run_commands(
        [
            "git clone https://github.com/saliksyed/nanoGPT"
        ]
    )
)

@stub.function(
    gpu=modal.gpu.A10G(count=1),
    timeout=1800,
)
async def run_training():
    subprocess.run(["chmod", "+rwx", "prep.sh"], check=True, cwd="/nanoGPT")
    subprocess.run(["./prep.sh"], check=True, cwd="/nanoGPT")
    run_cmd = "python train.py --device=cuda:0"
    subprocess.run(run_cmd.split(), check=True, cwd="/nanoGPT")


@stub.local_entrypoint()
def main():
    run_training.remote()
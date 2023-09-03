import modal
import subprocess
volume = modal.NetworkFileSystem.new()

pyvista_requirements = "mesa-libGL mesa-dri-drivers xorg-x11-drv-dummy xserver-xorg-video-dummy".split()
stub = modal.Stub(
    "nanoGPT",
    image=modal.Image.micromamba()
    .apt_install(["git"] + pyvista_requirements)
    .pip_install(["pyvista", "torch", "scikit-image"])
    .run_commands(
        [
            "git clone https://github.com/saliksyed/nanoGPT"
        ],
        force_rebuild=True
    )
)

@stub.function(
    gpu="any",
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
import modal
import subprocess

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
    subprocess.run(["git", "pull"], check=True, cwd="/nanoGPT")
    subprocess.run(["git", "checkout", "simple"], check=True, cwd="/nanoGPT")
    subprocess.run(["chmod", "+rwx", "prep.sh"], check=True, cwd="/nanoGPT")
    run_cmd = "./prep.sh && python train.py --device=cuda:0"
    subprocess.run(args=run_cmd, check=True, shell=True, cwd="/nanoGPT")


@stub.local_entrypoint()
def main():
    run_training.remote()
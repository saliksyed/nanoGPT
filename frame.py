import subprocess

BG_COLOR = "black"


def render_polygon():
    blender = "Blender --background --python render_scene.py"
    subprocess.run(blender.split(), check=True, cwd="blender")


if __name__ == "__main__":
    render_polygon()

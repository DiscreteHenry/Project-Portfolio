# crystal.spec  ->  build with:  pyinstaller crystal.spec
import os
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

for pkg in ("cellpose", "torch"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Bundle downloaded Cellpose model weights for offline use.
# IMPORTANT: run the app once before building so weights are downloaded to ~/.cellpose/models
model_dir = os.path.join(os.path.expanduser("~"), ".cellpose", "models")
if os.path.isdir(model_dir):
    datas += [(model_dir, "cellpose_models")]

a = Analysis(
    ["gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["scipy._cyutility", "PIL._tkinter_finder"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "PyQt5"],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CrystalCounter",
    console=False,   # no terminal window for GUI
    onefile=True,
)
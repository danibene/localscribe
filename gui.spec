# -*- mode: python ; coding: utf-8 -*-
from localscribe.packaging import (
    get_pyinstaller_binaries,
    get_pyinstaller_datas,
    get_pyinstaller_hiddenimports,
)


datas = get_pyinstaller_datas()
binaries = get_pyinstaller_binaries()
hiddenimports = get_pyinstaller_hiddenimports()


a = Analysis(
    ['src/gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='localscribe-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

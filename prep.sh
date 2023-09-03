#!/bin/bash
sudo X :99 -config dummy.conf
export DISPLAY=:99
export MESA_GL_VERSION_OVERRIDE=3.2
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
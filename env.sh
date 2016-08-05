#!/bin/sh
# Set up the environment needed for our build.  Tries to keep
# everything except pip3, python3, and virtualenv inside virutalenv.

# For EC2, follow: http://markus.com/install-theano-on-aws/
# Remember to pick .deb(network), not .deb(local)
# Before you run env.sh, also install:
# liblapack-dev
# libfreetype6-dev
# libpng12-dev
# libjpeg-dev

set -e
command -v python3 >/dev/null 2>&1 || { \
  echo >&2 "python3 is required"; sudo apt-get install python3; }
command -v pip3 >/dev/null 2>&1 || { \
  echo >&2 "pip3 is required"; sudo apt-get install python3-pip; }
python3 -c 'import ensurepip' >/dev/null 2>&1 || { \
  echo >&2 "python3-venv is required"; sudo apt-get install python3.4-venv; }
pkg-config --modversion libpng > /dev/null 2>&1 || { \
  echo >&2 "libpng is required"; sudo apt-get install libpng12-dev; }
test -e /usr/include/freetype2/ft2build.h || { \
  echo >&2 "libfreetype is required"; sudo apt-get install libfreetype6-dev; }
test -e /usr/include/jpeglib.h || { \
  echo >&2 "libjpeg is required"; sudo apt-get install libjpeg-dev; }
test -e /usr/lib/liblapack.so || { \
  echo >&2 "liblapack is required"; sudo apt-get install liblapack-dev; }
test -e /usr/include/hdf5.h || { \
  echo >&2 "libhdf5 is required"; sudo apt-get install libhdf5-dev; }

rm -rf env
python3 -m venv env
. env/bin/activate

# upgrade pip inside venv since Ubuntu 14.04 uses a really old one
python3 -m pip install --upgrade pip

# install wheel in venv so we get wheel caching
python3 -m pip install wheel

# numpy isn't listed as a dependency in scipy, so we need to do it by hand
python3 -m pip install numpy
python3 -m pip install scipy

# Try using the blocks install.
pip install git+git://github.com/mila-udem/blocks.git \
  -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt

pip install git+git://github.com/mila-udem/blocks-extras.git

# pip install -e works for everything else.
python3 -m pip install --upgrade -e .
# exit the venv
deactivate

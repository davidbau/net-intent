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

# install theano 0.8.x, not master
pip install git+https://github.com/Theano/Theano.git@0.8.X#egg=theano

# Try using the blocks install.
pip install git+git://github.com/mila-udem/blocks.git \
  -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt

pip install git+git://github.com/mila-udem/blocks-extras.git

# pip install -e works for everything else.
python3 -m pip install --upgrade -e .
# exit the venv
deactivate

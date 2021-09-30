#!/usr/bin/env sh

set -e -x
cd "$(dirname ""${0}"")"

sudo apt install -y python3 python3-pip
python3 -m pip install poetry
cd analysis
poetry install

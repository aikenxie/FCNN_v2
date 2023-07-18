#!/bin/bash
# use the bash shell

cd /hildafs/projects/phy230010p/xiea/npzs/dy/processed

find . -maxdepth 1 -name "*.pt" -exec sh -c 'cp "$@" "$0"' /hildafs/projects/phy230010p/xiea/npzs/FCNN_data/processed {} +
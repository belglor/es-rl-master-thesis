#!/bin/bash

# Exit if error
set -e

# Get inputs
INPUTS=$@

# Set open file limit
ulimit -Sn 32768

# Set $HOME if running as a bsub script
if [ -z "$BSUB_O_WORKDIR" ]; then
    export HOME=$BSUB_O_WORKDIR
fi
# Set $HOME if running as a qsub script
if [ -z "$PBS_O_WORKDIR" ]; then
    export HOME=$PBS_O_WORKDIR
fi
cd $HOME

# Execute script
python3 ../experiments/main.py $INPUTS

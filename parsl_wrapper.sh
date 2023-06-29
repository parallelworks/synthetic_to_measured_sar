#!/bin/bash

# Confirm parsl_utils is cloned
rm -rf parsl_utils
git clone -b new-workflow https://github.com/parallelworks/parsl_utils.git parsl_utils

# Run the template
./parsl_utils/main.sh

#!/bin/bash
#!/bin/bash
set -x
# Can't get workflow type in PW to clone --recurse-submodules
rm -rf parsl_utils
git clone https://github.com/parallelworks/parsl_utils.git parsl_utils
cp parsl_utils/kill.sh .
cp parsl_utils/main.sh .
bash main.sh $@

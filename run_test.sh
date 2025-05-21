#!/bin/bash

#TT_OPTS=("USE_BF16_IN_L1" "USE_BF16" "USE_FP32")
TT_OPTS=("USE_BF16_IN_L1" "USE_BF16" "USE_FP32")
TEST_VALUES=(100 128 256 500 1000)

for opt in "${TT_OPTS[@]}"; do
    for val in "${TEST_VALUES[@]}"; do
        echo "Running with TT_BOLTZ_OPT=$opt and test_pairformer[$val]..."
        TT_BOLTZ_OPT="$opt" pytest -s test_tenstorrent.py::test_pairformer["$val"] \
            | tee "log_${opt}_pairformer${val}.log"
    done
done

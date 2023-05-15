This is a custom version of roc_auc in C++ (ab)using radix sort and the memory layout of floats in the range of 1.0-2.0.

## Compile

To compile, run `./compile.sh`.

## Changes

To accelerate the code beyond the original implementation, `sample_weights` support was removed.
Additionally, the return type was changed from `float` to `double` for enhanced precision.

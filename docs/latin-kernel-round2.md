# Exact-gradient pooling optimization

The second optimization round preserves the existing BF16-math generator computation. Its new Metal kernel replaces only frozen FP32 tokenizer area pooling. Calls with autograd enabled, unsupported dimensions, or unverified layouts retain the original adapter.

The kernel retains the original FP32 summation sequence. Short bins use one output per thread. Larger bins use the original SIMD reduction sequence. Profiling shows that pooling removes 928 sliced-mean calls per global update. The generator still performs its original forward, backward, clipping, and AdamW operations.

All compiler and tracing candidates that changed gradients were rejected. Whole-generator compilation produced a large embedding-gradient error. Narrower compilation still changed gradient rounding. TorchScript tracing changed accumulation through a shared weight cast. Allocator scheduling changes and scalar caches did not provide a useful additional gain and are not enabled.

The verification record is generated after complete training, exact state comparison, saved resume, and MLX evaluation checks. It records whether the requested additional ten percent reduction was reached. It compares this round against the previous BF16-math implementation, not the older FP32 or BF16 reference attention paths.

Model the Domain restricted the kernel to verified tensor layouts. Prove It Works required byte equality for raw and clipped gradients, updated weights, and saved optimizer and RNG state. Explain the Number required complete command timings with the same update count, memory checks, and cooling pauses. Attack the Premise led to a per-parameter gradient comparison before rejecting whole-model compilation.

The reduction scheduling follows the [pinned PyTorch MPS implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/kernels/ReduceOps.metal). Its [license notice](pytorch-kernel-notice.txt) accompanies this implementation.

Five alternating complete invocations per side measured 20 updates each. Median training time fell from 18.530 seconds to 17.988 seconds, a 2.9 percent reduction. Ranges were 18.249 to 18.644 seconds before and 17.860 to 18.053 seconds after. The requested additional ten percent reduction was not reached. Full-process timing fell by 2.8 percent. The generator matrix operations still dominate.

All five pairs saved byte-identical weights, AdamW tensors, sampler state, and RNG. The raw and clipped gradient check also matched byte for byte with dropout enabled. Frozen codes and contexts matched byte for byte for 160 puzzles. Saved resume produced the same full training state as an uninterrupted two-update run. The export completed a 100-puzzle MLX smoke evaluation. All 89 focused tests passed.

The final comparisons recorded no new swapouts or pageouts. The MPS allocator cap stayed at 8 GiB. Candidate endpoint allocations peaked at 4.14 GiB. The target work duty stayed at 80 percent. Independent whole-device GPU means ranged from 68.1 to 78.1 percent, with brief peaks up to 100 percent.

Start a new continuation directory to bind the updated runtime hashes. This command adds up to 600 updates to the existing 1,000-update source, then evaluates 2,000 validation puzzles.

```sh
caffeinate -i .venv/bin/python continue_latin.py --run runs/latin9-exact-pooling --evaluate
```

The training estimate is about eight minutes. Evaluation adds roughly one minute outside the training deadline. These are extrapolations from bounded trials. No full 600-update run was launched for this change.

To resume that exact run after a saved stop, add `--resume`. To select the previous frozen pooling path in a different new run, add `--frozen-pooling reference`. Pooling choice and runtime hashes are part of the immutable recipe. Changing them requires a new continuation.

[Verification record](latin-kernel-round2-verification.json) contains the timing, equality, resource, resume, and MLX artifacts.

| Model | Accuracy | Balanced acc. | Macro-F1 | 95% CI (recordings) | F1 feeding | F1 grooming | F1 pumping |
|---|---|---|---|---|---|---|---|
| Majority class | 0.470 | 0.333 | 0.213 | 0.17 to 0.25 | 0.640 | 0.000 | 0.000 |
| Fly position only | 0.665 | 0.665 | 0.653 | 0.58 to 0.72 | 0.715 | 0.480 | 0.763 |
| Motion statistics | 0.852 | 0.853 | 0.850 | 0.78 to 0.89 | 0.866 | 0.816 | 0.868 |
| Position + motion statistics | 0.843 | 0.843 | 0.840 | 0.79 to 0.88 | 0.859 | 0.792 | 0.867 |
| ResNet-18 on single frames | 0.871 | 0.863 | 0.859 | 0.79 to 0.89 | 0.909 | 0.866 | 0.802 |
| ResNet-18 on single frames + temporal shift | 0.893 | 0.886 | 0.883 | 0.84 to 0.91 | 0.931 | 0.874 | 0.844 |
| **ResNet-18 on frame triplets** | 0.938 | 0.937 | **0.935** | 0.89 to 0.96 | 0.951 | 0.926 | 0.927 |
| ResNet-18 on frame triplets + temporal shift | 0.939 | 0.939 | 0.936 | 0.89 to 0.96 | 0.949 | 0.932 | 0.927 |
| ResNet-18 on frame triplets, repeated with another seed | 0.933 | 0.934 | 0.930 | 0.88 to 0.96 | 0.945 | 0.921 | 0.925 |

| Final model compared with | Macro-F1 difference | 95% CI (recordings) |
|---|---|---|
| Fly position only | +0.282 | +0.201 to +0.366 |
| Motion statistics | +0.085 | +0.048 to +0.141 |
| Position + motion statistics | +0.095 | +0.049 to +0.133 |
| ResNet-18 on single frames | +0.075 | +0.051 to +0.110 |
| ResNet-18 on single frames + temporal shift | +0.052 | +0.031 to +0.074 |
| ResNet-18 on frame triplets + temporal shift | -0.001 | -0.011 to +0.006 |
| ResNet-18 on frame triplets, repeated with another seed | +0.004 | -0.006 to +0.015 |

| Clip length | Clips | Accuracy |
|---|---|---|
| < 2 s | 201 | 0.896 |
| 2 to 5 s | 441 | 0.937 |
| 5 to 15 s | 553 | 0.942 |
| > 15 s | 603 | 0.949 |

| Motion in the clip | feeding: clips (accuracy) | grooming: clips (accuracy) | pumping: clips (accuracy) |
|---|---|---|---|
| stillest fifth | 36 (61%) | 8 (0%) | 316 (97%) |
| 2nd | 153 (86%) | 116 (94%) | 90 (90%) |
| 3rd | 150 (99%) | 207 (97%) | 3 (67%) |
| 4th | 230 (99%) | 126 (93%) | 3 (0%) |
| most motion | 277 (97%) | 83 (90%) | 0 |

Low motion alone separates pumping from the rest with AUC 0.976.

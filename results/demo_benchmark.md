# Benchmark summary

Input table: `outputs\motion_detector\results\benchmark.csv`

Results use a synthetic 240-frame, 320x240 clip. They validate the pipeline and are not a claim about surveillance accuracy.

| Method | Pixel F1 | Pixel IoU | Frame F1 | Processing FPS | Events |
|---|---:|---:|---:|---:|---:|
| mog2 | 0.896 | 0.812 | 0.973 | 277.8 | 1 |
| knn | 0.895 | 0.811 | 0.970 | 248.0 | 1 |
| diff | 0.235 | 0.133 | 0.742 | 496.9 | 0 |
| running_average | 0.500 | 0.334 | 0.931 | 478.0 | 1 |

A crossing event is generated from a confirmed track crossing the configured line. The synthetic foreground is deliberately bright and simple. For CDnet or camera footage, replace the demo labels and tune the ROI, area, learning rate and shadow settings.

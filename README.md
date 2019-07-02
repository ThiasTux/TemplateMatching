# TemplateMatching

Research project on Template Matching (Training, Optimization and Evaluation). The TMM used is WarpingLCSS. The training is based on genetic algorithms.

## Training

WLCSSLearn is an algorithm for TMM training and parameters' optimization based on GA. It partially runs on CUDA, accelerating the training of WLCSS using GPGPU. 

## Template Matching

WLCSS is a TMM based on LCS that can be used on gestures warped in time. An implementation using CUDA framework is included in this project.

## Evaluation

The evaluation of the training methods and of TMM are performed using several datasets, in isolated and continuous recognition.

## References:

[1] Mathias Ciliberto, Luis Ponce Cuspinera, and Daniel Roggen. *"WLCSSLearn: learning algorithm for template matching-based gesture recognition systems."* International Conference on Activity and Behavior Computing. Institute of Electrical and Electronics Engineers, 2019.

[2] Mathias Ciliberto, and Daniel Roggen. *"WLCSSCuda: A CUDA Accelerated Template Matching Method for Gesture Recognition."* Proceedings of the 2019 ACM International Joint Conference and 2019 International Symposium on Pervasive and Ubiquitous Computing and Wearable Computers. ACM, 2019.





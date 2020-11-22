# Epitome: constructing the perfect handwritten digits


## Workflow
* `lenet.py`
    * Boolean to denote the "mode"
* `classification.py`
    * Print the loss plots after training
    * Allow exporting/importing the class for evaluation-only
* `optimization.py`
    * Freeze all current weights
    * Add gradient descent to the input `optimizer = optim.LBFGS([input_img.requires_grad_()])`
    * Change loss function to only look at nth softmax probability
    * Mask/bound inputs to [0, 1]
    * Show pictures of final results (rerunning multiple times)


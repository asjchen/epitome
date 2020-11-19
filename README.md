# Epitome: constructing the perfect handwritten digits


## Workflow
* `lenet.py`
    * Model at hand (initialized)
    * Boolean to denote the "mode"
* `classification.py`
    * Train the model
    * Print the loss plots after training
    * Allow exporting/importing the class for evaluation-only
    * Evaluate the model
* `optimization.py`
    * Freeze all current weights
    * Add gradient descent to the input `optimizer = optim.LBFGS([input_img.requires_grad_()])`
    * Change loss function to only look at nth softmax probability
    * Mask/bound inputs to [0, 255]
    * Show pictures of final results (rerunning multiple times)


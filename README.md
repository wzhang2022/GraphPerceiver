This is the research framework for our CS229br project.
As of right now, we are evaluating against the `ogbg-molhiv` dataset for graph property prediction.
The plan will be to add other datasets as well once we find a model that is able to perform well.
- The `ogb-molhiv` dataset is extremely imbalanced, so accuracy is meaningless.
  Instead, we benchmark with ROC-AUC.
- Additionally, it makes sense to train the model with a weighted loss function.
However, this seems to have mixed results on the ROC-AUC.
- a

Installed packages (Conda environment):
```
pytorch==1.8.0
einops==0.3.0
ogb==1.3.0
```
How to run example:

``
python train.py --model perceiver --save_file nosave --run_name init_test --batch_size 48 --depth 6
``

What each argument does:
- `depth` is just the number cross-attention followed by self-attention blocks in the Perceiver module.
- `model` specifies that we are using the perceiver model, but this is redundant until we test out multiple models.
- `run_name` is the name of the run as shown by `wandb`.
- `device` should always be `0` for `cuda`. We should not be running on CPU, so this argument is redundant.
- `save_file` specifies where to save the weights of the best model, as measured by validation loss.
- `n_epochs`, `batch_size`, `learning_rate`, and `clip` should all be self-explanatory.

TODO:
- Set up code to run an extensive hyperparameter search.
- Implement Laplacian Eigenvector positional encodings, as in [https://arxiv.org/pdf/2012.09699.pdf][here].
- Maybe there is a better way of encoding graphs than just a set of edges?

[here]: https://arxiv.org/pdf/2012.09699.pdf
This is the research framework for our CS229br project.
As of right now, we are evaluating against the `ogbg-molhiv` dataset for graph property prediction.
The plan will be to add other datasets as well once we find a model that is able to perform well.
- The `ogb-molhiv` dataset is extremely imbalanced, so accuracy is meaningless.
  Instead, we benchmark with ROC-AUC.
- Additionally, it makes sense to train the model with a weighted loss function.
However, this seems to have mixed results on the ROC-AUC.

Installed packages (Conda environment):
```
pytorch==1.8.0
einops==0.3.0
ogb==1.3.0
```
How to run example:

``
python train.py --model perceiver --save_file nosave --run_name William_BigModel --k_eigs --atom_emb_dim --bond_emb_dim  --batch_size 84 --depth 5 --num_latents 128 --latent_dim 128 --cross_heads 2 --latent_heads 4 --cross_dim_head 64 --latent_dim_head 64 --attn_dropout 0.2 --ff_dropout 0.2 --batch_size 16 --learning_rate 0.0001 --lr_decay 1 --scheduler exponential --n_epochs 40``

What each argument does:
- A bunch of the parameters are just model architecture details.
- `model` specifies that we are using the perceiver model, but this is redundant until we test out multiple models.
- `run_name` is the name of the run as shown by `wandb`.
- `k_eigs` specifies how many eigenvectors to use for Laplacian positional encodings. If 0, it forgoes using LPEs.
- `atom_emb_dim`, `bond_emb_dim` specifies the dimensions for the atom feature and bond feature embeddings.
- `device` should always be `0` for `cuda`. We should not be running on CPU, so this argument is redundant.
- `save_file` specifies where to save the weights of the best model, as measured by validation loss.
- `learning_rate` is self-explanatory. `lr_decay` is the gamma/factor. `scheduler` can be "exponential", "multistep", or "plateau".
- `milestone_frequency` and `milestone_start` determine scheduler milestones, i.e. \[start, start+freq, start+2freq...\]
- `n_epochs`, `batch_size`, `clip` should all be self-explanatory.

TODO (in order of importance):
- Implement the LAMB optimizer, which the Perceiver paper chooses over regular SGD. Paper: [https://arxiv.org/pdf/1904.00962.pdf][here]. Implementation:  [https://github.com/cybertronai/pytorch-lamb][here].
- Run until we observe memorization of training data and deep double descent.
- Maybe there is a better way of encoding graphs than just a set of edges?
- This [https://arxiv.org/pdf/2002.08709.pdf][a] might be an interesting regularizer to investigate?
- Set up code to run an extensive hyperparameter search.

DONE:

- Implement Laplacian Eigenvector positional encodings, as in [https://arxiv.org/pdf/2012.09699.pdf][here] 





Guidelines for good collaboration:
- Include your name in the `run_name` so that we know who is responsible for each run.
- Remove your failed runs from wandb so that it does not clutter, but do not delete other people's runs without consulting them first.
- Use meaningful run names. Not just "run1" or "run5". We can play around with wandb features so that we can organize runs better. If you know how to do this or learn how to, let us know.

[here]: https://arxiv.org/pdf/2012.09699.pdf
[a]: https://arxiv.org/pdf/2002.08709.pdf
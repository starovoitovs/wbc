# White blood cell classification

We used a modification of an [[adversarial algorithm with Margin Disparity Discrepancy (MDD) loss]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) for a domain
adaptation task presented in the 2022 [HIDA hackathon "Help A Hematologist Out"](https://www.helmholtz-hida.de/en/events/data-challenge-help-a-hematologist-out/).
 
The model can be found in `models/mdd.py`.

The code uses parts of the [tllib](https://github.com/thuml/Transfer-Learning-Library) repository of transfer learning algorithms.

We adapted the algorithm for the unsupervised DA setup and introduced SND and entropy losses as validation criteria.

# Usage

Run this command from the root of `tlda`:

    PYTHONPATH=/home/tmp/starokon/projects/tllib \
        CUDA_VISIBLE_DEVICES=0 \
        python models/mdd.py \
        Datasets \
        -d WBC \
        --source A M --target W \
        -a resnet18 \
        --epochs 300 \
        --iters-per-epoch 100 \
        --seed 1 \
        --log logs/WBC_AM2W \
        --margin 4 \
        --trade-off 1 \
        --phase train

Arguments `-s` and `-t` specify source and target dataset:

* A = Acevedo_20
* M = Matek_19
* W = WBC1
* T = WBC2

You can specify several letters (for example `A M` for Acevedo_20 and Matek_19).

Don't forget to put the root of `tlda` in the `PYTHONPATH` so the `tllib` can be imported.

# References

* Jiang, Junguang, Bo Fu, and Mingsheng Long. "Transfer-learning-library." (2020).
* Zhang, Yuchen, et al. "Bridging theory and algorithm for domain adaptation." International Conference on Machine Learning. PMLR, 2019.
* Musgrave, Kevin, Serge Belongie, and Ser-Nam Lim. "Benchmarking Validation Methods for Unsupervised Domain Adaptation."


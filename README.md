# CatEmbed

This repository contains the scripts used to obtain CatEmbed representations.

The `categorical_entity_embedder.py` is a modified version of the script available in the [Categorical Embedder](https://github.com/Shivanandroy/CategoricalEmbedder.git) package.

The datasets used in this work were obtained from [Mamun *et al.*](https://doi.org/10.1038/s41597-019-0080-z). In the `data` folder, we included the precise datasets used to obtain the results in our manuscript. The GP-CAF features in the adsorption datasets were generated using the [GP-representations](https://github.com/UON-comp-chem/GP-representations.git) package developed by [Li *et al.*](https://doi.org/10.1021/acs.jpclett.1c01319). Additionally, this folder contains the script used to create the train/validation split where subsets of bimetallic surfaces not included in the training set.

The `Models` folder contains example training scripts, train/validation datasets, and models for each of the representations presented in our manuscript.

## Dependencies

- **tensorflow** (v2.13.1)
- **tqdm** (4.58.0)
- **scikit-learn** (v1.4.2)
- **pandas** (v2.2.2 )
- **catboost** (v1.2.5)

## Usage

To train the embedding network and obtain an embedding dictionary for the categorical descriptors, run:
```bash
python perform_entity_embedding.py
```
This will call categorical_entity_embedder.py, which trains the embedding network.

To transform a dataset using the embedding dictionary obtained from perform_entity_embedding.py, run:

```bash
python load_transform_embeddings.py
```
## Acknowledgments

- The `categorical_entity_embedder.py` script was modified from the script available in the [Categorical Embedder](https://github.com/Shivanandroy/CategoricalEmbedder.git) package.
- The GP-CAF representation was developed by [Li *et al.*](https://doi.org/10.1021/acs.jpclett.1c01319), and their [repository](https://github.com/UON-comp-chem/GP-representations.git) was used to obtain the GP-CAF features.
- Datasets were obtained from [Mamun *et al.*](https://doi.org/10.1038/s41597-019-0080-z).

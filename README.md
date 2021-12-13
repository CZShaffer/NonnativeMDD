# NonnativeMDD
Repository for GaTech CS 7643 final project

## Data preparation
Download TIMIT from https://data.deepai.org/timit.zip and unzip it in `data/timit`.

## To reproduce experiments
You must have a CUDA capable device.

If you have not, create the environment using conda.

```shell
conda env create -f environment.yaml
```

Activate the environment.

```shell
conda activate cs7643-mdd
```

Run the main script.

```shell
python main.py
```

To summarize wav2vec 2.0 and HuBERT results across multiple runs,

```shell
cd experiments/results
python summarize.py
```
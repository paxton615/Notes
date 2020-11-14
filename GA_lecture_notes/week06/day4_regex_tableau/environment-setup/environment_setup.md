<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# Python Environment Setup

```bash
conda create -n textacy
source activate textacy
conda install pip
conda install ipykernel
python -m ipykernel install --user --name textacy --display-name "textacy"
```

Deactivate the environment with

```bash
conda deactivate
```

Remove the environment (not now!) with

```bash
conda remove -n textacy --all
```

To remove any traces from the known python kernels, ask for

```bash
jupyter kernelspec list
```

and remove the folder of the unwanted environment specifications.

# Pyspark Setup

```bash
conda create -n pyspark_env python=3.7.4
conda activate pyspark_env
conda install pip
conda install ipykernel
conda install -c conda-forge pyspark matplotlib seaborn scikit-learn
python -m ipykernel install --user --name pyspark_env --display-name "pyspark_env"
conda deactivate
```

# Java Installation

Additionally, you need to install Java JDK 8 from [here](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).

Choose the Java SE Development Kit 8u221 for your operating system.

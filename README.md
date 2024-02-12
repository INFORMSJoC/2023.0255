[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Graph-based Approach for Relating Integer Programs

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The purpose of this repository is to share the data and results reported on in the paper 
[A Graph-based Approach for Relating Integer Programs](https://doi.org/10.1287/ijoc.2023.0255) by Z. Steever, K. Hunt, M. Karwan, J. Yuan, and C. Murray. 

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0255

https://doi.org/10.1287/ijoc.2023.0255.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@article{ILPGraphs,
  author =        {Steever, Zachary and Hunt, Kyle and Karwan, Mark and Yuan, Junsong and Murray, Chase},
  publisher =     {INFORMS Journal on Computing},
  title =         {{A Graph-based Approach for Relating Integer Programs}},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0255.cd},
  note =          {Available for download at {https://github.com/INFORMSJoC/2023.0255}},
}  
```

## Data

The [data](data) directory contains the 950 instances that were used in this research. These 950 instances were originally collected in .lp format from [strIPlib](https://striplib.or.rwth-aachen.de/login/?next=/browser/) in early 2020. Using the [instanceformulation_to_instancegraph.py](scripts/instanceformulation_to_instancegraph.py) script, each .lp file (i.e., instance) was converted to its graph-based representation (following the methodology outlined in the paper) and stored as a .bin file. In general, the file [instanceformulation_to_instancegraph.py](scripts/instanceformulation_to_instancegraph.py) in the [scripts](scripts) directory will convert .lp files to their graph-based representation as defined in this research.

## Reproducing Results

The [results](results) directory contains the trained graph neural network ([trained_GCN.pt](results/trained_GCN.pt)) and the script to test the model ([test_GCN.py](results/test_GCN.py)). The documents [train_set.csv](results/train_set.csv) and [test_set.csv](results/test_set.csv) contain the paths to the instances (in the [data](data) directory) that were used for training (760 instances) and testing the GCN (190 instances). Depending on whether the train set or test set is being scored using the trained GCN, either [train_set.csv](results/train_set.csv) or [test_set.csv](results/test_set.csv) will be read into [test_GCN.py](results/test_GCN.py) allowing the script to automatically extract the needed instances from the [data](data) directory.

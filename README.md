[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Graph-based Approach for Relating Integer Programs

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The purpose of this repository is to share the data, model, and code used in the research reported on in the paper 
[A Graph-based Approach for Relating Integer Programs](https://doi.org/10.1287/ijoc.2023.0255) by Z. Steever, K. Hunt, M. Karwan, J. Yuan, and C. Murray. 

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0255

https://doi.org/10.1287/ijoc.2023.0255.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@article{ILPGraphs,
  author =        {Steever, Zachary and Hunt, Kyle and Karwan, Mark  and Yuan, Junsong and Murray, Chase},
  publisher =     {INFORMS Journal on Computing},
  title =         {{A Graph-based Approach for Relating Integer Programs}},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0255.cd},
  note =          {Available for download at {https://github.com/INFORMSJoC/2023.0255}},
}  
```

## Data

The [data](data) directory contains the 950 instances that were used in this research (stored as .bin files). These 950 instances were originally collected in .lp format from strIPlib in early 2020. Using the [instanceformulation_to_instancegraph.py][scripts/instanceformulation_to_instancegraph.py] script in the [scripts](scripts) directory, these .lp files were then converted to .bin files, where the .bin files store the graph-based representations of the formulations given in the .lp files. In general, the file “instanceformulation_to_instancegraph.py” in the “scripts” directory will convert .lp files to their graph-based representation using Deep Graph Library (following the methodology outlined in the paper).

## Reproducing Results

The “results” directory contains the trained graph neural network (“trained_GCN.py”) and the script to test the model (“test_gcn.py”). The documents “train_set.csv” and “test_set.csv” contain the paths to the instances in the “data” directory that were selected for training the GCN (760 instances) and testing the GCN (190 instances). Depending on whether the train set or test set is being scored using the trained model, either “train_set.csv”  or “test_set.csv” will be read into  “test_gcn.py” allowing the script to automatically extract the needed instances from the “data” directory. These are the same train and test sets used in the paper, allowing for reproducibility. 

# Multi-level Disentanglement Graph Neural Network (MD-GNN)


This is a PyTorch implementation of the MD-GNN, and the code includes the following modules:

* Datasets (Cora, Citeseer, Pubmed, Synthetic, and ZINC)

* Training paradigm for node classification, graph classification, and graph regression tasks

* Visualization

* Evaluation metrics 

  

## Main Requirements

* dgl==0.4.3.post2
* networkx==2.4
* numpy==1.18.1
* ogb==1.1.1
* scikit-learn==0.22.2.post1
* scipy==1.4.1
* torch==1.5.0



## Description

* train.py  
  * main() -- Train a new model for **node classification** task on the *Cora, Citeseer, and Pubmed* datasets
  * evaluate() -- Test the learned model for **node classification** task on the *Cora, Citeseer, and Pubmed* datasets
  * main_synthetic() -- Train a new model for **graph classification** task on the *Synthetic* dataset
  * evaluate_synthetic() -- Test the learned model for **graph classification** task on the *Synthetic* dataset
  * main_zinc() -- Train a new model for **graph regression** task on the *ZINC* datasets
  * evaluate_zinc() -- Test the learned model for **graph regression** task on the *ZINC* datasets
* dataset.py  
  
  * load_data() -- Load data of selected dataset
* MDGNN.py  
  
  * MDGNN() -- model and loss
* utils.py  
  * evaluate_att() -- Evaluate attribute-level disentanglement with *the visualization of relation-related attributes*
  * evaluate_corr() -- Evaluate node-level disentanglement with *the correlation analysis of latent features*
  * evaluate_graph() -- Evaluate graph-level disentanglement with *the visualization of disentangled relation graphs*



## Running the code

1. Install the required dependency packages and unzip files in the **data** folder.

2. We use [DGL](https://www.dgl.ai/) to implement all the GNN models on three citation datasets (Cora, Citeseer,  and Pubmed).  In order to evaluate the model with different splitting strategy (fewer and harder label rates), you need to replace the following file with the `citation_graph.py` provided.

> dgl/data/citation_graph.py

3. To get the results on a specific *dataset*, run with proper hyperparameters

  ```
python train.py --dataset data_name
  ```

where the *data_name* is one of the five datasets (cora, citeseer, pubmed, synthetic, and zinc). The model as well as the training log will be saved to the corresponding dir in **./log** for evaluation.

4. The evaluation the performance of three-level disentanglement performance, run

  ```
python utils.py
  ```



## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@article{wu2022multi,
  title={Multi-level disentanglement graph neural network},
  author={Wu, Lirong and Lin, Haitao and Xia, Jun and Tan, Cheng and Li, Stan Z},
  journal={Neural Computing and Applications},
  pages={1--15},
  year={2022},
  publisher={Springer}
}
```

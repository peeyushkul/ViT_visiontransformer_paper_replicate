# ViT_visiontransformer_paper_replicate
Creating Vision Transformer model replicating the paper - https://arxiv.org/abs/2010.11929

**Broad Steps:**
1. Break the main ViT paper into 4 key equations to be implemented
2. Get data and create the datasets and dataloaders
3. Split images data into patches and create the class, position and patch embeddings (Equation 1)
4. Implement MSA block (Multihead Self-Attention), with 12 heads and including LayerNorm (Equation 2)
5. Implement MLP block (Multilayer Perceptron), which includes 2 layers with GeLU non-linearity, and LayerNorm (Equation 3)
6. Get the classification output Y from the prepended learnable encoding Z0 (Equation 4)
7. Create the Transformer Encoder block, with 12 encoder layers, each layer having alternate MSA and MLP blocks
8. Plot the loss curves
9. Save the model if accuracy is satisfactory

**Details of files:**
* `ViT_vision_transformer_paper_replicate.ipynb` - the main file for paper replication. Opens in google colab or jupyter notebook
* `data_setup.py` - a file to prepare and download data if needed.
* `engine.py` - a file containing various training functions.
* `model_builder.py` - a file to create a PyTorch TinyVGG model.
* `train.py` - a file to leverage all other files and train a target PyTorch model.
* `utils.py` - a file dedicated to helpful utility functions.
* `predictions.py` - a file for making predictions with a trained PyTorch model and input image

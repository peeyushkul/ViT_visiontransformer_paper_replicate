# ViT_visiontransformer_paper_replicate
Creating Vision Transformer model replicating the paper - https://arxiv.org/abs/2010.11929

**Broad Steps**
1. Break the main ViT paper into 4 key equations to be implemented
2. Get data and create the datasets and dataloaders
3. Split images data into patches and creating the class, position and patch embeddings (Equation 1)
4. Implement MSA block (Multihead Self-Attention), with 12 heads and including LayerNorm as well (Equation 2)
5. Implement MLP block (Multilayer Perceptron), which includes 2 layers with GeLU non-linearity, and LayerNorm as well (Equation 3)
6. Get the classification output Y from the prepended learnable encoding Z0 (Equation 4)
7. Create the Transformer Encoder block, with 12 encoder layers
8. Plot the loss curves
9. Save the model if accuracy is satisfactory

**Details of files:**
* `data_setup.py` - a file to prepare and download data if needed.
* `engine.py` - a file containing various training functions.
* `model_builder.py` - a file to create a PyTorch TinyVGG model.
* `train.py` - a file to leverage all other files and train a target PyTorch model.
* `utils.py` - a file dedicated to helpful utility functions.
* `predictions.py` - a file for making predictions with a trained PyTorch model and input image

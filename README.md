# IEOR-223-Final-Project
Final Group Project for 223 about Generative Models

TimeGAN
	For the TimeGAN, please use the notebook included in the GitHub. A pre-trained model will be used to produce the samples from the pkl files. This was done using Google Colab. If you want to create samples from non_PCA data, change the ‘pca_synthesizer_stock.pkl’ to ‘synthesizer_stock.pkl’ and change ‘n_seq’ in the model parameters to 1. Please only run the t-SNE and PCA plots after creating the samples unless you use the non-PCA model.

Consistency Model
For the Consistency model, see images output in contents_cifar10 or run the Notebook consistency_debug.ipynb.

VAE
	For VAE models, please use the notebook included in the github. Only the “VAE Model” cells and the “Generate” cells need to be run if you don’t want to retrain the model. If you want to retrain the model, you can run the whole notebook.

PCA-GAN
	to generate you should run generation_gan3 file, it will create and sace generated pca points. Then you can run the recovery file which will create TS from the generated pca point and save the generated time series. You can assess the quality of the data and the valuation metrics running the file evaluation 2.


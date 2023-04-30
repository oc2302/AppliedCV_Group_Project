# AppliedCV_Group_Project
Spring 2023
Olivia Chu 
Approach 2 of Spatiotemporal Action Recognition: Reducing Computational Demand

# Adapted from codebases: 
TSM https://github.com/mit-han-lab/temporal-shift-module

TDM https://github.com/MCG-NJU/TDN

# Prerequisites
Python 3.6 or higher

PyTorch 1.4 or higher

Torchvision

TensorboardX

tqdm

scikit-learn

ffmpeg

decord


# Introduction
While 3D CNNs, such as the S3D, combine spatial and temporal analysis volumetrically, 2D CNNs also have the potential to learn spatiotemporal features.  The two most promising models for this task are the TSM and TDN.  TSM uses a zero-computation shift that merges previous and future frames.  This is done within the residual branch of the CNN architecture.  This approach then implements TDM after shifting to account for temporal difference.  The result is a 2D CNN architecture that learns spatial and temporal features more efficiently than a 3D architecture.

*Note: TDN/TDM and TSM/TSN abbreviations both appear throughout source code and papers.  The difference is whether refering to network or module.

# Datasets
Trained on Kinetics-400 & Something-something V1.  Evaluated on HMDB51.  Adapted dataloader from TSN dataloader. Aach frame was resized into 256 x 256 pixels and randomly cropped.  The output format was time x channel x height x width with 64 frames for each clip.


# Code
Built from TSN code source - training structure, ResNet backbone, dataloader, etc.  Adapted to run TSM and TDM sequentially as stacked 2D layers.  


# AppliedCV_Group_Project
Spring 2023
Olivia Chu 
Approach 2 of Spatiotemporal Action Recognition: Reducing Computational Demand

#Adapted from codebases: 
TSM https://github.com/mit-han-lab/temporal-shift-module
TDM https://github.com/MCG-NJU/TDN

#Introduction
While 3D CNNs, such as the S3D, combine spatial and temporal analysis volumetrically, 2D CNNs also have the potential to learn spatiotemporal features.  The two most promising models for this task are the TSM and TDN.  TSM uses a zero-computation shift that merges previous and future frames.  This is done within the residual branch of the CNN architecture.  This approach then implements TDM after shifting to account for temporal difference.  The result is a 2D CNN architecture that learns spatial and temporal features more efficiently than a 3D architecture.







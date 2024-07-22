# BUNDL
Uncertainty-Aware Bayesian Deep Learning with Noisy Training Labels for Epileptic Seizure Detection <br>
Accepted at Uncertainty for Safe Utilization of Machine Learning in Medical Imaging Workshop at MICCAI 2024 <br>
 <br>
<h2> Abstract </h2>

Supervised learning has become the dominant paradigm in computer-aided diagnosis. Generally, these methods assume that the training labels represent “ground truth” information about the target phenomena. In actuality, the labels, often derived from human annotations, are noisy/unreliable. This aleatoric uncertainty poses significant challenges for modalities such as electroencephalography (EEG), in which “ground truth” is difficult to ascertain without invasive experiments. In this paper, we propose a novel Bayesian framework to mitigate the effects of aleatoric label uncertainty in the context of supervised deep learning. Our target application is EEG-based epileptic seizure detection. Our framework, called BUNDL, leverages domain knowledge to design a posterior distribution for the (unknown) “clean labels” that automatically adjusts based on the data uncertainty. Crucially, BUNDL can be wrapped around any existing detection model and trained using a novel KL divergence-based loss function. We validate BUNDL on both a simulated EEG dataset and the Temple University Hospital (TUH) corpus using three state-of-the-art deep networks. In all cases, integrating BUNDL improves the seizure detection performance. We also demonstrate that accounting for label noise using BUNDL improves seizure onset localization from EEG by reducing false predictions from artifacts. 
<br>
![alt text](sozviz_single.png)


<h2> Data access </h2>
Download TUH data from: [TUH EEG Seizure Corpus
](https://isip.piconepress.com/projects/tuh_eeg/) <br>
Simulated data can be recreated using the provided matlab scripts. For the exact copy of data used in paper, please email dshama1@jhu.edu to setup large file transfer. <br>



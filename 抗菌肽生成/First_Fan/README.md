## Requirements
- python 3.9.7 or higher (3.9.7 is recommended as priority)
- keras==2.10.0
- pandas==1.5.2
- matplotlib==3.0.3 (not compulsory)
- propy3 (tutorial: https://propy3.readthedocs.io/en/latest/UserGuide.html)
- numpy==1.23.5
- sklearn=1.2.0
- propy3=1.1.0
- gensim=4.2.0 or 4.3.0
- scipy=1.9.3
- tensorflow=2.10.0 (it may raise errors if tensorflow is installed by default "pip install tensorflow"; Please indicate tensorflow=2.10.0 specifically)
## Implementation details:

1. The categorical data is stored in the folder Antifungal.
2. The regression data is stored in the folder MIC.

This algorithm demands one-hot code matrix (sequential information，50×20) and physical/chemical descriptors matrix (91×17) as input.
The one-hot code can be calculated by the two .csv documents aforementioned.

For example:\
  ```train_file_name = 'Training set.csv'  # Training dataset```\
  ```win1 = 100```\
  ```X1, T, rawseq, length = getMatrixLabelh(train_file_name, win1)```

Due to the size limitations of physiochemical descriptors of all sequences, the .npy documents containing these datasets were not submitted to Git Hub. For convenience, you can calculate it by the codes provided in Physicochemical properties.py. Or you can contact the author for these documents.Additionally, the trained model parameters (mode.h5) cannot be uploaded, but you can contact the author to obtain them. 

## The Vendor Problem: Training at the Far Edge
Kernel optimized Unet Model for sesmic inversion problems.

## Dataset

For the training process, we generate the simulated velocity models and their corresponding measurement by solving the acoustic wave equation.

## Training & Testing
The script FCNVMB_train_orig.py runs  baseline model without any kernel optimization.
The scripts FCNVMB_train.py and FCNVMB_test.py are implemented for training and testing. If you want to train your own network, please firstly checkout the files named ParamConfig.py and PathConfig.py, to be sure that all the parameters and the paths are consistent, e.g.
```
####################################################
####             MAIN PARAMETERS                ####
####################################################
SimulateData  = True          # If False denotes training the CNN with SEGSaltData
ReUse         = False         # If False always re-train a network 
DataDim       = [2000,301]    # Dimension of original one-shot seismic data
data_dsp_blk  = (5,1)         # Downsampling ratio of input
ModelDim      = [201,301]     # Dimension of one velocity model
label_dsp_blk = (1,1)         # Downsampling ratio of output
dh            = 10            # Space interval 


####################################################
####             NETWORK PARAMETERS             ####
####################################################   
BatchSize         = 10        # Number of batch size
LearnRate         = 1e-3      # Learning rate
Nclasses          = 1         # Number of output channels
Inchannels        = 29        # Number of input channels, i.e. the number of shots
SaveEpoch         = 20        
DisplayStep       = 2         # Number of steps till outputting stats
```
and
```
###################################################
####                   PATHS                  #####
###################################################
 
main_dir   = '/home/yfs/Code/pytorch/FCNVMB/'     # Replace your main path here

## Check the main directory
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!')
    
    
## Data path
if os.path.exists('./data/'):
    data_dir    = main_dir + 'data/'               # Replace your data path here
else:
    os.makedirs('./data/')
    data_dir    = main_dir + 'data/'
    
# Define training/testing data directory

train_data_dir  = data_dir  + 'train_data/'        # Replace your training data path here
test_data_dir   = data_dir  + 'test_data/'         # Replace your testing data path here

```

Then checkout these two main files to train/test the network, simply type
```
python FCNVMB_train.py
python FCNVMB_test.py
```
Please run python FCNVMB_train_orig.py baseline model training time.

## Enviroment Requirement

```
python = 3.8.5
pytorch = 1.4.0
numpy
scipy
matplotlib
scikit-image
math
```
All of them can be installed via ```conda (anaconda)```, e.g.
```
conda install pytorch torchvision torchaudio cudatoolkit=10.0 -c pytorch
```


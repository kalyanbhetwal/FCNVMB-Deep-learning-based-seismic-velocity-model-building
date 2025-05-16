# -*- coding: utf-8 -*-
"""
Fully Convolutional neural network (U-Net) for velocity model building from prestack
unmigrated seismic data directly
"""

################################################
########        IMPORT LIBARIES         ########
################################################

from ParamConfig import *
from PathConfig import *
from LibConfig_orig import *

from torch.profiler import profile, record_function, ProfilerActivity
enable_profiling = True

################################################
########             NETWORK            ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

net = UnetModel(n_classes=Nclasses, in_channels=Inchannels, is_deconv=True, is_batchnorm=True) 
if torch.cuda.is_available():
    net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

if ReUse:
    print('***************** Loading the pre-trained model *****************')
    print('')
    premodel_file = models_dir + premodelname + '.pkl'
    net = net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish downloading:', str(premodel_file))

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading Training DataSet *****************')
train_set, label_set, data_dsp_dim, label_dsp_dim = DataLoad_Train(
    train_size=TrainSize,
    train_data_dir=train_data_dir,
    data_dim=DataDim,
    in_channels=Inchannels,
    model_dim=ModelDim,
    data_dsp_blk=data_dsp_blk,
    label_dsp_blk=label_dsp_blk,
    start=1,
    datafilename=datafilename,
    dataname=dataname,
    truthfilename=truthfilename,
    truthname=truthname
)

train = data_utils.TensorDataset(torch.from_numpy(train_set), torch.from_numpy(label_set))
train_loader = data_utils.DataLoader(train, batch_size=BatchSize, shuffle=True)

################################################
########            TRAINING            ########
################################################

print('\n' + '*' * 43)
print('           START TRAINING')
print('*' * 43 + '\n')

print(f'Original data dimension: {DataDim}')
print(f'Downsampled data dimension: {data_dsp_dim}')
print(f'Original label dimension: {ModelDim}')
print(f'Downsampled label dimension: {label_dsp_dim}')
print(f'Training size: {TrainSize}')
print(f'Training batch size: {BatchSize}')
print(f'Number of epochs: {Epochs}')
print(f'Learning rate: {LearnRate:.5f}')

loss1 = 0.0
step = int(TrainSize / BatchSize)
start = time.time()

for epoch in range(Epochs): 
    epoch_loss = 0.0
    since = time.time()
    for i, (images, labels) in enumerate(train_loader):        
        iteration = epoch * step + i + 1
        net.train()

        images = images.view(BatchSize, Inchannels, data_dsp_dim[0], data_dsp_dim[1]).to(device)
        labels = labels.view(BatchSize, Nclasses, label_dsp_dim[0], label_dsp_dim[1]).to(device)
        
        optimizer.zero_grad()     

        # ========= PROFILING SECTION =========
        if enable_profiling and epoch == 0 and i == 0:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    outputs = net(images, label_dsp_dim)
                    loss = F.mse_loss(outputs, labels, reduction='sum') / (label_dsp_dim[0]*label_dsp_dim[1]*BatchSize)
            enable_profiling = False

            print("\n[Profiler] --- Top CUDA Ops ---")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        else:
            outputs = net(images, label_dsp_dim)
            loss = F.mse_loss(outputs, labels, reduction='sum') / (label_dsp_dim[0]*label_dsp_dim[1]*BatchSize)
        # =======================================

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if iteration % DisplayStep == 0:
            print('Epoch: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'.format(
                epoch + 1, Epochs, iteration, step * Epochs, loss.item()))

    if (epoch + 1) % 1 == 0:
        print(f'Epoch: {epoch+1} finished! Loss: {epoch_loss/i:.5f}')
        loss1 = np.append(loss1, epoch_loss/i)
        time_elapsed = time.time() - since
        print(f'Epoch consuming time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), models_dir + modelname + '_epoch' + str(epoch + 1) + '.pkl')
        print(f'Trained model saved: {int((epoch+1)*100/Epochs)} percent completed')

time_elapsed = time.time() - start
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 17}
font3 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 21}
SaveTrainResults(loss=loss1, SavePath=results_dir, font2=font2, font3=font3)

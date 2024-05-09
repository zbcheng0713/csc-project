# %load predict_location3c4d.py

### train and test the localization model ####
import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from socket import gethostname
import geopy.distance
from geopy import distance

#========Preparing datasets for PyTorch DataLoader=====================================
# Custom data pre-processor to transform X and y from numpy arrays to torch tensors
class PrepareData(Dataset):
    def __init__(self, path):
        self.X, self.y = torch.load(path)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#========Network architecture=====================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d takes C_in, C_out, kernel size, stride, padding
        # input array (3,55,2500)
        self.conv1 = nn.Conv2d(3, 4, (1,9), stride = 1, padding=(0,4))
        self.pool1 = nn.MaxPool2d((1,5), stride=(1,5))
        self.conv2 = nn.Conv2d(4, 4, (5,3), stride = 1, padding=(2,1))
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(4, 8, (5,3), stride = 1, padding=(2,1))

        self.fc1 = nn.Linear(8*55*125, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(torch.squeeze(x,1))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 8*55*125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

 
#========Training the model =====================================
# ds_train is the training dataset loader
# ds_test is the testing dataset loader

# def train_model(ds_train, ds_test):
#     net = Net()
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(net.parameters(), lr = 5e-5)
#     num_epoch = 80
#     df = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Training Loss'])
#     losses = []
#     accs = []
#     for epoch in range(num_epoch):  # loop over the dataset multiple times
#         running_loss = 0.0
#         epoch_loss = 0.0
#         for i, (_x, _y) in enumerate(ds_train):

#             optimizer.zero_grad() # zero the gradients on each pass before the update

#             #========forward pass=====================================
#             outputs = net(_x.unsqueeze(1))
#             loss = criterion(outputs, _y)
#             # acc = tr.eq(outputs.round(), _y).float().mean() # accuracy
#             # print(loss.item())

#             #=======backward pass=====================================
#             loss.backward() # backpropagate the loss through the model
#             optimizer.step() # update the gradients w.r.t the loss

#             running_loss += loss.item()
#             epoch_loss += loss.item()
#             if i % 10 == 9:    # print running_loss for every 10 mini-batches
#                 print('[%d, %5d] loss: %.4f' %
#                 (epoch + 1, i + 1, running_loss / 10))
#                 running_loss = 0.0
        
#         test_loss = 0.0
#         with torch.no_grad():
#             for i, (_x, _y) in enumerate(ds_test):
#                 outputs = net(_x.unsqueeze(1))
#                 loss = criterion(outputs,_y)
#                 test_loss += loss.item()
#         print('[epoch %d] test loss: %.4f training loss: %.4f' %
#                 (epoch + 1, test_loss / len(ds_test), epoch_loss / len(ds_train))) 
#         temp_df = pd.DataFrame({'Epoch': ['%d'%(epoch+1)],'Test Loss': ['%.4f'%(test_loss / len(ds_test))], 'Training Loss': ['%.4f'%(epoch_loss / len(ds_train))]})
#         df = pd.concat([df, temp_df], ignore_index=True)
#     df.to_csv('../models/loc_model_loss.csv',index=False)	
#     print('Finished Training')
#     return net


#========Training the model with time analysis=====================================
def train_model(ds_train, ds_test,local_rank, rank):
    # net = Net() ##default
    net = Net().to(local_rank) ##to gpu
    net = DDP(net, device_ids=[local_rank])
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr = 5e-5)
    num_epoch = 80
    df = pd.DataFrame(columns=['Epoch', 'Test Loss', 'Training Loss'])
    losses = []
    accs = []
        
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        ds_train_loader.sampler.set_epoch(epoch)
        profile_out = epoch == num_epoch // 2
        if profile_out:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA], 
                                                          record_shapes=True, profile_memory=True, 
                                                          with_stack=True)
            profiler.start()
        
        running_loss = 0.0
        epoch_loss = 0.0
        test_loss = 0.0
        
        for i, (_x, _y) in enumerate(ds_train):
            
            _x, _y = _x.to(device), _y.to(device) ### move data to gpu

            optimizer.zero_grad() # zero the gradients on each pass before the update

            #========forward pass=====================================
            outputs = net(_x.unsqueeze(1))
            loss = criterion(outputs, _y)
            # acc = tr.eq(outputs.round(), _y).float().mean() # accuracy
            # print(loss.item())

            #=======backward pass=====================================
            loss.backward() # backpropagate the loss through the model
            optimizer.step() # update the gradients w.r.t the loss

            running_loss += loss.item() ##default
            epoch_loss += loss.item()
            # running_loss += loss.detach()
            # epoch_loss += loss.detach()
            if i % 10 == 9:    # print running_loss for every 10 mini-batches
                running_loss = running_loss.item()
                print('[%d, %5d] loss: %.4f' %
                (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        if profile_out:
            profiler.stop() 
            print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        epoch_loss_tensor = torch.tensor(epoch_loss, device=local_rank)
        # epoch_loss_local= epoch_loss.sourceTensor.clone().detach()
            
        dist.reduce(epoch_loss_tensor, dst=0)
        if rank == 0:
            epoch_loss = epoch_loss_tensor.item() / world_size        
            with torch.no_grad():
                if profile_out:
                    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA], 
                                                          record_shapes=True, profile_memory=True, 
                                                          with_stack=True)
                    profiler.start()    
                net.eval() 
                for i, (_x, _y) in enumerate(ds_test):
                    
                    _x, _y = _x.to(device), _y.to(device) ### move data to gpu
                    
                    outputs = net(_x.unsqueeze(1))
                    loss = criterion(outputs,_y)
                    test_loss += loss.item()
                    # test_loss += loss.detach()
                net.train()
                test_loss=test_loss.item()
                if profile_out:
                    profiler.stop() 
                    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            print('[epoch %d] test loss: %.4f training loss: %.4f' %
                    (epoch + 1, test_loss / len(ds_test), epoch_loss / len(ds_train))) 
            temp_df = pd.DataFrame({'Epoch': ['%d'%(epoch+1)],'Test Loss': ['%.4f'%(test_loss / len(ds_test))], 'Training Loss': ['%.4f'%(epoch_loss / len(ds_train))]})
            df = pd.concat([df, temp_df], ignore_index=True)
            if epoch == num_epoch - 1:
                df.to_csv('../models/loc_model_loss.csv',index=False)	
    print('Finished Training')
    return net

#========Get the distance between two points=====================================
# true is a list of true locations
# predicted is a list of predicted locations
# Returns in a list of distances between each true/predicted point, in km 
def dist_list(true, predicted):
    dist_list = np.zeros((predicted.shape[0]))
    for i in range(predicted.shape[0]):
        origin = (true[i,0], true[i,1])
        dest = (predicted[i,0], predicted[i,1])
        dist_list[i] = distance.distance(origin, dest).km
    return dist_list

#========Testing the model =====================================
# ds is the testing dataset
# ds_loader is the testing dataset loader
# net is the trained network
def test_model(ds,ds_loader, net):
    net.to(device)##move to gpu
    criterion = nn.MSELoss()
    test_no = len(ds)
    # batch_size=32 ##default
    batch_size=32
    print(test_no)
    # y_hat = np.zeros((test_no,4)) ##default
    # y_ori = np.zeros((test_no,4))
    y_hat = torch.zeros((test_no,4))
    y_ori = torch.zeros((test_no,4))
    accurate = 0
    with torch.no_grad():
        for i, (_x, _y) in enumerate(ds_loader):
            _x, _y = _x.to(device), _y.to(device) ### move data to gpu
            outputs = net(_x.unsqueeze(1))
            loss = criterion(outputs,_y)
            y_hat[batch_size*i:batch_size*(i+1),:] = outputs
            y_ori[batch_size*i:batch_size*(i+1),:] = _y
    
    y_hat=y_hat.cpu().detach().numpy()
    y_ori=y_ori.cpu().detach().numpy()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    event_coordref = (19.5,-155.5,0,0)
    event_norm = (1.0,1.0,50.0,10.)

    # first rescale and then shift the earthquake source values, reverse the steps in generate*.py
    ds.y = np.multiply(ds.y,event_norm)
    y_hat = np.multiply(y_hat,event_norm)
    y_ori = np.multiply(y_ori,event_norm)
    ds.y = np.add(ds.y,event_coordref)
    y_hat = np.add(y_hat,event_coordref)
    y_ori = np.add(y_ori,event_coordref)

    for k in range(len(y_hat)):
        if y_hat[k,2] < 0:
            y_hat[k,2] = 0 # no earthquake in the air (note: HVO catalog ignore topo)

    dista = dist_list(y_ori, y_hat)
    for k in range(len(dista)):
        print(y_ori[k,0],y_ori[k,1],y_ori[k,2],y_hat[k,0],y_hat[k,1],y_hat[k,2],y_ori[k,3],y_hat[k,3])
    dep_diff = y_ori[:,2] - y_hat[:,2]
    print(np.mean(dista), np.std(dista), np.mean(abs(dep_diff)), np.std(dep_diff))
    
    df=pd.DataFrame({'mean dist': [np.mean(dista)],'std dist': [np.std(dista)], 'mean abs dep diff': [np.mean(abs(dep_diff))],'std dep diff': [np.std(dep_diff)]})
    df.to_csv('../models/loc_test.csv',index=False)
    # Visualize  the  results of the predicted location
    ax.scatter(ds.y[:,0], ds.y[:,1], -ds.y[:,2], marker='o',s=10,label="Catalog")
    ax.scatter(y_hat[:,0], y_hat[:,1], -y_hat[:,2], marker='^',s=10,label="predicted")
    ax.set_xlim(18, 20.5)
    ax.set_ylim(-154, -157)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Depth')
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig("{}/{}".format('../models','test_pred_loc.pdf'),dpi=500,bbox_inches='tight')
    plt.savefig("{}/{}".format('../models','test_pred_loc.png'),dpi=500,bbox_inches='tight')


if __name__ == "__main__":
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ['WORLD_SIZE'])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    num_workers= int(os.environ["SLURM_CPUS_PER_TASK"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
      f" {gpus_per_node} allocated GPUs per node.", flush=True)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Running on Device {device}")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {torch.cuda.get_device_name(0)}")
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    t1 = time.time()
    # Prepare the training dataset and loader 
    # path is where the preprocessed training event data is housed, can be replaced with your updated location 
    # ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/train_data3c4d_NotAbs2017Mcut50sLin.pt')
    ds_train = PrepareData(path = '../pts/train_data3c4d.pt')
    train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank)
    ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    # ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=True) ##default
    # ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=True,num_workers=8,pin_memory=True)
    t2=time.time()
    print(f"Train loader takes {(t2-t1)/3600} h")
    
    
    # Prepare the testing dataset and loader 
    # path is where the preprocessed testing event data is housed, can be replaced with your updated location 
    # ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/test_data3c4d_NotAbs2017Mcut50sLin.pt')
    ds_test = PrepareData(path = '../pts/test_data3c4d.pt')
    # ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=True) ##default
    ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=True,num_workers=num_workers,pin_memory=True)
    t3=time.time()
    print(f"Test loader takes {(t3-t2)/3600} h")
    
    net = train_model(ds_train_loader, ds_test_loader,local_rank, rank)
    t4=time.time()
    print(f"Model training takes {(t4-t3)/3600} h")
    
    if rank == 0:
        # Analyze our final model on the testing dataset 
        accuracy = test_model(ds_test, ds_test_loader, net)
        t5=time.time()
        print(f"Model testing takes {(t5-t4)/3600} h")

        # predict_path is where we will store our trained model
        # predict_path = './SeisConvNetLoc_NotAbs2017Mcut50sLin.pth'
        predict_path = '../models/SeisConvNetLoc.pth'
        torch.save(net.state_dict(), predict_path)
    
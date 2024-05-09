# %load detect_3c.py
###   train and test the detection model ####
import time
import pandas as pd
import pdb
import os
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


#========Preparing datasets for PyTorch DataLoader=====================================
# Custom data pre-processor to transform X and y from numpy arrays to torch tensors
class PrepareData(Dataset):
    def __init__(self, path):
        self.X, self.y = torch.load(path)
        # CE loss only accepts ints as classes
        self.y = self.y.type(torch.LongTensor) 

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
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(torch.squeeze(x,1))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 8*55*125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#========Training the model with time analysis=====================================
def train_model(ds_train, ds_test,local_rank, rank):
    # net = Net() ##default
    # net = Net().to(device) ##to gpu
    net = Net().to(local_rank) ##to gpu
    net = DDP(net, device_ids=[local_rank])
    # Cross Entropy Loss is used for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr = 2e-5)
    num_epoch = 80
    df=pd.DataFrame(columns=['Epoch', 'Test Loss', 'Training Loss'])
    losses = []
    accs = []
    
    for epoch in range(num_epoch):# loop over the dataset multiple times
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
        correct = 0
        total = 0

        for i, (_x, _y) in enumerate(ds_train):
            
            _x, _y = _x.to(device), _y.to(device) ### move data to gpu
    
            optimizer.zero_grad() # zero the gradients on each pass before the update
    
            #========forward pass=====================================
            outputs = net(_x.unsqueeze(1))
    
            loss = criterion(outputs, _y)
    
            #=======backward pass=====================================
            loss.backward() # backpropagate the loss through the model
            optimizer.step() # update the gradients w.r.t the loss
    
            running_loss += loss.item()
            epoch_loss += loss.item()
            # running_loss += loss.detach()
            # epoch_loss += loss.detach()
            if i % 10 == 9:    # print every 10 mini-batches
                running_loss = running_loss.item() 
                print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        if profile_out:
            profiler.stop() 
            print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        epoch_loss_tensor = torch.tensor(epoch_loss, device=local_rank)
        # epoch_loss_local= epoch_loss.sourceTensor.clone().detach()
            
        dist.reduce(epoch_loss_tensor, dst=0)
            
        # For each epoch, monitor test loss to ensure we are not overfitting
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
                    _x, _y = _x.to(local_rank), _y.to(local_rank) ### move data to gpu
                    outputs = net(_x.unsqueeze(1))
                    loss = criterion(outputs,_y)
                    test_loss += loss.item()
                    # test_loss += loss.detach()
        
                    _, predicted = torch.max(outputs.data,1)
                    total += _y.size(0)
                    correct += (predicted == _y).sum().item()
                    # correct += (predicted == _y).sum().detach()
                net.train()
                test_loss=test_loss.item()
                # test_loss=test_loss.detach()
                if profile_out:
                    profiler.stop() 
                    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                
            print('[epoch %d] test loss: %.3f training loss: %.3f' %
                    (epoch + 1, test_loss / len(ds_test), epoch_loss / len(ds_train))) 
            temp_df = pd.DataFrame({'Epoch': ['%d'%(epoch+1)],'Test Loss': ['%.4f'%(test_loss / len(ds_test))], 'Training Loss': ['%.4f'%(epoch_loss / len(ds_train))]})
            df = pd.concat([df, temp_df], ignore_index=True)
            if epoch == num_epoch - 1:
                df.to_csv('../models/det_model_loss.csv',index=False)	
    print('Finished Training')
    return net

#========Testing the model =====================================
# ds is the testing dataset
# ds_loader is the testing dataset loader
# net is the trained network
def test_model(ds,ds_loader, net):
    net.to(device) ##move to gpu
    net.eval()
    criterion = nn.CrossEntropyLoss()
    test_no = len(ds)
    # batch_size=32 ##default
    batch_size=32

    df = pd.DataFrame(columns=['Threshold', 'Accuracy', 'Precision','Recall','TPR','FPR','FScore'])
    # precision and recall values for classification threshold, thre
    thre = torch.arange(0,1.01,0.05)
    thre_no = len(thre)
    true_p = torch.zeros(thre_no)
    false_p =torch.zeros(thre_no)
    false_n = torch.zeros(thre_no)
    true_n = torch.zeros(thre_no)

    y_hat = torch.zeros(test_no,2)
    y_ori = torch.zeros(test_no)
    y_pre = torch.zeros(test_no)
    with torch.no_grad():
        for i, (_x, _y) in enumerate(ds_loader):
            
            _x, _y = _x.to(device), _y.to(device) ### move data to gpu
            
            outputs = net(_x.unsqueeze(1))
            
            # view output as probability and set classification threshold
            prob = F.softmax(outputs,1)

            for j in range(thre_no):
                pred_threshold = (prob>thre[j]).float()
                predicted = pred_threshold[:,1]
            
                for m in range(len(_y)):
                    if _y[m] == 1. and pred_threshold[m,1] == 1.:
                        true_p[j] += 1.
                    if _y[m] == 0. and pred_threshold[m,1] == 1.:
                        false_p[j] += 1.
                    if _y[m] == 1. and pred_threshold[m,1] == 0.:
                        false_n[j] += 1.
                    if _y[m] == 0. and pred_threshold[m,1] == 0.:
                        true_n[j]  += 1.
        
            y_hat[batch_size*i:batch_size*(i+1),:] = outputs
            y_ori[batch_size*i:batch_size*(i+1)] = _y
            y_pre[batch_size*i:batch_size*(i+1)] = predicted

    print("Threshold, Accuracy, Precision, Recall, TPR, FPR, FScore")
    for j in range(thre_no):
        acc = 100*(true_p[j]+true_n[j])/(true_p[j]+true_n[j]+false_p[j]+false_n[j])

        if (true_p[j]+false_p[j]) > 0.:
            pre = 100*true_p[j]/(true_p[j]+false_p[j])
        else:
            pre = 100*torch.ones(1)

        if (true_p[j]+false_n[j]) > 0.:
            rec = 100*true_p[j]/(true_p[j]+false_n[j])
        else:
            rec = 100*torch.ones(1)

        tpr = 100*true_p[j]/(true_p[j]+false_n[j])
        fpr = 100*false_p[j]/(false_p[j]+true_n[j])
        fscore = 2*pre*rec/(pre+rec)
        print(" %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" %(thre[j].item(),acc.item(),pre.item(),rec.item(),tpr.item(),fpr.item(),fscore))
        temp_df = pd.DataFrame({'Threshold': ['%.2f'%thre[j].item()],
                      'Accuracy': ['%.2f'%acc.item()], 
                      'Precision': ['%.2f'%pre.item()],
                      'Recall':['%.2f'%rec.item()],
                      'TPR':['%.2f'%tpr.item()],
                      'FPR':['%.2f'%fpr.item()],
                      'FScore':['%.2f'%fscore]})
        df = pd.concat([df, temp_df], ignore_index=True)
    df.to_csv('../models/det_model_test.csv',index=False)

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
    # print(f"Running on Device {torch.cuda.get_device_name(0)}")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {torch.cuda.get_device_name(0)}")
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    t1 = time.time()
    # Prepare the training dataset and loader 
    # path is where the preprocessed training event data is housed
    # ds_train = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_train_data_sortedAbs50s.pt')
    ds_train = PrepareData(path = '../pts/detect_train_data_sortedAbs50s.pt')
    train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank)
    ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    # ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=True) ## default
    # ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=True,num_workers=8,pin_memory=True)
    t2=time.time()
    print(f"Train loader takes {(t2-t1)/3600} h")
    
    # Prepare the testing dataset and loader 
    # path is where the preprocessed test event data is housed
    # ds_test = PrepareData(path = '/Volumes/jd/data.hawaii/pts/detect_test_data_sortedAbs50s.pt')
    ds_test = PrepareData(path = '../pts/detect_test_data_sortedAbs50s.pt')
    # test_sampler = DistributedSampler(ds_test, num_replicas=world_size, rank=rank, shuffle=False)
    # ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, sampler=test_sampler)

    # ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=True)## default
    ds_test_loader = DataLoader(ds_test, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    t3=time.time()
    print(f"Test loader takes {(t3-t2)/3600} h")
    
    net = train_model(ds_train_loader, ds_test_loader,local_rank, rank)
    # net = train_model(ds_train_loader, ds_test_loader)
    t4=time.time()
    print(f"Model training takes {(t4-t3)/3600} h")
    
    if rank == 0:
        # Analyze our final model on the testing dataset 
        accuracy = test_model(ds_test, ds_test_loader, net)
        t5=time.time()
        print(f"Model testing takes {(t5-t4)/3600} h")

        # detect_net_path is where we will store our trained model
        # detect_net_path = './SeisConvNetDetect_sortedAbs50s.pth'
        pts_dir = "../models"
        os.makedirs(pts_dir, exist_ok=True)
        detect_net_path = '../models/SeisConvNetDetect_sortedAbs50s.pth'
        torch.save(net.state_dict(), detect_net_path)
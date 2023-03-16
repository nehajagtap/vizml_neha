
import os
import numpy as np
import pickle
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import ground_truth
import torch.nn.functional as F
#from data_loader import TableDataset
from datasets import TableDataset, SemanticTableDataset



class NeuralNet(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        add_non_linear = True
        self.non_linearity = nn.Tanh
        self.in_features = in_features
        self.net = nn.Sequential(
            #  1474
            nn.Linear(in_features=self.in_features, out_features= 512),
            #nn.ReLU(),
            #nn.Linear(in_features=1024, out_features=512),
            nn.Identity() if not add_non_linear else self.non_linearity(),
            nn.Linear(in_features=512, out_features=256),
            nn.Identity() if not add_non_linear else self.non_linearity(),
            nn.Linear(in_features=256, out_features=128),
            nn.Identity() if not add_non_linear else self.non_linearity(),
            nn.Linear(in_features=128, out_features=64),
            nn.Identity() if not add_non_linear else self.non_linearity(),
            nn.Linear(in_features=64, out_features=1)
        )


    def forward(self, x):
        # x is a tuple of tensors: (x1_features, x2_features, agg_features, gr_t)
        out = self.net(x)
        return out


def cal_acc(preds, gr_t):
    preds_tag = F.softmax(preds).detach().argmax()
    #print(f"preds_tag: {preds_tag}")
    correct_res_sum = (preds_tag == gr_t).sum().float()
    acc = correct_res_sum / gr_t.shape[0]
    acc = torch.round(acc*100)
    return acc


def main():
    #0: set device and run-id
    random_seed=0
    np.random.seed(random_seed)
    run_id = uuid.uuid4()
    PATH = f"./logs/{run_id}/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print(f"Made path for run: {PATH}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #1: build the dataset
    """dataset = TableDataset(
        "/home/nehaj/infinity/altair/clean_data/", ground_truth.adj_matrices)"""
    
    """dataset = TableDataset(
        "/home/nehaj/infinity/altair/clean_data/")"""

    dataset =SemanticTableDataset(
        "/home/nehaj/infinity/altair/clean_data/")
    
    #2: Split the dataset
    validation_split = .1
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    

    #3: Build the dataloaders and the dataloader dict
    train_dataloader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset, batch_size=1, sampler=valid_sampler)
    dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader}

    #4: Define the model
    model = NeuralNet(in_features=852).to(device)
    print(model)

    #5: Define the criterion and optimizer
    lr = 3e-5
    weight = torch.tensor([0.99]) #torch.tensor([0.01, 0.99])
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight) #nn.CrossEntropyLoss(weight=weight) #nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #5: Train loop
    epochs = 10
    

    history = {'train_loss': list(), 'train_acc': list(), 'valid_loss': list(), 'valid_acc': list()}
    for epoch in range(epochs):
        val_count =0
        train_count =0
        for i, phase in enumerate(['train', "valid"]):
            if phase == 'train':
                model.train()
                print("TRAINING: MODEL SET TO TRAIN MODE.")
            elif phase == 'valid':
                model.eval()
                print("VALIDATION: MODEL SET TO EVAL MODE.")
            epoch_loss = 0
            epoch_acc = 0
            
            
            #print(f"[{epoch+1} / {epochs}]")
            tqdm_dataloader = tqdm(dataloader_dict[phase], leave=False, colour='blue')
            for data in tqdm_dataloader:
                # data is a tuple of tensors: (x1_features, x2_features, agg_features, gr_t)
                x1_final_features, x2_final_features, agg_features, x1_semantic_tensor, x2_semantic_tensor, gr_t = data
                #print(x1_features.shape)
                
                #print(x1_semantic_tensor.shape)
                #print(x2_semantic_tensor.shape)
                #
                # 
                # print(agg_features_dataset.shape)
                #print(gr_t.shape)
                #print(gr_t)
                #print(b_gr_t.shape)
                #print(b_gr_t)
                # concat to build the input to the neural net and push to device
                X = torch.cat((x1_final_features, x2_final_features, agg_features, x1_semantic_tensor, x2_semantic_tensor), dim=1).float()
                #print(X.shape)
                X = X.to(device)
                gr_t = gr_t.float().to(device)
                # forward pass
                preds = model(X)
                #print(f"predicted label: {F.softmax(preds).argmax()}")
                #print(f"gt: :{gr_t}")
                #print(f"gt == PL?: {gr_t == F.softmax(preds).argmax()}")
                if ( gr_t == F.softmax(preds).argmax() and phase == "train"):
                    train_count+=1
                if ( gr_t == F.softmax(preds).argmax() and phase == "valid"):
                    val_count+=1
                #print(preds)
                #print(preds.shape)
                #print(F.softmax(preds))
                if gr_t[0] == 1:
                    print(f"Predicted prob: {F.sigmoid(preds)} | Predicted label: {torch.round(F.sigmoid(preds))} | GT: {gr_t} | Same : {gr_t == torch.round(F.sigmoid(preds))}")
                #print(gr_t)
                #print(gr_t == F.softmax(preds).argmax())
                #print(f"input : {X}")
                #print(f"ground_truth: {gr_t.item()}")
                #print(f"predictions: {preds.item()}")
                # Calculate loss
                loss = criterion(preds, gr_t.unsqueeze(1))#gr_t.unsqueeze(1))
                # backprop (train only)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # track loss
                epoch_loss += loss.item()
                # calculate batch accuracy and keep track of it.
                #print(f"BATCH ACC: {cal_acc(preds, gr_t.unsqueeze(1))}")
                epoch_acc += cal_acc(preds, gr_t.unsqueeze(1))
                # tqdm progress bar update
                tqdm_dataloader.set_description(f"Epoch [{epoch + 1} / {epochs}]")
                tqdm_dataloader.set_postfix(batch_loss = loss.item())
            
            # print the epoch stats
            print(f"{phase.upper()} : [{epoch+1} / {epochs}] Loss: {epoch_loss/len(dataloader_dict[phase]):.4f} | Acc: {epoch_acc/len(dataloader_dict[phase]):.4f}%")
            #print(len(dataloader_dict[phase]))

            if phase == 'train':
                try:
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        }, PATH+f"{epoch+1}.pth")
                    #print(f"Model saved after epoch:{epoch+1}")
                except Exception as e:
                    print(e)
            
            history[f'{phase}_loss'].append(round(epoch_loss,4))
            history[f'{phase}_acc'].append(round(epoch_acc.item(),2))

        print(f"{train_count/(len(dataloader_dict['train']))}")
        print(f"{val_count/(len(dataloader_dict['valid']))}")

        #print(count/count1*100)
    # 6: Save history to pickle file

    with open(f"./logs/{run_id}/history.pkl", "wb") as outfile:
        pickle.dump(history, outfile, protocol=-1)



if __name__ == '__main__':
    main()

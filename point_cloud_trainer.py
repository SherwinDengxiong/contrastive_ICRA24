import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

import torch_geometric
import plotly.express as px


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
import tqdm
from pytorch_metric_learning.losses import NTXentLoss
import numpy as np
from dataloader import PointCloudDataloader,PointCloudDataset
import os
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

augmentation = T.Compose([T.RandomJitter(0.03), T.RandomFlip(1), T.RandomShear(0.2)])

class Model(torch.nn.Module):
    def __init__(self, k=20, aggr='max'):
        super().__init__()
        # Feature extraction
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        # Encoder head 
        self.lin1 = Linear(128 + 64, 128)
        # Projection head (See explanation in SimCLRv2)
        self.mlp = MLP([128, 256, 32], norm=None)

    def forward(self, data, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = augmentation(data)
            augm_2 = augmentation(data)

            # Extract properties
            pos_1, batch_1 = augm_1.pos, augm_1.batch
            pos_2, batch_2 = augm_2.pos, augm_2.batch

            # Get representations for first augmented view
            x1 = self.conv1(pos_1, batch_1)
            x2 = self.conv2(x1, batch_1)
            h_points_1 = self.lin1(torch.cat([x1, x2], dim=1))

            # Get representations for second augmented view
            x1 = self.conv1(pos_2, batch_2)
            x2 = self.conv2(x1, batch_2)
            h_points_2 = self.lin1(torch.cat([x1, x2], dim=1))
            
            # Global representation
            h_1 = global_max_pool(h_points_1, batch_1)
            h_2 = global_max_pool(h_points_2, batch_2)
        else:
            x1 = self.conv1(data.pos, data.batch)
            x2 = self.conv2(x1, data.batch)
            h_points = self.lin1(torch.cat([x1, x2], dim=1))
            return global_max_pool(h_points, data.batch)

        # Transformation for loss function
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)
        return h_1, h_2, compact_h_1, compact_h_2
def plot_3d_shape(shape):
    # print("Number of data points: ", shape.x.shape[0])
    x = shape.pos[:, 0]
    y = shape.pos[:, 1]
    z = shape.pos[:, 2]
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.3)
    fig.show()

def train(model,data_loader,optimizer,loss_func,device,dataset_length):
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(data_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        # Get data representations
        h_1, h_2, compact_h_1, compact_h_2 = model(data)
        # Prepare for loss
        embeddings = torch.cat((compact_h_1, compact_h_2))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / dataset_length

def main():
    epochstart=18
    filepath="./raw"
    file_list=os.listdir(filepath)
    print(file_list)
    filename_list=["train-00000-of-00025-e0ad045cbeacfebc.parquet","train-00011-of-00025-94737b0c76f2cf37.parquet"]
    PointCloudDataloader1=PointCloudDataloader(file_list,filepath)
    object_clouds=PointCloudDataloader1.get_object_clouds()
    
   
    dataset=PointCloudDataset(object_clouds,augmentation)
    dataset_length=PointCloudDataloader1.get_dataset_len()
    print("Dataset loaded successfully")
    batch_size=32

    data_loader=DataLoader(dataset,batch_size,shuffle=True)

    
    loss_func = NTXentLoss(temperature=0.10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)

    loaddict="saved_model_"+str(epochstart)+".pth"
    model.load_state_dict(torch.load(loaddict))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    print("start training")
    for epoch in range(epochstart, epochstart+4):
        loss = train(model,data_loader,optimizer,loss_func,device,dataset_length)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
        scheduler.step()
        if epoch%2==0:
            dict_name="saved_model_"+str(epoch)+".pth"
            torch.save(model.state_dict(),dict_name)
    
    sample = next(iter(data_loader))

    # Get representations
    h = model(sample.to(device), train=False)
    h = h.cpu().detach()
    labels = sample.category.cpu().detach().numpy()

    # Get low-dimensional t-SNE Embeddings
    h_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(h.numpy())

    # Plot
    ax = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1], hue=labels, 
                        alpha=0.5, palette="tab10")

    # Add labels to be able to identify the data points
    annotations = list(range(len(h_embedded[:,0])))

    def label_points(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.02, point['y'], str(int(point['val'])))

    label_points(pd.Series(h_embedded[:,0]), 
                pd.Series(h_embedded[:,1]), 
                pd.Series(annotations), 
                plt.gca()) 
    
    def sim_matrix(a, b, eps=1e-8):
        """
        Eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    similarity = sim_matrix(h, h)
    max_indices = torch.topk(similarity, k=2)[1][:, 1]
    max_vals  = torch.topk(similarity, k=2)[0][:, 1]

    # Select index
    idx = 17
    similar_idx = max_indices[idx]
    print(f"Most similar data point in the embedding space for {idx} is {similar_idx}")

    plot_3d_shape(sample[idx].cpu())
    plot_3d_shape(sample[similar_idx].cpu())

if __name__=="__main__":
    main()

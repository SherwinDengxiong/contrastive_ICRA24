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
from dataloader import PointCloudDataloader,PointCloudDataset,DemoPointCloudDataloader
from point_cloud_trainer import Model
import os
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

augmentation = T.Compose([T.RandomJitter(0.03), T.RandomFlip(1), T.RandomShear(0.2)])
def plot_3d_shape(shape):
    # print("Number of data points: ", shape.x.shape[0])
    x = shape.pos[:, 0]
    y = shape.pos[:, 1]
    z = shape.pos[:, 2]
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.3)
    fig.show()
def main():
    filepath="./src/FastSAM/trajectory_dict/information"
    demonstration_data="./src/FastSAM/weights/demo_label.csv"

    filename_list=[f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath,f))]
    PointCloudDataloader1=DemoPointCloudDataloader(filename_list,filepath,demonstration_data)
    object_clouds=PointCloudDataloader1.get_object_clouds()
    
   
    dataset=PointCloudDataset(object_clouds,augmentation)
    dataset_length=PointCloudDataloader1.get_dataset_len()
    print("Dataset loaded successfully")
    batch_size=6

    data_loader=DataLoader(dataset,batch_size,shuffle=True)

    
    loss_func = NTXentLoss(temperature=0.10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    loaddict="./src/FastSAM/weights/saved_model_34.pth"
    model.load_state_dict(torch.load(loaddict))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # print("start training")
    # for epoch in range(1, 11):
    #     loss = train(model,data_loader,optimizer,loss_func,device,dataset_length)
    #     print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    #     scheduler.step()
    #     if epoch%5==0:
    #         dict_name="saved_model_"+str(epoch)+".pth"
    #         torch.save(model.state_dict(),dict_name)
    
    sample = next(iter(data_loader))
    print(sample)

    # Get representations
    h = model(sample.to(device), train=False)
    h = h.cpu().detach()
    labels = sample.category.cpu().detach().numpy()
    print(labels)

    # # Get low-dimensional t-SNE Embeddings
    # h_embedded = TSNE(n_components=2, learning_rate='auto',
    #                 init='random').fit_transform(h.numpy())

    # # Plot
    # ax = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1], hue=labels, 
    #                     alpha=0.5, palette="tab10")

    # # Add labels to be able to identify the data points
    # annotations = list(range(len(h_embedded[:,0])))

    # def label_points(x, y, val, ax):
    #     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    #     for i, point in a.iterrows():
    #         ax.text(point['x']+.02, point['y'], str(int(point['val'])))

    # label_points(pd.Series(h_embedded[:,0]), 
    #             pd.Series(h_embedded[:,1]), 
    #             pd.Series(annotations), 
    #             plt.gca()) 
    
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
    print(similarity)

    # Select index
    idx = 0
    similar_idx = max_indices[idx]
    print("the similarity index is {}".format(max_vals))
    print("the similarity index is {}".format(max_indices))
    # print(f"Most similar data point in the embedding space for {idx} is {similar_idx}")

    # plot_3d_shape(sample[idx].cpu())
    # plot_3d_shape(sample[similar_idx].cpu())
    
    for idx in range(len(max_indices)):
        
        similar_idx = max_indices[idx]
        
        print(f"Most similar data point in the embedding space for {idx} is {similar_idx}")
    

if __name__=="__main__":
    main()

## Graph Neural Network for phase field damage

# Import necessary modules
import numpy as np
import torch
import sys
import networkx as nx
import matplotlib.pyplot as plt

import torch_geometric as torchG
from torch.nn import Module, Linear, ReLU, Tanh, Sigmoid
from torch.nn import MSELoss
from torch_geometric.nn import GATConv, global_max_pool
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import copy
import vtk

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

## Read data
print("Loading data")
# Graph nodal features - psi_active
E = 121.15e3
graph_psi_active = np.loadtxt("data/psi_active_data.csv",delimiter=",")
graph_psi_active = np.log(graph_psi_active + 1.e-9)
np.random.shuffle(graph_psi_active)
# Graph prediction data - damage
graph_damage = np.loadtxt("data/damage_data.csv",delimiter=",")
np.random.shuffle(graph_damage)
# Nodal coordinates
coords = np.loadtxt("data/position_data.csv",delimiter=",")

## Construct graph adjacency
print("Constructing graph adjacency")
# length scale param
ls = 0.05
# Number of nodes in the graph
num_nodes = np.shape(coords)[1]
# Initialize edges list
edges = []
# Populate edges list
for i in range(0,num_nodes):
  for j in range(i+1,num_nodes):
    dist = np.linalg.norm(coords[:,i] - coords[:,j])
    if dist < ls:
      edges.append([i,j])
      edges.append([j,i])

edges = np.asarray(np.transpose(edges))
edges = torch.tensor(edges)

## Create data loader
print("Creating data loader")
# We use 90% of the dataset for training and 10% for testing
train_graph_list = []
test_graph_list = []
for i in range(200):
  x_i = torch.tensor(graph_psi_active[i,:]).reshape(-1,1)
  y_i = torch.tensor(graph_damage[i,:]).reshape(-1,1)
  train_graph_list.append(torchG.data.Data(x=x_i,edge_index=edges,y=y_i))
for i in range(800,975):
  x_i = torch.tensor(graph_psi_active[i,:]).reshape(-1,1)
  y_i = torch.tensor(graph_damage[i,:]).reshape(-1,1)
  test_graph_list.append(torchG.data.Data(x=x_i,edge_index=edges,y=y_i))
# graph data loader with list of graph data
loader = torchG.data.DataLoader(train_graph_list, batch_size=128, shuffle=True)

## Define graph autoencoder
class MyGNN(torch.nn.Module):
  # Initialization
  def __init__(self, num_features, num_nodes, latent_dim=128):
    super(MyGNN, self).__init__()
    # Encoding layers
    self.gnn1_e = GATConv(num_features, 16)
    self.gnn2_e = GATConv(16, 32)
    self.gnn3_e = GATConv(32, 64)
    self.fc_e = Linear(64, latent_dim)
    self.pool = global_max_pool
    # Decoding layers
    self.gnn1_d = GATConv(latent_dim, 64)
    self.gnn2_d = GATConv(64,16)
    self.gnn3_d = GATConv(16, num_features)
    self.fc_d = Linear(latent_dim, latent_dim*num_nodes)
    # Miscellaneous
    self.act = ReLU()
    self.sigmoid = Sigmoid()
    self.latent_dim = latent_dim
    self.num_nodes = num_nodes
    # Initialize weights
    self.apply(self._init_weights)

  # defining weight initialization
  def _init_weights(self, module):
    if isinstance(module, torch.nn.Linear):
      torch.nn.init.xavier_uniform_(module.weight)
      module.bias.data.fill_(0.01)

  # Encoder process
  def encode(self, x, edge_index, batch):
    h = self.gnn1_e(x,edge_index)
    h = self.act(h)
    h = self.gnn2_e(h,edge_index)
    h = self.act(h)
    h = self.gnn3_e(h,edge_index)
    h = self.act(h)
    h = self.pool(h,batch)
    h = self.fc_e(h)

    return h

  # Decoder process
  def decode(self, x, edge_index):
    h = self.fc_d(x)
    h = h.reshape(-1,self.latent_dim)
    h = self.act(h)
    h = self.gnn1_d(h,edge_index)
    h = self.act(h)
    h = self.gnn2_d(h,edge_index)
    h = self.act(h)
    h = self.gnn3_d(h,edge_index)
    h = self.sigmoid(h)

    return h

  # Feed-forward autoencoder process
  def forward(self, x, edge_index, batch):
    encoded = self.encode(x, edge_index, batch)
    decoded = self.decode(encoded, edge_index)

    return decoded

## Training configurations
print("Defining training configurations")
# Model
model = MyGNN(num_features=1,num_nodes=num_nodes)
# Define the optimizer
opt = Adam(model.parameters(), lr=2.e-3)
# Learning rate scheduler
sch = StepLR(opt, step_size=500, gamma=0.8)
# Define the epochs
EPOCHS = 2

## Training
print("Beginning training")
# Best loss storage
best_loss = np.inf
best_states = None
loss_history = []
loss_fn = MSELoss()
# Training loop
for t in range(EPOCHS):
  train_losses = []
  model.train()
  for batch in loader:
    pred = model(batch.x, edge_index=batch.edge_index, batch=batch.batch)
    loss = loss_fn(pred,batch.y)
    # automatic gradients
    opt.zero_grad()
    loss.backward(retain_graph=False)
    # Back propagation via the optimizer
    opt.step()
    # assemble the epoch loss
    train_losses.append(loss.detach().item())

  # cache the NN states with the best performance
  epoch_loss = np.mean(train_losses)
  epoch_loss_max = np.amax(train_losses)
  if epoch_loss < best_loss:
    best_loss = np.copy(epoch_loss)
    best_states = copy.deepcopy(model.state_dict())

  # record loss history
  if (t+1) % 1 == 0:
    print("epoch {:4d}, training mean loss = {:1.5e} and max loss = {:1.5e}".format(t+1, epoch_loss,epoch_loss_max))
  loss_history.append(epoch_loss)

  # adjust learning rate
  sch.step()

# Load the trained NN
model.eval()
model.load_state_dict(best_states)

## Prediction on test data
test_mse = list()
test_loader = torchG.data.DataLoader(test_graph_list, batch_size=1, shuffle=True)
test_pred = list()
sim_res = list()
for batch in test_loader:
  pred = model(batch.x,edge_index=batch.edge_index,batch=batch.batch)
  test_mse.append(loss_fn(pred,batch.y).detach().item())
  test_pred.append(pred.detach().numpy())
  sim_res.append(batch.y.detach().numpy())

test_loss = np.mean(test_mse)
print("Average test loss = {:1.5e}".format(test_loss))

## Function to write to vtk file for visualization
def write_vtk_file(filename, xy, point_data):
  # Open file and create unstructured grid
  writer = vtk.vtkUnstructuredGridWriter()
  writer.SetFileName(filename)
  grid = vtk.vtkUnstructuredGrid()

  # Set point coordinates
  vtk_points = vtk.vtkPoints()
  for i,(x,y) in enumerate(xy):
    vtk_points.InsertNextPoint(x,y,0.0)
  grid.SetPoints(vtk_points)

  # Set point data
  vtk_point_scalars = vtk.vtkDoubleArray()
  vtk_point_scalars.SetName("damage")
  for i in point_data:
    vtk_point_scalars.InsertNextValue(i)
  grid.GetPointData().AddArray(vtk_point_scalars)

  # Write to file
  writer.SetInputData(grid)
  writer.Write()

## Write to vtk file
write_vtk_file("out_pred1.vtk", np.transpose(coords), test_pred[0].reshape(-1,1))
write_vtk_file("out_sim1.vtk", np.transpose(coords), sim_res[0].reshape(-1,1))

write_vtk_file("out_pred2.vtk", np.transpose(coords), test_pred[1].reshape(-1,1))
write_vtk_file("out_sim2.vtk", np.transpose(coords), sim_res[1].reshape(-1,1))

## Plot training loss
it = np.asarray(range(EPOCHS))
plt.figure(0)
plt.plot(it,loss_history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("loss_evolution.eps")

## Plot test loss histogram
plt.figure(1)
plt.hist(test_mse)
plt.xlabel("test MSE")
plt.ylabel("frequency")
plt.savefig("test_mse.eps")

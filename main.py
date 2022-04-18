# from mpnn import *
# from dataset import *


# # Create BP tree dataset
# tree_dataset = BPDataset(root='data/',num_data=3000, loop=False)
# train_dataset, val_dataset, test_dataset = split(tree_dataset, 3)

# # Create loopy graph datatset
# loop_dataset = BPDataset(root='data/',num_data=1000, loop=True)

# # Train on tree
# model = MPNN_2Layer(x_dim=3)
# model_name = type(model).__name__
# best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
#     model, 
#     model_name, 
#     train_loader,
#     val_loader, 
#     test_loader,
#     n_epochs=200
# )

# #Test on loopy graph
# eval(model,....)
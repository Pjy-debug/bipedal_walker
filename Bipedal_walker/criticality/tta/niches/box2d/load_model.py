import torch

model_path = 'mlp.ep142'
print('loading model...')
print(model_path)
criticality_model = torch.load(model_path)
criticality_model.eval()
print("Successfully loading model!")
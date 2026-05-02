"""
Filename: model_steganography.py

Description:
Model Steganography class for embedding and extracting secret information
into/from model parameters.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from utils.util import get_secretbits, bch_decode, modify_distribution, interpolate


class ModelSteganography:
    def __init__(self, init_function, size=128, batch_size=64, target_var=1e-3, min_nums=1000):
        '''
        :param init_function: Initialization method to be used.
        :param size: Size configuration for the encoder/decoder.
        :param batch_size: Batch size for encoder/decoder operations.
        :param target_var: Minimum variance threshold for parameter selection.
        :param min_nums: Minimum number of parameters for embedding.
        '''
        self.size = size
        self.batch_size = batch_size
        self.target_var = target_var
        self.min_nums = min_nums
        self.init_function = init_function

    def encode(self, model: torch.nn.Module) -> (torch.Tensor, torch.Tensor):
        '''
        Encodes secret information into the target model's parameters.
        Requires encoder models to be present in the 'models/' directory.

        Args:
             model (torch.nn.Module): The model to embed with secret information.

        Returns:
            torch.Tensor: The raw embedded secret bits.
            torch.Tensor: The BCH-decoded secret bits.
        '''
        secret_bits_encoder = torch.load(f"models/encoder{self.size}.pth").train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        secret_bits_bch_arr = []
        secret_bits_arr = []
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
                    # Get variance for this layer
                    weight_var, bias_var = self.init_function(m)

                    # Count parameters in this layer
                    if hasattr(m, 'bias') and m.bias is not None and bias_var > self.target_var:
                        params_num = m.weight.numel() + m.bias.numel()
                    else:
                        params_num = m.weight.numel()

                    # Skip layers with too few parameters
                    if params_num < self.min_nums:
                        continue

                    # Skip layers with variance below threshold
                    if weight_var < self.target_var:
                        continue

                    # Generate secret bits
                    secret_bits, secret_bits_bch = get_secretbits(params_num)
                    secret_bits_bch_arr.append(secret_bits_bch)
                    secret_bits_arr.append(secret_bits)

                    # Batch generation of parameters
                    dataset = TensorDataset(secret_bits)
                    data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                    original_params_list = []
                    for batch in data_loader:
                        original_params = secret_bits_encoder(batch[0].to(device))
                        original_params_list.append(original_params)
                    original_params = torch.concatenate(original_params_list)
                    original_params = F.adaptive_max_pool1d(original_params.view(1, -1), params_num).view(-1)

                    # Inject parameters into the model
                    if hasattr(m, 'bias') and m.bias is not None and bias_var > self.target_var:
                        new_bias = modify_distribution(original_params[0:m.bias.numel()], bias_var)
                        m.bias = nn.Parameter(new_bias)
                        new_weight = modify_distribution(original_params[m.bias.numel():params_num], weight_var)
                        m.weight = nn.Parameter(new_weight.reshape(m.weight.shape))
                    else:
                        original_params = modify_distribution(original_params, weight_var)
                        m.weight = nn.Parameter(original_params.reshape(m.weight.shape))

                elif isinstance(m, (nn.LSTM, nn.RNN)):
                    weight_params = {name: param for name, param in m.named_parameters() if 'weight' in name}

                    for key, value in weight_params.items():
                        bias_name = key.replace("weight", "bias")
                        if hasattr(m, bias_name):
                            params_num = value.numel() + getattr(m, bias_name).numel()
                        else:
                            params_num = value.numel()

                        if params_num < self.min_nums:
                            continue
                        
                        var = torch.var(getattr(m, key)).item()
                        if var < self.target_var:
                            continue
                        
                        secret_bits, secret_bits_bch = get_secretbits(params_num)
                        secret_bits_bch_arr.append(secret_bits_bch)
                        secret_bits_arr.append(secret_bits)

                        dataset = TensorDataset(secret_bits)
                        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                        original_params_list = []
                        for batch in data_loader:
                            original_params = secret_bits_encoder(batch[0].to(device))
                            original_params_list.append(original_params)
                        original_params = torch.concatenate(original_params_list)
                        original_params = F.adaptive_max_pool1d(original_params.view(1, -1), params_num).view(-1)
                        original_params = modify_distribution(original_params, var)

                        if hasattr(m, bias_name):
                            setattr(m, bias_name, nn.Parameter(original_params[:getattr(m, bias_name).numel()]))
                            setattr(m, key,
                                    nn.Parameter(
                                        original_params[getattr(m, bias_name).numel():].reshape(getattr(m, key).shape)))
                        else:
                            setattr(m, key, nn.Parameter(original_params.reshape(getattr(m, key).shape)))

        secret_bits_bch_tensor = torch.concatenate(secret_bits_bch_arr)
        secret_bits_tensor = torch.concatenate(secret_bits_arr)
        del secret_bits_encoder
        return secret_bits_tensor.view(-1), secret_bits_bch_tensor

    def decode(self, model: torch.nn.Module) -> (torch.Tensor, torch.Tensor):
        '''
        Decodes secret information from a model's parameters.
        Requires decoder models to be present in the 'models/' directory.

        Args:
            model (torch.nn.Module): The model to extract information from.

        Returns:
            torch.Tensor: The extracted raw secret bits.
            torch.Tensor: The BCH-decoded secret bits.
        '''
        secret_bits_decoder = torch.load(f"models/decoder{self.size}.pth").train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        outputs_arr_bch = []
        outputs_arr = []
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
                    weight_var, bias_var = self.init_function(m)
                    
                    if hasattr(m, 'bias') and m.bias is not None and bias_var > self.target_var:
                        params_num = m.weight.numel() + m.bias.numel()
                    else:
                        params_num = m.weight.numel()
                    
                    if params_num < self.min_nums:
                        continue
                    
                    if weight_var < self.target_var:
                        continue
                    
                    if hasattr(m, 'bias') and m.bias is not None and bias_var > self.target_var:
                        last_params_tensor = torch.concatenate([m.bias, m.weight.reshape(-1)])
                    else:
                        last_params_tensor = m.weight.reshape(-1)

                    last_params_tensor = interpolate(last_params_tensor).view(-1, 1024)

                    dataset = TensorDataset(last_params_tensor)
                    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                    output_list = []
                    for batch in dataloader:
                        outputs = secret_bits_decoder(batch[0].to(device)).to('cpu')
                        output_list.append(outputs)
                    outputs = torch.concatenate(output_list)

                    predictions = (outputs > 0.5).float()

                    outputs_arr.append(predictions)
                    predictions = bch_decode(predictions.detach().numpy())
                    outputs_arr_bch.append(predictions)

                elif isinstance(m, (nn.LSTM, nn.RNN)):
                    weight_params = {name: param for name, param in m.named_parameters() if 'weight' in name}
                    for key, value in weight_params.items():
                        bias_name = key.replace("weight", "bias")

                        if hasattr(m, bias_name):
                            params = value.numel() + getattr(m, bias_name).numel()
                        else:
                            params = value.numel()
                        if params < self.min_nums:
                            continue

                        if hasattr(m, bias_name):
                            last_params_tensor = torch.tensor(
                                [*getattr(m, bias_name).detach().tolist(),
                                 *getattr(m, key).detach().reshape(-1).tolist()],
                                dtype=torch.float32).to(device)
                        else:
                            last_params_tensor = torch.tensor([*getattr(m, key).detach().reshape(-1).tolist()],
                                                              dtype=torch.float32).to(device)

                        last_params_tensor = modify_distribution(last_params_tensor, var=1).view(1, 1, -1)
                        last_params_tensor = interpolate(last_params_tensor).view(-1, 1024)

                        dataset = TensorDataset(last_params_tensor)
                        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                        output_list = []
                        for batch in dataloader:
                            outputs = secret_bits_decoder(batch[0].to(device)).to('cpu')
                            output_list.append(outputs)
                        outputs = torch.concatenate(output_list)

                        predictions = (outputs > 0.5).float()

                        outputs_arr.append(predictions)
                        predictions = bch_decode(predictions.detach().numpy())
                        outputs_arr_bch.append(predictions)

        outputs_tensor = torch.concatenate(outputs_arr)
        outputs_tensor_bch = torch.concatenate(outputs_arr_bch)
        return outputs_tensor.view(-1), outputs_tensor_bch.view(-1)

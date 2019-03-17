import numpy as np
import torch


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x
    
    
def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
    
    
def forward(model, generate_func, cuda, return_input=False, 
    return_target=False):
    '''Forward data to model in mini-batch. 
    
    Args: 
      model: object
      generate_func: function
      cuda: bool
      return_input: bool
      return_target: bool
      max_validate_num: None | int, maximum mini-batch to forward to speed up 
          validation
    '''
    output_dict = {}
    
    # Evaluate on mini-batch
    for batch_data_dict in generate_func:

        # Predict
        batch_feature = move_data_to_gpu(batch_data_dict['feature'], cuda)
        
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_feature)

        append_to_dict(dict=output_dict, key='audio_name', 
            value=batch_data_dict['audio_name'])
        
        append_to_dict(dict=output_dict, key='output', 
            value=batch_output.data.cpu().numpy())
            
        if return_input:
            append_to_dict(dict=output_dict, key='feature', 
                value=batch_data_dict['feature'])
            
        if return_target:
            if 'fine_target' in batch_data_dict.keys():
                append_to_dict(dict=output_dict, key='fine_target', 
                    value=batch_data_dict['fine_target'])
                
            if 'coarse_target' in batch_data_dict.keys():
                append_to_dict(dict=output_dict, key='coarse_target', 
                    value=batch_data_dict['coarse_target'])
                
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict
def is_error(model_path = "mlsquare/pico_seshu", adapter = "model.layers.3.x_proj"):
    print("CHECKING ERROR")
    model = MambaForCausalLM.from_pretrained(model_path)
    A_B = []
    for param in model.parameters():
        param.data.fill_(0)
    model = load_model_with_LoRA(model, [adapter])
    for name, param in model.named_parameters():
        if param.requires_grad:
            A_B.append(param)
    
    for name, param in model.named_parameters():
        if 'lora_B' in name:
            parameter = dict(model.named_parameters())[name]
            init.normal_(parameter, mean=0, std=1)
            
    AB = A_B[1]@A_B[0]
    model = model.merge_and_unload()
    non_zero_matrices = []
    for param in model.parameters():
        sum_param = param.sum().item()
        if sum_param != 0:
            non_zero_matrices.append(param.clone().detach())
            
    if (len(non_zero_matrices) == 1) and (AB == non_zero_matrices[0]).all():
        return False
    return True

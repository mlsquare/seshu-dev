def evaluation(data, model, tokenizer, batch_size=32, max_length = 64):
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    total_loss = 0
    
    with torch.no_grad():
        model.eval()
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_data = data['tgt'][start_idx:end_idx]
            inputs = [tokenizer.encode(datum, return_tensors="pt",truncation=True, padding='max_length', max_length=max_length) for datum in batch_data]
            input_ids = torch.cat(inputs, dim=0)
            
            loss = compute_loss(model, input_ids)
            total_loss += loss.item() * (end_idx - start_idx)
            
    avg_loss = total_loss / num_samples
    print("LOSS: ", avg_loss)
    return avg_loss

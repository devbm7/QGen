def fine_tune_model(model, tokenizer):
    import json
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AdamW, get_linear_schedule_with_warmup

    # Load feedback data
    with open('feedback_data.json', 'r') as f:
        data = json.load(f)

    # Filter high-quality questions (overall_rating >= 4)
    high_quality_data = [item for item in data if item['overall_rating'] >= 4]

    class FeedbackDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            input_text = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer: {item['answer']}"
            target_text = item['question']
            
            inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            targets = tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            
            return {
                "input_ids": inputs.input_ids.squeeze(),
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": targets.input_ids.squeeze()
            }

    dataset = FeedbackDataset(high_quality_data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader))

    model.train()
    for epoch in range(3):  # 3 epochs for fine-tuning
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    return model

# Call this function periodically or when a certain amount of new feedback is collected
# model = fine_tune_model(model, tokenizer)
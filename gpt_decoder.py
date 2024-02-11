from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class HotelExplanationDataset(Dataset):
    def __init__(self, input_texts, output_texts, tokenizer, max_length=512):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]

        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True
        )

        outputs = self.tokenizer.encode(
            output_text, 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True
        )

        return {
            "input_ids": inputs[0],
            "labels": outputs[0],
        }

    @staticmethod
    def collate_fn(batch):
        input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
        labels = pad_sequence([item['labels'] for item in batch], batch_first=True)

        max_len = max(input_ids.shape[1], labels.shape[1])

        input_ids = torch.nn.functional.pad(input_ids, (0, max_len - input_ids.shape[1]))
        labels = torch.nn.functional.pad(labels, (0, max_len - labels.shape[1]))

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

input_texts = ["romantic hotels","affordable hotel","Times Square location", "hotel in New York City", "luxury hotel in New York", "city hotel","seaside oasis hotel in NYC"]
output_texts = ["A good romantic getaway in New York City, spend time with your loved one or ones! There is plenty of time to be with your partner and explore the heart of the city.","If you want a cost efficient trip to New York City, check this place out! It is the best bang for your buck, in terms of location and amenities!","This hotel is lcoated in the heart of New York City, with a Times Square location and convenient subway locations!","This hotel in New York City is very nice and amazing, suitable for all your needs!", "If you're looking for a luxury experience in Paris, this hotel has high-quality amenities and state of the art facilities!","This hotel is walking distance from tourist destinations...","This hotel is perfect for tropical vibes, with nice views and easy access! It's convenient and beautiful, right in prime location!"]

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

dataset = HotelExplanationDataset(input_texts, output_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=HotelExplanationDataset.collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.save_pretrained("hotel_explanation_model")
tokenizer.save_pretrained("hotel_explanation_model")


def generateText(data):
    model = GPT2LMHeadModel.from_pretrained("hotel_explanation_model")
    tokenizer = GPT2Tokenizer.from_pretrained("hotel_explanation_model")

    model.eval()

    input_text = data

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids, max_length=80, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    sentences = generated_text.split('. ')
    if len(sentences) > 1:
        generated_text = '. '.join(sentences[:-1]) + '.'

    return generated_text
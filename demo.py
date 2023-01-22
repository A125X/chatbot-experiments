# !pip install transformers sentencepiece
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rut5-small-chitchat2")

model = AutoModelForSeq2SeqLM.from_pretrained("cointegrated/rut5-small-chitchat2")

while True:
    text = input()
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(
            **inputs, 
            do_sample=True, top_p=0.5, num_return_sequences=3, 
            repetition_penalty=2.5,
            max_length=32,
        )
    #for h in hypotheses:
    #    print(tokenizer.decode(h, skip_special_tokens=True))
    
    print(tokenizer.decode(hypotheses[0], skip_special_tokens=True))
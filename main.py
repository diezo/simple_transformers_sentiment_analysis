import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def compute(text: str) -> int:
    """
    Takes text and returns sentiment score between 0-1.
    """

    tokens = tokenizer.encode(text, return_tensors="pt")
    result = model(tokens)

    return round(int(torch.argmax(result.logits) + 1) * .2, 1)  # Convert Range 1-5 to 0-1


if __name__ == "__main__":
    review: str = "Guys I dont know what to say, but i find it average. its not too bad or too good. it just works."
    print("Score:", compute(review))

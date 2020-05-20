from transformers import GPT2Tokenizer


BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
MAX_LEN = 64
TRAIN_PATH = "/kaggle/input/short-jokes/shortjokes.csv"  #ADD PATH TO YOUR DATASET HERE
MODEL_FOLDER = "/kaggle/working/trained_models"  # ADD PATH TO WHERE YOU WANT TO SAVE YOUR MODEL
Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
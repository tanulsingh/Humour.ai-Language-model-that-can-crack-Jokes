from transformers import GPT2Tokenizer

class config:
  BATCH_SIZE = 16
  EPOCHS = 4
  LEARNING_RATE = 3e-5
  MAX_LEN = 64
  TRAIN_PATH = "/content/gdrive/My Drive/shortjokes.csv"
  MODEL_FOLDER = "/content/gdrive/My Drive/Colab Notebooks/trained_models"
  Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
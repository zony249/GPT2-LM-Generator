from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from time import sleep


if __name__ == "__main__":

    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    generator = pipeline("text-generation", model=gpt2, pad_token_id=50256, tokenizer=tok, device=0)
    
    inp = input("Type a sentence: ")

    i = 0
    count = 1
    while True:
        inp = generator(inp, max_length=count*100)[0]["generated_text"]

        display_text = inp.split()
        while i < len(display_text):
            for j in range(len(display_text[i])):
                print(display_text[i][j], end="", flush=True)
                sleep(0.05)
            print(" ", end="", flush=True)
            sleep(0.05)
            i += 1
        count += 1
        x = input()

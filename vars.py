from os.path import dirname, join

PROJ_DIR = dirname(dirname(dirname(__file__)))
DATA_DIR = join(PROJ_DIR, "data")


def GPT_4_TOKENIZER(x: str):
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4")
    return enc.encode(x)

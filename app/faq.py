def load_faq():
    with open("Faq.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]
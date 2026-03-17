from app.utils.text_preprocessing import split_into_sentences


def test_text_processing():

    texts = [
        "Albert Einstein was born in 1879. He developed relativity.",
        "Tesla was founded in 2003. It is based in Austin.",
        "Python is widely used. Many developers love it."
    ]

    print("\nProcessed Sentences:\n")

    for text in texts:
        sentences = split_into_sentences(text)

        print("\nInput:", text)

        for s in sentences:
            print("-", s)


if __name__ == "__main__":
    test_text_processing()
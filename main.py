from agents.word_explain_agent import explain_word

if __name__ == "__main__":
    term = "스마트팜"
    result = explain_word(term)
    print("📘 용어 설명 결과:\n")
    print(result)

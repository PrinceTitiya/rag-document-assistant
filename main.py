from src.pipeline import RAGPipeline


def main():
    rag = RAGPipeline()
    question = input("Enter your query here:\n")
    answer = rag.query(question)
    print("\nAnswer:\n")
    print(answer)

if __name__ == "__main__":
    main()
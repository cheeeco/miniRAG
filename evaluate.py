import json
from typing import List, Dict
from loguru import logger
from difflib import SequenceMatcher

from answer_questions import QASystem


class QABenchmark:
    def __init__(self, rag_system: QASystem, questions: List[Dict[str, str]]):
        self.rag_system = rag_system
        self.questions = questions
        logger.info("Initialized RAGBenchmark with {} questions.".format(len(questions)))

    def _calculate_similarity(self, generated: str, expected: str) -> float:
        return SequenceMatcher(None, generated.strip().lower(), expected.strip().lower()).ratio()

    def run(self) -> Dict[str, float]:
        total = len(self.questions)
        correct = 0
        similarities = []

        for i, qa in enumerate(self.questions):
            question = qa["question"]
            expected_answer = qa["expected_answer"]

            try:
                generated_answer = self.rag_system.invoke(question)
                similarity = self._calculate_similarity(generated_answer, expected_answer)
                similarities.append(similarity)

                if similarity > 0.9:  # Define "correct" if similarity is above 90%
                    correct += 1

                logger.info(f"Q{i + 1}: {question}")
                logger.info(f"Expected: {expected_answer}")
                logger.info(f"Generated: {generated_answer}")
                logger.info(f"Similarity: {similarity:.2f}")

            except Exception as e:
                logger.error(f"Error processing question {i + 1}: {e}")
                similarities.append(0.0)

        accuracy = correct / total
        avg_similarity = sum(similarities) / total

        metrics = {
            "accuracy": accuracy,
            "average_similarity": avg_similarity,
        }
        logger.info(f"Benchmark completed: {metrics}")
        return metrics


# Example Usage
if __name__ == "__main__":
    config_path = "base_config.json"
    questions_path = "questions.json"
    with open(questions_path, "r") as file:
        questions = json.load(file)

    rag_system = QASystem(config_path=config_path)
    benchmark = QABenchmark(rag_system=rag_system, questions=questions)

    # Run the benchmark
    results = benchmark.run()

    # Print results
    print(json.dumps(results, indent=2))

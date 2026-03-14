"""Test loading questions from CSV"""

from config import Config
from dataset import setup_datasets

# Create a config with fewer questions for testing
config = Config()
config.NUM_QUESTIONS = 100  # Load only 100 questions for testing

print("=" * 60)
print("Testing QANTA CSV Dataset Loader")
print("=" * 60)

# Load datasets
train_dataset, val_dataset, test_dataset = setup_datasets(config)

print("\n" + "=" * 60)
print("Sample Questions from Training Set")
print("=" * 60)

# Show a few sample questions
for i in range(min(3, len(train_dataset))):
    question = train_dataset.questions[i]
    print(f"\n--- Question {i+1} ---")
    print(f"ID: {question.question_id}")
    print(f"Category: {question.category}")
    print(f"Number of clues: {len(question.clues)}")
    print(f"\nClues:")
    for j, clue in enumerate(question.clues, 1):
        print(f"  {j}. {clue}")
    print(f"\nAnswer choices:")
    for j, choice in enumerate(question.answer_choices):
        marker = " ✓" if j == question.correct_answer_idx else ""
        print(f"  {chr(65+j)}. {choice}{marker}")
    print(f"\nMetadata: {question.metadata}")

print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)

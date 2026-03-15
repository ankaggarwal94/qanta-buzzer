"""
Dataset handling for Quiz Bowl questions
"""

import json
import csv
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict

from environment import Question
from config import Config


class QuizBowlDataset:
    """Dataset class for quiz bowl questions with multiple-choice answers"""
    
    def __init__(self, questions: List[Question]):
        """
        Initialize dataset with questions.
        
        Args:
            questions: List of Question objects
        """
        self.questions = questions
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Question:
        return self.questions[idx]
    
    def shuffle(self):
        """Shuffle questions in place"""
        random.shuffle(self.questions)
    
    def get_batch(self, batch_size: int) -> List[Question]:
        """Get a random batch of questions"""
        return random.sample(self.questions, min(batch_size, len(self.questions)))
    
    def save(self, filepath: str):
        """Save dataset to JSON file"""
        data = [self._question_to_dict(q) for q in self.questions]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'QuizBowlDataset':
        """Load dataset from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        questions = [cls._dict_to_question(d) for d in data]
        return cls(questions)
    
    @staticmethod
    def _question_to_dict(question: Question) -> Dict:
        """Convert Question to dictionary"""
        return {
            'question_id': question.question_id,
            'clues': question.clues,
            'answer_choices': question.answer_choices,
            'correct_answer_idx': question.correct_answer_idx,
            'category': question.category,
            'metadata': question.metadata or {}
        }
    
    @staticmethod
    def _dict_to_question(data: Dict) -> Question:
        """Convert dictionary to Question"""
        return Question(
            question_id=data['question_id'],
            clues=data['clues'],
            answer_choices=data['answer_choices'],
            correct_answer_idx=data['correct_answer_idx'],
            category=data['category'],
            metadata=data.get('metadata', {})
        )


class QANTADatasetLoader:
    """
    Load Quiz Bowl questions from QANTA CSV format.
    Generates multiple-choice questions by selecting distractors from the same category.
    """
    
    @classmethod
    def load_from_csv(cls, 
                     csv_path: str,
                     num_questions: int = None,
                     num_choices: int = 4,
                     min_clues: int = 3,
                     max_clues: int = 6,
                     seed: int = 42) -> 'QuizBowlDataset':
        """
        Load questions from QANTA CSV file.
        
        Args:
            csv_path: Path to questions.csv file
            num_questions: Number of questions to load (None = all)
            num_choices: Number of answer choices (default: 4)
            min_clues: Minimum clues to include per question
            max_clues: Maximum clues to include per question
            seed: Random seed
            
        Returns:
            QuizBowlDataset object
        """
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"Loading questions from {csv_path}...")
        
        # Load all questions from CSV
        raw_questions = []
        category_answers = defaultdict(list)  # For generating distractors
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse clues (separated by "|||")
                text = row['Text']
                clues = [clue.strip() for clue in text.split('|||')]
                
                # Store question data
                question_data = {
                    'question_id': row['Question ID'],
                    'fold': row['Fold'],
                    'answer': row['Answer'],
                    'category': row['Category'],
                    'clues': clues
                }
                raw_questions.append(question_data)
                
                # Store answer by category for distractor selection
                if row['Answer'] not in category_answers[row['Category']]:
                    category_answers[row['Category']].append(row['Answer'])
        
        print(f"Loaded {len(raw_questions)} raw questions")
        
        # Shuffle and optionally limit
        random.shuffle(raw_questions)
        if num_questions is not None:
            raw_questions = raw_questions[:num_questions]
        
        # Convert to Question objects with multiple choice
        questions = []
        for idx, raw_q in enumerate(raw_questions):
            # Get available clues
            available_clues = len(raw_q['clues'])
            if available_clues < 1:
                # Skip questions with no clues
                continue
            
            # Select number of clues to use
            if available_clues < min_clues:
                # Use all available clues if less than minimum
                num_clues = available_clues
            else:
                # Randomly select between min and max (capped by available)
                num_clues = min(
                    random.randint(min_clues, max_clues),
                    available_clues
                )
            
            clues = raw_q['clues'][:num_clues]
            
            # Generate distractors from same category
            correct_answer = raw_q['answer']
            category = raw_q['category']
            
            # Get potential distractors (exclude correct answer)
            potential_distractors = [
                ans for ans in category_answers[category] 
                if ans != correct_answer
            ]
            
            # If not enough distractors in category, use from other categories
            if len(potential_distractors) < num_choices - 1:
                other_answers = []
                for cat, answers in category_answers.items():
                    if cat != category:
                        other_answers.extend(answers)
                potential_distractors.extend(
                    random.sample(other_answers, 
                                min(num_choices - 1 - len(potential_distractors), 
                                    len(other_answers)))
                )
            
            # Sample distractors
            distractors = random.sample(
                potential_distractors, 
                min(num_choices - 1, len(potential_distractors))
            )
            
            # Create answer choices
            answer_choices = [correct_answer] + distractors
            correct_idx = 0
            
            # Shuffle choices
            shuffle_indices = list(range(len(answer_choices)))
            random.shuffle(shuffle_indices)
            answer_choices = [answer_choices[i] for i in shuffle_indices]
            correct_idx = shuffle_indices.index(0)
            
            # Pad with empty choices if needed
            while len(answer_choices) < num_choices:
                answer_choices.append(f"[No answer {len(answer_choices)}]")
            
            # Create Question object
            question = Question(
                question_id=raw_q['question_id'],
                clues=clues,
                answer_choices=answer_choices,
                correct_answer_idx=correct_idx,
                category=category,
                metadata={
                    'source': 'qanta',
                    'fold': raw_q['fold'],
                    'full_answer': correct_answer,
                    'total_clues': available_clues
                }
            )
            
            questions.append(question)
        
        print(f"Created {len(questions)} multiple-choice questions")
        
        return QuizBowlDataset(questions)


class SyntheticDatasetGenerator:
    """
    Generate synthetic quiz bowl questions for development and testing.
    Use QANTADatasetLoader for real QANTA data.
    """
    
    SAMPLE_QUESTIONS = {
        'history': [
            {
                'entity': 'Napoleon Bonaparte',
                'clues': [
                    'This military leader established the Continental System to economically isolate Britain.',
                    'He crowned himself emperor in 1804 at Notre-Dame Cathedral in Paris.',
                    'His Russian campaign of 1812 ended in catastrophic retreat from Moscow.',
                    'He was finally defeated at Waterloo in 1815 by Wellington and Blücher.',
                    'This French emperor was exiled to Elba and later to Saint Helena.',
                ],
                'distractors': ['Julius Caesar', 'Alexander the Great', 'Charlemagne']
            },
            {
                'entity': 'Abraham Lincoln',
                'clues': [
                    'This leader delivered an address at the dedication of a military cemetery in Pennsylvania.',
                    'He issued a proclamation in 1863 that changed the legal status of enslaved people.',
                    'He was assassinated by John Wilkes Booth at Ford\'s Theatre.',
                    'His debates with Stephen Douglas helped him gain national prominence.',
                    'This 16th U.S. president led the country through the Civil War.',
                ],
                'distractors': ['George Washington', 'Thomas Jefferson', 'Andrew Jackson']
            }
        ],
        'literature': [
            {
                'entity': 'The Great Gatsby',
                'clues': [
                    'This novel features a green light at the end of a dock as a central symbol.',
                    'The narrator is Nick Carraway, who moves to West Egg, Long Island.',
                    'Characters include Tom and Daisy Buchanan and Meyer Wolfsheim.',
                    'The title character throws lavish parties hoping to attract his lost love.',
                    'F. Scott Fitzgerald wrote this Jazz Age novel published in 1925.',
                ],
                'distractors': ['Tender Is the Night', 'This Side of Paradise', 'The Beautiful and Damned']
            },
            {
                'entity': 'Franz Kafka',
                'clues': [
                    'This author wrote about a man who wakes up transformed into a monstrous insect.',
                    'His novel "The Trial" features Josef K. arrested for an unknown crime.',
                    'Works like "The Castle" and "Amerika" were published posthumously.',
                    'He worked as an insurance officer in Prague while writing.',
                    'This German-language writer is known for absurdist and existential themes.',
                ],
                'distractors': ['Thomas Mann', 'James Joyce', 'Virginia Woolf']
            }
        ],
        'science': [
            {
                'entity': 'Mitochondria',
                'clues': [
                    'These organelles have their own circular DNA separate from nuclear DNA.',
                    'They are believed to have originated from endosymbiotic bacteria.',
                    'The inner membrane is folded into structures called cristae.',
                    'They produce ATP through oxidative phosphorylation.',
                    'These are often called the "powerhouse of the cell".',
                ],
                'distractors': ['Chloroplast', 'Ribosome', 'Endoplasmic Reticulum']
            },
            {
                'entity': 'Quantum Entanglement',
                'clues': [
                    'Einstein famously called this "spooky action at a distance".',
                    'Bell\'s theorem provides a way to test this phenomenon experimentally.',
                    'Measuring one particle instantly affects its correlated partner.',
                    'This property is exploited in quantum computing and cryptography.',
                    'This quantum mechanical phenomenon links particle states non-locally.',
                ],
                'distractors': ['Wave Function Collapse', 'Superposition', 'Decoherence']
            }
        ],
        'arts': [
            {
                'entity': 'The Starry Night',
                'clues': [
                    'This painting features a prominent cypress tree in the foreground.',
                    'It depicts a view from an asylum window in Saint-Rémy-de-Provence.',
                    'Swirling patterns dominate the night sky in this post-impressionist work.',
                    'It was painted in 1889, one year before the artist\'s death.',
                    'Vincent van Gogh created this iconic painting.',
                ],
                'distractors': ['Café Terrace at Night', 'The Night Watch', 'Nighthawks']
            }
        ]
    }
    
    @classmethod
    def generate_dataset(cls, 
                        num_questions: int = 500,
                        category_distribution: Dict[str, float] = None,
                        min_clues: int = 4,
                        max_clues: int = 6,
                        seed: int = 42) -> QuizBowlDataset:
        """
        Generate synthetic dataset.
        
        Args:
            num_questions: Total number of questions to generate
            category_distribution: Distribution of categories
            min_clues: Minimum clues per question
            max_clues: Maximum clues per question
            seed: Random seed
            
        Returns:
            QuizBowlDataset object
        """
        random.seed(seed)
        np.random.seed(seed)
        
        if category_distribution is None:
            category_distribution = Config.CATEGORY_DISTRIBUTION
        
        questions = []
        
        # Calculate questions per category
        categories = list(category_distribution.keys())
        category_counts = {cat: int(num_questions * prob) 
                          for cat, prob in category_distribution.items()}
        
        # Adjust for rounding errors
        total = sum(category_counts.values())
        if total < num_questions:
            category_counts[categories[0]] += num_questions - total
        
        # Generate questions
        question_id = 0
        for category, count in category_counts.items():
            for _ in range(count):
                # Sample a template from this category
                template = random.choice(cls.SAMPLE_QUESTIONS.get(category, 
                                        cls.SAMPLE_QUESTIONS['history']))
                
                # Randomly select number of clues
                num_clues = random.randint(min_clues, max_clues)
                
                # Sample clues (repeat if needed)
                if len(template['clues']) >= num_clues:
                    clues = template['clues'][:num_clues]
                else:
                    clues = template['clues'] + random.choices(
                        template['clues'], k=num_clues - len(template['clues']))
                
                # Shuffle answer choices (correct answer at random position)
                answer_choices = [template['entity']] + template['distractors'][:3]
                correct_idx = random.randint(0, 3)
                
                # Swap correct answer to correct_idx position
                answer_choices[0], answer_choices[correct_idx] = \
                    answer_choices[correct_idx], answer_choices[0]
                
                question = Question(
                    question_id=f"{category}_{question_id:04d}",
                    clues=clues,
                    answer_choices=answer_choices,
                    correct_answer_idx=correct_idx,
                    category=category,
                    metadata={
                        'source': 'synthetic',
                        'template_entity': template['entity']
                    }
                )
                
                questions.append(question)
                question_id += 1
        
        return QuizBowlDataset(questions)


def create_train_val_test_splits(
    dataset: QuizBowlDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[QuizBowlDataset, QuizBowlDataset, QuizBowlDataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    random.seed(seed)
    
    questions = dataset.questions.copy()
    random.shuffle(questions)
    
    total = len(questions)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_questions = questions[:train_end]
    val_questions = questions[train_end:val_end]
    test_questions = questions[val_end:]
    
    return (
        QuizBowlDataset(train_questions),
        QuizBowlDataset(val_questions),
        QuizBowlDataset(test_questions)
    )


def setup_datasets(config: Config) -> Tuple[QuizBowlDataset, QuizBowlDataset, QuizBowlDataset]:
    """
    Setup datasets with proper splits.
    Tries to load from questions.csv first, falls back to synthetic data.
    
    Args:
        config: Configuration object
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    # Check for questions.csv in project root or data directory
    csv_path = None
    for potential_path in [
        Path("questions.csv"),
        Path(__file__).parent / "questions.csv",
        data_dir / "questions.csv"
    ]:
        if potential_path.exists():
            csv_path = potential_path
            print(f"Found questions.csv at {csv_path}")
            break
    
    dataset_path = data_dir / "processed_dataset.json"
    
    # Try to load existing processed dataset
    if dataset_path.exists():
        print(f"\nLoading existing processed dataset from {dataset_path}")
        full_dataset = QuizBowlDataset.load(str(dataset_path))
    
    # Load from CSV if available
    elif csv_path is not None:
        print(f"\nLoading from QANTA CSV file: {csv_path}")
        full_dataset = QANTADatasetLoader.load_from_csv(
            csv_path=str(csv_path),
            num_questions=config.NUM_QUESTIONS,
            num_choices=config.NUM_ANSWER_CHOICES,
            min_clues=config.MIN_CLUES_PER_QUESTION,
            max_clues=config.MAX_CLUES_PER_QUESTION,
            seed=config.SEED
        )
        # Save processed dataset
        full_dataset.save(str(dataset_path))
        print(f"Saved processed dataset to {dataset_path}")
    
    # Fall back to synthetic data
    else:
        print(f"\nNo questions.csv found, generating synthetic dataset with {config.NUM_QUESTIONS} questions")
        full_dataset = SyntheticDatasetGenerator.generate_dataset(
            num_questions=config.NUM_QUESTIONS,
            category_distribution=config.CATEGORY_DISTRIBUTION,
            min_clues=config.MIN_CLUES_PER_QUESTION,
            max_clues=config.MAX_CLUES_PER_QUESTION,
            seed=config.SEED
        )
        full_dataset.save(str(dataset_path))
        print(f"Saved synthetic dataset to {dataset_path}")
    
    # Check if splits already exist
    train_path = data_dir / "train_dataset.json"
    val_path = data_dir / "val_dataset.json"
    test_path = data_dir / "test_dataset.json"
    
    if train_path.exists() and val_path.exists() and test_path.exists():
        print(f"\nLoading existing splits from {data_dir}")
        train_dataset = QuizBowlDataset.load(str(train_path))
        val_dataset = QuizBowlDataset.load(str(val_path))
        test_dataset = QuizBowlDataset.load(str(test_path))
    else:
        print(f"\nCreating new train/val/test splits...")
        # Create splits
        train_dataset, val_dataset, test_dataset = create_train_val_test_splits(
            full_dataset,
            train_ratio=config.TRAIN_SPLIT,
            val_ratio=config.VAL_SPLIT,
            test_ratio=config.TEST_SPLIT,
            seed=config.SEED
        )
        
        # Save splits separately
        train_dataset.save(str(train_path))
        val_dataset.save(str(val_path))
        test_dataset.save(str(test_path))
        print(f"Saved splits to {data_dir}")
    
    print(f"\nDataset splits: Train={len(train_dataset)}, "
          f"Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Print category distribution
    train_categories = [q.category for q in train_dataset.questions]
    category_counts = {}
    for cat in train_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution in training set:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({100*count/len(train_dataset):.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

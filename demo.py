"""
Interactive demo for testing the trained model
"""

import torch
import argparse
from pathlib import Path

from model import T5PolicyModel
from environment import QuizBowlEnvironment, Question
from config import Config


class InteractiveDemo:
    """Interactive demo for question answering"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize demo with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on
        """
        print(f"Loading model from {model_path}...")
        self.model = T5PolicyModel.load_pretrained(model_path, device=device)
        self.model.to(device)
        self.model.eval()
        self.device = device
        print("Model loaded successfully!")
    
    def run_episode(self, question: Question, verbose: bool = True):
        """
        Run a single episode with the given question.
        
        Args:
            question: Question object
            verbose: Whether to print step-by-step details
            
        Returns:
            Dictionary with episode results
        """
        env = QuizBowlEnvironment(question)
        obs = env.reset()
        done = False
        step_count = 0
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"Question ID: {question.question_id}")
            print(f"Category: {question.category}")
            print(f"Total Clues: {len(question.clues)}")
            print("=" * 70)
        
        with torch.no_grad():
            while not done:
                step_count += 1
                
                # Get current observation
                text = env.get_text_representation(obs)
                
                if verbose:
                    print(f"\n--- Step {step_count} (Clue {obs['clue_position'] + 1}/{obs['total_clues']}) ---")
                    print(f"Current clue: {obs['clues'][-1]}")
                
                # Tokenize
                inputs = self.model.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get model prediction
                outputs = self.model.forward(inputs['input_ids'], inputs['attention_mask'])
                
                # Get action probabilities
                action_probs = outputs['action_probs'][0].cpu().numpy()
                wait_prob = outputs['wait_prob'][0].item()
                answer_logits = outputs['answer_logits'][0].cpu()
                answer_probs = torch.softmax(answer_logits, dim=-1).numpy()
                
                if verbose:
                    print(f"\nModel predictions:")
                    print(f"  Wait probability: {wait_prob:.3f}")
                    print(f"  Answer probabilities:")
                    for i, (choice, prob) in enumerate(zip(obs['answer_choices'], answer_probs)):
                        marker = "✓" if i == question.correct_answer_idx else " "
                        print(f"    ({i+1}) {choice}: {prob:.3f} {marker}")
                
                # Select action (deterministic - argmax)
                action = action_probs.argmax()
                
                if action == 0:
                    if verbose:
                        print(f"\nAction: WAIT (continue to next clue)")
                else:
                    selected_idx = action - 1
                    if verbose:
                        print(f"\nAction: SELECT answer ({selected_idx + 1}) {obs['answer_choices'][selected_idx]}")
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                obs = next_obs
        
        # Episode complete
        if verbose:
            print("\n" + "=" * 70)
            print("EPISODE COMPLETE")
            print("=" * 70)
            
            if 'is_correct' in info:
                result = "CORRECT ✓" if info['is_correct'] else "INCORRECT ✗"
                print(f"Result: {result}")
                print(f"Selected: ({info['answer_idx'] + 1}) {question.answer_choices[info['answer_idx']]}")
                print(f"Correct: ({info['correct_idx'] + 1}) {question.answer_choices[info['correct_idx']]}")
                print(f"Buzz Position: {info['clue_position'] + 1}/{len(question.clues)}")
                print(f"Reward: {reward:.3f}")
            print("=" * 70)
        
        return {
            'is_correct': info.get('is_correct', False),
            'reward': reward,
            'buzz_position': info.get('clue_position', 0),
            'selected_answer': info.get('answer_idx', -1),
            'correct_answer': info.get('correct_idx', -1)
        }
    
    def interactive_mode(self):
        """Run interactive mode where user can input questions"""
        print("\n" + "=" * 70)
        print("INTERACTIVE QUESTION ANSWERING DEMO")
        print("=" * 70)
        print("\nEnter 'quit' to exit")
        
        while True:
            print("\n" + "-" * 70)
            
            # Get question from user
            question_id = input("\nQuestion ID (or 'quit'): ").strip()
            if question_id.lower() == 'quit':
                break
            
            category = input("Category: ").strip()
            
            # Get clues
            clues = []
            print("\nEnter clues (press Enter twice when done):")
            while True:
                clue = input(f"Clue {len(clues) + 1}: ").strip()
                if not clue:
                    break
                clues.append(clue)
            
            if not clues:
                print("No clues entered. Skipping question.")
                continue
            
            # Get answer choices
            choices = []
            print("\nEnter 4 answer choices:")
            for i in range(4):
                choice = input(f"Choice {i+1}: ").strip()
                choices.append(choice)
            
            correct_idx = int(input("\nCorrect answer index (1-4): ")) - 1
            
            # Create question
            question = Question(
                question_id=question_id,
                clues=clues,
                answer_choices=choices,
                correct_answer_idx=correct_idx,
                category=category
            )
            
            # Run episode
            self.run_episode(question, verbose=True)


def demo_with_sample_questions(model_path: str, device: str = 'cpu'):
    """Run demo with pre-defined sample questions"""
    
    demo = InteractiveDemo(model_path, device)
    
    # Sample questions
    sample_questions = [
        Question(
            question_id="demo_history_001",
            clues=[
                "This military leader established the Continental System to economically isolate Britain.",
                "He crowned himself emperor in 1804 at Notre-Dame Cathedral in Paris.",
                "His Russian campaign of 1812 ended in catastrophic retreat from Moscow.",
                "He was finally defeated at Waterloo in 1815 by Wellington and Blücher.",
                "This French emperor was exiled to Elba and later to Saint Helena.",
            ],
            answer_choices=["Napoleon Bonaparte", "Julius Caesar", "Alexander the Great", "Charlemagne"],
            correct_answer_idx=0,
            category="history"
        ),
        Question(
            question_id="demo_science_001",
            clues=[
                "These organelles have their own circular DNA separate from nuclear DNA.",
                "They are believed to have originated from endosymbiotic bacteria.",
                "The inner membrane is folded into structures called cristae.",
                "They produce ATP through oxidative phosphorylation.",
                "These are often called the 'powerhouse of the cell'.",
            ],
            answer_choices=["Mitochondria", "Chloroplast", "Ribosome", "Endoplasmic Reticulum"],
            correct_answer_idx=0,
            category="science"
        )
    ]
    
    print("\n" + "=" * 70)
    print("DEMO WITH SAMPLE QUESTIONS")
    print("=" * 70)
    
    results = []
    for question in sample_questions:
        result = demo.run_episode(question, verbose=True)
        results.append(result)
        input("\nPress Enter to continue to next question...")
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    correct = sum(1 for r in results if r['is_correct'])
    print(f"Questions: {len(results)}")
    print(f"Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    avg_reward = sum(r['reward'] for r in results) / len(results)
    print(f"Average Reward: {avg_reward:.3f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Interactive demo for QA model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use')
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['sample', 'interactive'],
                       help='Demo mode: sample questions or interactive')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        demo_with_sample_questions(args.model_path, args.device)
    else:
        demo = InteractiveDemo(args.model_path, args.device)
        demo.interactive_mode()


if __name__ == "__main__":
    main()

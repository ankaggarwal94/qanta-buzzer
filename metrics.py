"""
Evaluation metrics for question answering
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from sklearn.metrics import accuracy_score
from collections import defaultdict


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types to JSON-serializable Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


class MetricsTracker:
    """Track and compute various metrics for QA evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked values"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.rewards = []
        self.buzz_positions = []
        self.is_correct = []
        self.categories = []
        
    def update(self, 
               pred: int,
               target: int,
               confidence: float,
               reward: float = None,
               buzz_position: int = None,
               category: str = None):
        """Update metrics with new sample"""
        self.predictions.append(pred)
        self.targets.append(target)
        self.confidences.append(confidence)
        
        is_correct = (pred == target)
        self.is_correct.append(is_correct)
        
        if reward is not None:
            self.rewards.append(reward)
        if buzz_position is not None:
            self.buzz_positions.append(buzz_position)
        if category is not None:
            self.categories.append(category)
    
    def compute_accuracy(self) -> float:
        """Compute overall accuracy"""
        if len(self.predictions) == 0:
            return 0.0
        return accuracy_score(self.targets, self.predictions)
    
    def compute_average_reward(self) -> float:
        """Compute average reward"""
        if len(self.rewards) == 0:
            return 0.0
        return np.mean(self.rewards)
    
    def compute_average_buzz_position(self) -> float:
        """Compute average buzz position"""
        if len(self.buzz_positions) == 0:
            return 0.0
        return np.mean(self.buzz_positions)
    
    def compute_ece(self, num_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            num_bins: Number of bins for calibration
            
        Returns:
            ECE score
        """
        if len(self.confidences) == 0:
            return 0.0
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Average confidence in bin
                avg_confidence = np.mean(confidences[in_bin])
                # Average accuracy in bin
                avg_accuracy = np.mean(is_correct[in_bin])
                # Bin weight
                bin_weight = np.sum(in_bin) / len(confidences)
                
                # Add to ECE
                ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
        
        return ece
    
    def compute_brier_score(self) -> float:
        """
        Compute Brier score (mean squared error between confidence and correctness).
        
        Returns:
            Brier score
        """
        if len(self.confidences) == 0:
            return 0.0
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        return np.mean((confidences - is_correct) ** 2)
    
    def compute_category_accuracy(self) -> Dict[str, float]:
        """Compute accuracy per category"""
        if len(self.categories) == 0:
            return {}
        
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred, target, cat in zip(self.predictions, self.targets, self.categories):
            category_stats[cat]['total'] += 1
            if pred == target:
                category_stats[cat]['correct'] += 1
        
        return {cat: stats['correct'] / stats['total'] 
                for cat, stats in category_stats.items()}
    
    def compute_reliability_diagram_data(self, num_bins: int = 10) -> Dict:
        """
        Compute data for reliability diagram.
        
        Returns:
            Dictionary with bin information
        """
        if len(self.confidences) == 0:
            return {}
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_data = {
            'bin_centers': [],
            'accuracies': [],
            'confidences': [],
            'counts': []
        }
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_center = (bin_lower + bin_upper) / 2
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(is_correct[in_bin])
                count = np.sum(in_bin)
                
                bin_data['bin_centers'].append(bin_center)
                bin_data['accuracies'].append(avg_accuracy)
                bin_data['confidences'].append(avg_confidence)
                bin_data['counts'].append(count)
        
        return bin_data
    
    def compute_buzz_position_stats(self) -> Dict:
        """Compute statistics about buzzing positions"""
        if len(self.buzz_positions) == 0:
            return {}
        
        positions = np.array(self.buzz_positions)
        
        # Accuracy by position
        position_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pos, correct in zip(self.buzz_positions, self.is_correct):
            position_accuracy[pos]['total'] += 1
            if correct:
                position_accuracy[pos]['correct'] += 1
        
        return {
            'mean': np.mean(positions),
            'std': np.std(positions),
            'min': np.min(positions),
            'max': np.max(positions),
            'position_accuracy': {
                pos: stats['correct'] / stats['total']
                for pos, stats in position_accuracy.items()
            }
        }
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        summary = {
            'num_samples': len(self.predictions),
            'accuracy': self.compute_accuracy(),
        }
        
        if len(self.rewards) > 0:
            summary['average_reward'] = self.compute_average_reward()
        
        if len(self.buzz_positions) > 0:
            summary['average_buzz_position'] = self.compute_average_buzz_position()
            summary['buzz_stats'] = self.compute_buzz_position_stats()
        
        if len(self.confidences) > 0:
            summary['ece'] = self.compute_ece()
            summary['brier_score'] = self.compute_brier_score()
        
        if len(self.categories) > 0:
            summary['category_accuracy'] = self.compute_category_accuracy()
        
        # Convert all numpy types to JSON-serializable Python types
        return convert_to_json_serializable(summary)
    
    def print_summary(self):
        """Print summary of metrics"""
        summary = self.get_summary()
        
        print("=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Samples: {summary['num_samples']}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        
        if 'average_reward' in summary:
            print(f"Average Reward: {summary['average_reward']:.4f}")
        
        if 'average_buzz_position' in summary:
            print(f"Average Buzz Position: {summary['average_buzz_position']:.2f}")
            buzz_stats = summary['buzz_stats']
            print(f"  Min: {buzz_stats['min']}, Max: {buzz_stats['max']}, "
                  f"Std: {buzz_stats['std']:.2f}")
        
        if 'ece' in summary:
            print(f"ECE: {summary['ece']:.4f}")
            print(f"Brier Score: {summary['brier_score']:.4f}")
        
        if 'category_accuracy' in summary:
            print("\nCategory Accuracy:")
            for cat, acc in sorted(summary['category_accuracy'].items()):
                print(f"  {cat}: {acc:.4f}")
        
        print("=" * 60)


def compute_system_score(predictions: List[int],
                         targets: List[int],
                         buzz_positions: List[int],
                         total_clues: List[int]) -> float:
    """
    Compute QANTA system score S_q.
    
    S_q = (correct answers) / (total questions) * (1 - avg_position_ratio)
    where position_ratio = buzz_position / total_clues
    
    Args:
        predictions: Predicted answer indices
        targets: True answer indices
        buzz_positions: Position where model buzzed (0-indexed)
        total_clues: Total number of clues in each question
        
    Returns:
        System score
    """
    correct = np.array([p == t for p, t in zip(predictions, targets)], dtype=float)
    position_ratios = np.array([pos / total for pos, total in zip(buzz_positions, total_clues)])
    
    accuracy = np.mean(correct)
    avg_position_ratio = np.mean(position_ratios)
    
    system_score = accuracy * (1 - avg_position_ratio)
    
    return system_score


def evaluate_model(model, 
                  dataset,
                  device: str = 'cpu',
                  max_samples: int = None,
                  deterministic: bool = True) -> MetricsTracker:
    """
    Evaluate model on a dataset using the RL environment.
    
    Args:
        model: T5PolicyModel
        dataset: QuizBowlDataset
        device: Device to run on
        max_samples: Maximum samples to evaluate (None = all)
        deterministic: Use deterministic action selection
        
    Returns:
        MetricsTracker with results
    """
    from environment import QuizBowlEnvironment
    
    model.eval()
    metrics = MetricsTracker()
    
    questions = dataset.questions[:max_samples] if max_samples else dataset.questions
    
    with torch.no_grad():
        for question in questions:
            env = QuizBowlEnvironment(question)
            obs = env.reset()
            done = False
            
            while not done:
                # Get text representation
                text = env.get_text_representation(obs)
                
                # Tokenize
                inputs = model.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # Get action
                actions, info = model.select_action(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    deterministic=deterministic
                )
                
                action = actions.item()
                
                # Step environment
                obs, reward, done, step_info = env.step(action)
            
            # Get final metrics
            if 'is_correct' in step_info:
                # Extract confidence (max probability over answer choices)
                answer_probs = torch.softmax(info['answer_logits'], dim=-1)
                confidence = answer_probs.max().item()
                
                metrics.update(
                    pred=step_info['answer_idx'],
                    target=step_info['correct_idx'],
                    confidence=confidence,
                    reward=reward,
                    buzz_position=step_info['clue_position'],
                    category=question.category
                )
    
    return metrics


def evaluate_choices_only(model,
                          dataset,
                          device: str = 'cpu',
                          max_samples: int = None) -> MetricsTracker:
    """
    Evaluate model on answer choices only (control experiment).
    
    Args:
        model: T5PolicyModel
        dataset: QuizBowlDataset
        device: Device to run on
        max_samples: Maximum samples to evaluate
        
    Returns:
        MetricsTracker with results
    """
    from environment import QuizBowlEnvironment
    
    model.eval()
    metrics = MetricsTracker()
    
    questions = dataset.questions[:max_samples] if max_samples else dataset.questions
    
    with torch.no_grad():
        for question in questions:
            env = QuizBowlEnvironment(question)
            
            # Get choices-only text
            text = env.get_choices_only_text()
            
            # Tokenize
            inputs = model.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Get prediction
            answer_logits, predictions = model.predict_answer(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            pred = predictions.item()
            answer_probs = torch.softmax(answer_logits, dim=-1)
            confidence = answer_probs.max().item()
            
            metrics.update(
                pred=pred,
                target=question.correct_answer_idx,
                confidence=confidence,
                category=question.category
            )
    
    return metrics

"""
POMDP Environment for Quiz Bowl Question Answering
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Question:
    """Represents a quiz bowl question with pyramidal clues"""
    question_id: str
    clues: List[str]  # Ordered from difficult to easy
    answer_choices: List[str]  # 4 choices: [correct, distractor1, distractor2, distractor3]
    correct_answer_idx: int  # Index of correct answer (0-3)
    category: str
    metadata: Optional[Dict] = None


class QuizBowlEnvironment:
    """
    POMDP Environment for incremental question answering.
    
    States: Complete questions with all clues
    Observations: Partial questions (clues revealed so far) + answer choices
    Actions: WAIT (0) or SELECT answer i (1-4)
    Rewards: Shaped reward based on correctness and timing
    """
    
    WAIT_ACTION = 0
    
    def __init__(self, question: Question, reward_time_penalty: float = 0.1):
        """
        Initialize environment with a question.
        
        Args:
            question: Question object containing clues and answers
            reward_time_penalty: Penalty coefficient for late answering
        """
        self.question = question
        self.reward_time_penalty = reward_time_penalty
        
        self.num_clues = len(question.clues)
        self.num_actions = 1 + len(question.answer_choices)  # WAIT + SELECT answer
        
        # Episode state
        self.current_clue_idx = 0
        self.done = False
        self.selected_answer = None
        
    def reset(self) -> Dict:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_clue_idx = 0
        self.done = False
        self.selected_answer = None
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: 0 for WAIT, 1-4 for SELECT answer choice
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            raise ValueError("Episode is already done. Call reset().")
        
        info = {
            'clue_position': self.current_clue_idx,
            'total_clues': self.num_clues
        }
        
        # Action is WAIT
        if action == self.WAIT_ACTION:
            self.current_clue_idx += 1
            
            # Check if we've run out of clues
            if self.current_clue_idx >= self.num_clues:
                # Forced to answer at last clue
                self.done = True
                info['forced_answer'] = True
                return self._get_observation(), 0.0, True, info
            
            # Continue episode
            return self._get_observation(), 0.0, False, info
        
        # Action is SELECT answer (1-4 maps to 0-3)
        else:
            answer_idx = action - 1
            
            if answer_idx < 0 or answer_idx >= len(self.question.answer_choices):
                raise ValueError(f"Invalid action: {action}. Must be 0-{self.num_actions-1}")
            
            self.selected_answer = answer_idx
            self.done = True
            
            # Compute reward
            is_correct = (answer_idx == self.question.correct_answer_idx)
            time_penalty = self.reward_time_penalty * (self.current_clue_idx / self.num_clues)
            
            if is_correct:
                reward = 1.0 - time_penalty
            else:
                reward = -time_penalty
            
            info['is_correct'] = is_correct
            info['answer_idx'] = answer_idx
            info['correct_idx'] = self.question.correct_answer_idx
            
            return self._get_observation(), reward, True, info
    
    def _get_observation(self) -> Dict:
        """
        Get current observation (partial question + answer choices).
        
        Returns:
            Dictionary containing visible clues and answer choices
        """
        visible_clues = self.question.clues[:self.current_clue_idx + 1]
        
        return {
            'clues': visible_clues,
            'answer_choices': self.question.answer_choices,
            'clue_position': self.current_clue_idx,
            'total_clues': self.num_clues,
            'category': self.question.category
        }
    
    def get_text_representation(self, observation: Optional[Dict] = None) -> str:
        """
        Convert observation to text string for model input.
        
        Args:
            observation: If None, use current observation
            
        Returns:
            Formatted text string
        """
        if observation is None:
            observation = self._get_observation()
        
        clues_text = " ".join(observation['clues'])
        choices_text = " | ".join([f"({i+1}) {choice}" 
                                   for i, choice in enumerate(observation['answer_choices'])])
        
        return f"CLUES: {clues_text} | CHOICES: {choices_text}"
    
    def get_choices_only_text(self) -> str:
        """Get text with only answer choices (for control experiment)"""
        choices_text = " | ".join([f"({i+1}) {choice}" 
                                   for i, choice in enumerate(self.question.answer_choices)])
        return f"CHOICES: {choices_text}"
    
    def render(self) -> str:
        """Render current state as string"""
        obs = self._get_observation()
        
        output = [
            f"Question ID: {self.question.question_id}",
            f"Category: {self.question.category}",
            f"Clue Position: {self.current_clue_idx + 1}/{self.num_clues}",
            "",
            "Visible Clues:"
        ]
        
        for i, clue in enumerate(obs['clues']):
            output.append(f"  {i+1}. {clue}")
        
        output.append("")
        output.append("Answer Choices:")
        for i, choice in enumerate(obs['answer_choices']):
            marker = " ✓" if i == self.question.correct_answer_idx else ""
            output.append(f"  ({i+1}) {choice}{marker}")
        
        if self.done and self.selected_answer is not None:
            output.append("")
            is_correct = self.selected_answer == self.question.correct_answer_idx
            output.append(f"Selected: ({self.selected_answer + 1}) {obs['answer_choices'][self.selected_answer]}")
            output.append(f"Result: {'CORRECT ✓' if is_correct else 'INCORRECT ✗'}")
        
        return "\n".join(output)


class BatchedEnvironment:
    """Manages multiple environments in parallel for efficient training"""
    
    def __init__(self, questions: List[Question], reward_time_penalty: float = 0.1):
        """
        Initialize batched environments.
        
        Args:
            questions: List of Question objects
            reward_time_penalty: Penalty coefficient for late answering
        """
        self.envs = [QuizBowlEnvironment(q, reward_time_penalty) for q in questions]
        self.num_envs = len(self.envs)
    
    def reset(self) -> List[Dict]:
        """Reset all environments"""
        return [env.reset() for env in self.envs]
    
    def step(self, actions: List[int]) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """
        Take actions in all environments.
        
        Args:
            actions: List of actions, one per environment
            
        Returns:
            observations, rewards, dones, infos
        """
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        observations = [r[0] for r in results]
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        
        return observations, rewards, dones, infos
    
    def get_text_representations(self, observations: Optional[List[Dict]] = None) -> List[str]:
        """Get text representations for all environments"""
        if observations is None:
            observations = [None] * self.num_envs
        return [env.get_text_representation(obs) 
                for env, obs in zip(self.envs, observations)]

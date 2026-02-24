"""Quick import test"""

from config import Config
from environment import Question, QuizBowlEnvironment
from dataset import QuizBowlDataset, SyntheticDatasetGenerator
from model import T5PolicyModel, PolicyHead

print('✓ All core modules imported successfully!')
print('✓ Config:', Config.MODEL_NAME)
print('✓ Question class available')
print('✓ QuizBowlEnvironment class available')
print('✓ QuizBowlDataset class available')
print('✓ T5PolicyModel class available')

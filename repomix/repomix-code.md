This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.
The content has been processed where line numbers have been added.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: agents/**, evaluation/**, models/**, qb_data/**, qb_env/**, scripts/**, training/**, tests/**, configs/**, pyproject.toml, setup.cfg, README.md, AGENTS.md, CLAUDE.md
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Line numbers have been added to the beginning of each line
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
agents/
  __init__.py
  _math.py
  bayesian_buzzer.py
  ppo_buzzer.py
  softmax_profile_buzzer.py
  threshold_buzzer.py
configs/
  default.yaml
  smoke.yaml
  t5_policy.yaml
evaluation/
  __init__.py
  controls.py
  metrics.py
  plotting.py
models/
  __init__.py
  answer_profiles.py
  dspy_likelihood.py
  features.py
  likelihoods.py
  t5_policy.py
qb_data/
  __init__.py
  answer_profiles.py
  config.py
  data_loader.py
  dataset_splits.py
  dspy_answer_profiles.py
  huggingface_loader.py
  mc_builder.py
  text_utils.py
qb_env/
  __init__.py
  data_loader.py
  mc_builder.py
  opponent_models.py
  stop_only_env.py
  text_utils.py
  text_wrapper.py
  tossup_env.py
scripts/
  _common.py
  build_mc_dataset.py
  ci.sh
  compare_policies.py
  evaluate_all.py
  manual-smoke.sh
  optimize_dspy.py
  run_baselines.py
  run_full_pipeline.sh
  run_smoke_pipeline.py
  sweep_reward_shaping.py
  test_mc_builder.py
  train_ppo.py
  train_t5_policy.py
tests/
  conftest.py
  test_action_space_alignment.py
  test_agents.py
  test_answer_profile_cache.py
  test_build_mc_dataset.py
  test_dataset_splits.py
  test_dspy_answer_profiles.py
  test_dspy_likelihood.py
  test_dspy_optimize.py
  test_environment.py
  test_factories.py
  test_features.py
  test_hazard_pretrain.py
  test_likelihoods.py
  test_mc_builder_topk.py
  test_mc_builder_variable_k.py
  test_metrics.py
  test_opponent_models.py
  test_ppo_buzzer.py
  test_ppo_t5.py
  test_qb_rl_bridge.py
  test_supervised_t5.py
  test_t5_policy.py
  test_text_wrapper.py
  test_variable_k_integration.py
training/
  __init__.py
  hazard_pretrain.py
  train_ppo_t5.py
  train_supervised_t5.py
AGENTS.md
CLAUDE.md
pyproject.toml
README.md
```

# Files

## File: agents/_math.py
````python
 1: from __future__ import annotations
 2: 
 3: import math
 4: 
 5: 
 6: def sigmoid(x: float) -> float:
 7:     """Numerically stable logistic sigmoid for scalar confidence proxies."""
 8:     if x >= 0.0:
 9:         z = math.exp(-x)
10:         return 1.0 / (1.0 + z)
11: 
12:     z = math.exp(x)
13:     return z / (1.0 + z)
````

## File: agents/softmax_profile_buzzer.py
````python
 1: """qb-rl compatibility re-exports for Bayesian-family buzzers."""
 2: 
 3: from agents.bayesian_buzzer import (
 4:     SequentialBayesBuzzer,
 5:     SoftmaxEpisodeResult,
 6:     SoftmaxProfileBuzzer,
 7: )
 8: 
 9: __all__ = [
10:     "SoftmaxEpisodeResult",
11:     "SoftmaxProfileBuzzer",
12:     "SequentialBayesBuzzer",
13: ]
````

## File: configs/t5_policy.yaml
````yaml
 1: # T5 Policy Configuration
 2: # Hyperparameters for T5PolicyModel with supervised warm-start and PPO fine-tuning.
 3: # Use with: python -m training.train_supervised_t5 --config configs/t5_policy.yaml
 4: 
 5: model:
 6:   model_name: t5-large  # Use t5-base or t5-small if memory constrained
 7:   device: auto  # auto-detect cuda > mps > cpu
 8:   max_input_length: 512
 9:   num_choices: 4
10: 
11: supervised:
12:   lr: 3.0e-4
13:   epochs: 10
14:   batch_size: 8
15:   grad_accum_steps: 4  # Effective batch = 32
16:   max_grad_norm: 1.0
17:   weight_decay: 0.01
18:   checkpoint_dir: checkpoints
19: 
20: ppo:
21:   lr: 1.0e-5  # Lower than supervised for stability
22:   iterations: 100
23:   batch_size: 8
24:   epochs_per_iter: 4
25:   clip_ratio: 0.2
26:   value_coef: 0.5
27:   entropy_coef: 0.01
28:   max_grad_norm: 0.5
29:   gamma: 0.99
30:   gae_lambda: 0.95
31:   target_kl: 0.03
32:   checkpoint_dir: checkpoints
33: 
34: data:
35:   csv_path: "questions.csv"
36:   K: 4
37:   train_size: 0.7
38:   val_size: 0.15
39:   test_size: 0.15
40:   seed: 42
41: 
42: # Smoke test overrides (use with --smoke flag)
43: smoke:
44:   model:
45:     model_name: t5-small  # 60M params instead of 770M
46:     max_input_length: 128
47:   supervised:
48:     epochs: 2
49:     batch_size: 4
50:     grad_accum_steps: 1  # No accumulation for speed
51:   ppo:
52:     iterations: 5
53:     batch_size: 4
54:     epochs_per_iter: 2
55:   data:
56:     max_questions: 50
````

## File: evaluation/plotting.py
````python
  1: """
  2: Visualization Functions for Quiz Bowl Buzzer Evaluation
  3: 
  4: Provides plotting utilities for evaluation results including entropy curves,
  5: calibration plots, and comparison tables. All functions accept output paths
  6: and create parent directories as needed.
  7: 
  8: Ported from qb-rl reference implementation (evaluation/plotting.py) with
  9: import path adaptations for the unified qanta-buzzer codebase.
 10: """
 11: 
 12: from __future__ import annotations
 13: 
 14: from pathlib import Path
 15: from typing import Any
 16: 
 17: import matplotlib
 18: matplotlib.use("Agg")  # Non-interactive backend for headless environments
 19: import matplotlib.pyplot as plt
 20: import numpy as np
 21: import pandas as pd
 22: import seaborn as sns
 23: 
 24: 
 25: def _ensure_parent(path: str | Path) -> Path:
 26:     """Create parent directories for an output path if needed.
 27: 
 28:     Parameters
 29:     ----------
 30:     path : str or Path
 31:         Output file path.
 32: 
 33:     Returns
 34:     -------
 35:     Path
 36:         The resolved Path object.
 37:     """
 38:     p = Path(path)
 39:     p.parent.mkdir(parents=True, exist_ok=True)
 40:     return p
 41: 
 42: 
 43: def plot_learning_curve(
 44:     timesteps: list[int],
 45:     rewards: list[float],
 46:     output_path: str | Path,
 47: ) -> str:
 48:     """Plot training learning curve (reward vs timesteps).
 49: 
 50:     Parameters
 51:     ----------
 52:     timesteps : list[int]
 53:         Training timestep values.
 54:     rewards : list[float]
 55:         Corresponding episode reward values.
 56:     output_path : str or Path
 57:         File path for the saved figure.
 58: 
 59:     Returns
 60:     -------
 61:     str
 62:         Path to the saved figure.
 63:     """
 64:     p = _ensure_parent(output_path)
 65:     plt.figure(figsize=(7, 4))
 66:     sns.lineplot(x=timesteps, y=rewards)
 67:     plt.title("Learning Curve")
 68:     plt.xlabel("Timesteps")
 69:     plt.ylabel("Episode Reward")
 70:     plt.tight_layout()
 71:     plt.savefig(p)
 72:     plt.close()
 73:     return str(p)
 74: 
 75: 
 76: def plot_entropy_vs_clue_index(
 77:     entropy_traces: dict[str, list[float]],
 78:     output_path: str | Path,
 79: ) -> str:
 80:     """Plot policy entropy as a function of clue index.
 81: 
 82:     Creates a line plot with multiple agent entropy traces showing how
 83:     policy uncertainty decreases as more clues are revealed.
 84: 
 85:     Parameters
 86:     ----------
 87:     entropy_traces : dict[str, list[float]]
 88:         Mapping from agent name to per-step entropy values.
 89:     output_path : str or Path
 90:         File path for the saved figure.
 91: 
 92:     Returns
 93:     -------
 94:     str
 95:         Path to the saved figure.
 96:     """
 97:     p = _ensure_parent(output_path)
 98:     plt.figure(figsize=(7, 4))
 99:     for label, trace in entropy_traces.items():
100:         x = np.arange(len(trace))
101:         sns.lineplot(x=x, y=trace, label=label)
102:     plt.title("Belief Entropy vs Clue Index")
103:     plt.xlabel("Clue index")
104:     plt.ylabel("Entropy")
105:     plt.tight_layout()
106:     plt.savefig(p)
107:     plt.close()
108:     return str(p)
109: 
110: 
111: def plot_calibration_curve(
112:     confidences: list[float],
113:     outcomes: list[int],
114:     output_path: str | Path,
115:     n_bins: int = 10,
116: ) -> str:
117:     """Plot calibration curve (predicted confidence vs empirical accuracy).
118: 
119:     Bins confidences into uniform bins and plots mean accuracy per bin
120:     against mean confidence. The diagonal represents perfect calibration.
121: 
122:     Parameters
123:     ----------
124:     confidences : list[float]
125:         Predicted confidence values in [0, 1].
126:     outcomes : list[int]
127:         Binary outcomes (1 = correct, 0 = incorrect).
128:     output_path : str or Path
129:         File path for the saved figure.
130:     n_bins : int
131:         Number of uniform bins for confidence bucketing.
132: 
133:     Returns
134:     -------
135:     str
136:         Path to the saved figure.
137:     """
138:     p = _ensure_parent(output_path)
139:     conf = np.array(confidences, dtype=np.float64)
140:     y = np.array(outcomes, dtype=np.float64)
141:     bins = np.linspace(0.0, 1.0, n_bins + 1)
142:     xs = []
143:     ys = []
144:     for i in range(n_bins):
145:         lo, hi = bins[i], bins[i + 1]
146:         mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
147:         if not mask.any():
148:             continue
149:         xs.append(conf[mask].mean())
150:         ys.append(y[mask].mean())
151: 
152:     plt.figure(figsize=(5, 5))
153:     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
154:     plt.scatter(xs, ys, color="tab:blue")
155:     plt.title("Calibration Plot")
156:     plt.xlabel("Predicted confidence")
157:     plt.ylabel("Empirical accuracy")
158:     plt.xlim(0, 1)
159:     plt.ylim(0, 1)
160:     plt.tight_layout()
161:     plt.savefig(p)
162:     plt.close()
163:     return str(p)
164: 
165: 
166: def save_comparison_table(
167:     rows: list[dict[str, Any]],
168:     output_path: str | Path,
169: ) -> str:
170:     """Save agent comparison metrics as a CSV or markdown table.
171: 
172:     Parameters
173:     ----------
174:     rows : list[dict[str, Any]]
175:         List of metric dicts, each with agent name and metrics.
176:     output_path : str or Path
177:         File path for the saved table (.csv or .md).
178: 
179:     Returns
180:     -------
181:     str
182:         Path to the saved table file.
183:     """
184:     p = _ensure_parent(output_path)
185:     df = pd.DataFrame(rows)
186:     if p.suffix.lower() == ".csv":
187:         df.to_csv(p, index=False)
188:     else:
189:         df.to_markdown(p, index=False)
190:     return str(p)
````

## File: models/__init__.py
````python
 1: """
 2: Models Package
 3: 
 4: Likelihood models, belief feature extraction, and policy model interfaces
 5: for the quiz bowl RL buzzer system.
 6: """
 7: 
 8: from models.features import extract_belief_features, entropy_of_distribution
 9: from models.likelihoods import (
10:     LikelihoodModel,
11:     OpenAILikelihood,
12:     SBERTLikelihood,
13:     T5Likelihood,
14:     TfIdfLikelihood,
15:     build_likelihood_from_config,
16: )
17: 
18: # Lazy import: T5PolicyModel and PolicyHead require transformers + torch.
19: # Import on demand to keep package lightweight for belief-feature-only usage.
20: 
21: 
22: def __getattr__(name: str):
23:     if name in ("T5PolicyModel", "PolicyHead"):
24:         from models.t5_policy import T5PolicyModel, PolicyHead
25:         return {"T5PolicyModel": T5PolicyModel, "PolicyHead": PolicyHead}[name]
26:     raise AttributeError(f"module 'models' has no attribute {name!r}")
27: 
28: 
29: __all__ = [
30:     "extract_belief_features",
31:     "entropy_of_distribution",
32:     "LikelihoodModel",
33:     "TfIdfLikelihood",
34:     "SBERTLikelihood",
35:     "OpenAILikelihood",
36:     "T5Likelihood",
37:     "build_likelihood_from_config",
38:     "T5PolicyModel",
39:     "PolicyHead",
40: ]
````

## File: models/answer_profiles.py
````python
1: """qb-rl compatibility re-export for answer profile building."""
2: 
3: from qb_data.answer_profiles import AnswerProfileBuilder
4: 
5: __all__ = ["AnswerProfileBuilder"]
````

## File: qb_data/__init__.py
````python
 1: """Quiz Bowl Data Package.
 2: 
 3: Core data structures and utilities for quiz bowl question processing,
 4: including qb-rl compatibility loader helpers.
 5: """
 6: 
 7: from qb_data.data_loader import (
 8:     QANTADatasetLoader,
 9:     TossupQuestion,
10:     load_tossup_questions,
11:     load_tossup_questions_from_config,
12:     parse_row,
13: )
14: from qb_data.text_utils import normalize_answer
15: 
16: __all__ = [
17:     'TossupQuestion',
18:     'QANTADatasetLoader',
19:     'parse_row',
20:     'load_tossup_questions',
21:     'load_tossup_questions_from_config',
22:     'normalize_answer',
23: ]
````

## File: qb_data/config.py
````python
  1: """Configuration loading and management utilities.
  2: 
  3: Provides functions to load YAML configurations, apply small
  4: cross-codebase compatibility normalizations, and merge CLI overrides
  5: using dot notation (e.g., ``data.K=5`` updates ``config["data"]["K"]``).
  6: """
  7: 
  8: import argparse
  9: from pathlib import Path
 10: from typing import Any, Dict, Optional, Union
 11: 
 12: 
 13: def normalize_config(
 14:     config: Dict[str, Any],
 15:     smoke: bool = False,
 16: ) -> Dict[str, Any]:
 17:     """Apply compatibility defaults to a loaded configuration.
 18: 
 19:     Parameters
 20:     ----------
 21:     config : dict
 22:         Parsed configuration dictionary.
 23:     smoke : bool
 24:         Whether the caller intends to run in smoke mode.
 25: 
 26:     Returns
 27:     -------
 28:     dict
 29:         Normalized configuration dictionary.
 30:     """
 31:     data_cfg = config.setdefault("data", {})
 32:     env_cfg = config.setdefault("environment", {})
 33:     lik_cfg = config.setdefault("likelihood", {})
 34: 
 35:     if "reward" in env_cfg and "reward_mode" not in env_cfg:
 36:         env_cfg["reward_mode"] = env_cfg["reward"]
 37:     elif "reward_mode" in env_cfg and "reward" not in env_cfg:
 38:         env_cfg["reward"] = env_cfg["reward_mode"]
 39: 
 40:     if smoke and data_cfg.get("dataset_smoke") and "dataset" not in data_cfg:
 41:         data_cfg["dataset"] = data_cfg["dataset_smoke"]
 42:     if smoke and data_cfg.get("dataset_smoke_config") and "dataset_config" not in data_cfg:
 43:         data_cfg["dataset_config"] = data_cfg["dataset_smoke_config"]
 44: 
 45:     if "embedding_model" in lik_cfg and "sbert_name" not in lik_cfg:
 46:         lik_cfg["sbert_name"] = lik_cfg["embedding_model"]
 47:     if "sbert_name" in lik_cfg and "embedding_model" not in lik_cfg:
 48:         lik_cfg["embedding_model"] = lik_cfg["sbert_name"]
 49: 
 50:     return config
 51: 
 52: 
 53: def resolve_data_loading_options(
 54:     config: Dict[str, Any],
 55:     smoke: bool = False,
 56: ) -> Dict[str, Any]:
 57:     """Resolve CSV/Hugging Face data-loading options from a config dict.
 58: 
 59:     Parameters
 60:     ----------
 61:     config : dict
 62:         Parsed configuration dictionary.
 63:     smoke : bool
 64:         Whether the caller intends to run in smoke mode.
 65: 
 66:     Returns
 67:     -------
 68:     dict
 69:         Resolved data-loading settings.
 70:     """
 71:     data_cfg = config.get("data", {})
 72:     use_smoke_dataset = smoke and any(
 73:         data_cfg.get(key) is not None
 74:         for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
 75:     )
 76: 
 77:     csv_path = data_cfg.get("csv_path")
 78:     if smoke and data_cfg.get("csv_smoke_path"):
 79:         csv_path = data_cfg["csv_smoke_path"]
 80: 
 81:     dataset = data_cfg.get("dataset")
 82:     dataset_config = data_cfg.get("dataset_config")
 83:     split = data_cfg.get("split", "eval")
 84: 
 85:     if use_smoke_dataset:
 86:         dataset = data_cfg.get("dataset_smoke", dataset)
 87:         dataset_config = data_cfg.get("dataset_smoke_config", dataset_config)
 88:         split = data_cfg.get("split_smoke", split)
 89: 
 90:     return {
 91:         "csv_path": csv_path,
 92:         "dataset": dataset,
 93:         "dataset_config": dataset_config,
 94:         "split": split,
 95:         "use_huggingface": bool(data_cfg.get("use_huggingface", False) or dataset),
 96:         "max_questions": data_cfg.get("max_questions"),
 97:         "uses_dataset_smoke": use_smoke_dataset,
 98:     }
 99: 
100: 
101: def load_config(
102:     config_path: Optional[Union[str, Path]] = None,
103:     smoke: bool = False,
104: ) -> Dict[str, Any]:
105:     """Load configuration from YAML file.
106: 
107:     Parameters
108:     ----------
109:     config_path : str or Path, optional
110:         Path to configuration file. Defaults to configs/default.yaml.
111: 
112:     Returns
113:     -------
114:     dict
115:         Parsed configuration dictionary.
116: 
117:     Raises
118:     ------
119:     FileNotFoundError
120:         If config file doesn't exist.
121:     ImportError
122:         If PyYAML is not installed.
123:     """
124:     try:
125:         import yaml
126:     except ImportError:
127:         raise ImportError(
128:             "PyYAML is required for config loading. "
129:             "Install it with: pip install pyyaml"
130:         )
131: 
132:     # Default to configs/default.yaml if no path given
133:     if config_path is None:
134:         project_root = Path(__file__).parent.parent
135:         default_path = project_root / "configs" / "default.yaml"
136:         smoke_path = project_root / "configs" / "smoke.yaml"
137: 
138:         if smoke and default_path.exists():
139:             with open(default_path, "r", encoding="utf-8") as f:
140:                 default_config = yaml.safe_load(f) or {}
141:             default_data = default_config.get("data", {})
142:             if any(
143:                 default_data.get(key) is not None
144:                 for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
145:             ):
146:                 config_path = default_path
147:             elif smoke_path.exists():
148:                 config_path = smoke_path
149:             else:
150:                 config_path = default_path
151:         else:
152:             config_path = default_path
153:     else:
154:         config_path = Path(config_path)
155: 
156:     if not config_path.exists():
157:         raise FileNotFoundError(f"Configuration file not found: {config_path}")
158: 
159:     with open(config_path, "r", encoding="utf-8") as f:
160:         config = yaml.safe_load(f)
161: 
162:     return normalize_config(config or {}, smoke=smoke)
163: 
164: 
165: def merge_overrides(
166:     config: Dict[str, Any],
167:     overrides: Dict[str, Any]
168: ) -> Dict[str, Any]:
169:     """Merge override values into configuration using dot notation.
170: 
171:     Parameters
172:     ----------
173:     config : dict
174:         Base configuration dictionary.
175:     overrides : dict
176:         Override values to merge. Keys can use dot notation
177:         (e.g., {"data.K": 5} updates config["data"]["K"]).
178: 
179:     Returns
180:     -------
181:     dict
182:         Updated configuration with overrides applied.
183: 
184:     Examples
185:     --------
186:     >>> config = {"data": {"K": 4}, "ppo": {"batch_size": 32}}
187:     >>> overrides = {"data.K": 5, "ppo.batch_size": 16}
188:     >>> config = merge_overrides(config, overrides)
189:     >>> assert config["data"]["K"] == 5
190:     >>> assert config["ppo"]["batch_size"] == 16
191:     """
192:     for key, value in overrides.items():
193:         # Split on dots for nested keys
194:         keys = key.split(".")
195: 
196:         # Navigate to the nested location
197:         current = config
198:         for k in keys[:-1]:
199:             if k not in current:
200:                 current[k] = {}
201:             current = current[k]
202: 
203:         # Set the final value
204:         final_key = keys[-1]
205:         current[final_key] = value
206: 
207:     return normalize_config(config)
208: 
209: 
210: def build_argparse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
211:     """Convert argparse namespace to configuration overrides.
212: 
213:     Parameters
214:     ----------
215:     args : argparse.Namespace
216:         Parsed command-line arguments.
217: 
218:     Returns
219:     -------
220:     dict
221:         Configuration overrides extracted from args.
222: 
223:     Notes
224:     -----
225:     Special handling:
226:     - --smoke flag loads smoke.yaml config
227:     - --config specifies custom config path
228:     - --override key=value pairs for dot notation overrides
229:     """
230:     overrides = {}
231: 
232:     # Handle smoke test mode
233:     if hasattr(args, "smoke") and args.smoke:
234:         overrides["__smoke__"] = True
235: 
236:     # Handle custom config path
237:     if hasattr(args, "config") and args.config:
238:         overrides["__config_path__"] = args.config
239: 
240:     # Parse key=value override pairs
241:     if hasattr(args, "override") and args.override:
242:         for override_str in args.override:
243:             if "=" not in override_str:
244:                 print(f"Warning: Invalid override format '{override_str}', expected 'key=value'")
245:                 continue
246: 
247:             key, value_str = override_str.split("=", 1)
248: 
249:             # Try to parse value as appropriate type
250:             value = parse_value(value_str)
251:             overrides[key] = value
252: 
253:     return overrides
254: 
255: 
256: def parse_value(value_str: str) -> Any:
257:     """Parse string value to appropriate Python type.
258: 
259:     Parameters
260:     ----------
261:     value_str : str
262:         String representation of value.
263: 
264:     Returns
265:     -------
266:     any
267:         Parsed value with appropriate type.
268: 
269:     Examples
270:     --------
271:     >>> parse_value("5") == 5
272:     >>> parse_value("3.14") == 3.14
273:     >>> parse_value("true") == True
274:     >>> parse_value("false") == False
275:     >>> parse_value("null") == None
276:     >>> parse_value("hello") == "hello"
277:     """
278:     # Handle boolean values
279:     if value_str.lower() == "true":
280:         return True
281:     if value_str.lower() == "false":
282:         return False
283: 
284:     # Handle null/none
285:     if value_str.lower() in ("null", "none"):
286:         return None
287: 
288:     # Try to parse as number
289:     try:
290:         # Try integer first
291:         if "." not in value_str:
292:             return int(value_str)
293:         # Then float
294:         return float(value_str)
295:     except ValueError:
296:         pass
297: 
298:     # Return as string
299:     return value_str
300: 
301: 
302: def add_config_args(parser: argparse.ArgumentParser) -> None:
303:     """Add configuration-related arguments to parser.
304: 
305:     Parameters
306:     ----------
307:     parser : argparse.ArgumentParser
308:         Parser to add arguments to.
309:     """
310:     parser.add_argument(
311:         "--config",
312:         type=str,
313:         help="Path to configuration YAML file"
314:     )
315:     parser.add_argument(
316:         "--smoke",
317:         action="store_true",
318:         help="Use smoke test configuration for quick testing"
319:     )
320:     parser.add_argument(
321:         "--override",
322:         action="append",
323:         help="Override config values using dot notation (e.g., data.K=5)"
324:     )
325: 
326: 
327: def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
328:     """Load configuration and apply command-line overrides.
329: 
330:     Parameters
331:     ----------
332:     args : argparse.Namespace
333:         Parsed command-line arguments.
334: 
335:     Returns
336:     -------
337:     dict
338:         Final configuration with all overrides applied.
339:     """
340:     # Build overrides from args
341:     overrides = build_argparse_overrides(args)
342: 
343:     # Check for special config path
344:     config_path = overrides.pop("__config_path__", None)
345:     smoke = bool(overrides.pop("__smoke__", False))
346: 
347:     # Load base config
348:     config = load_config(config_path, smoke=smoke)
349: 
350:     # Apply remaining overrides
351:     if overrides:
352:         config = merge_overrides(config, overrides)
353: 
354:     return config
355: 
356: 
357: # Convenience exports
358: __all__ = [
359:     "load_config",
360:     "merge_overrides",
361:     "normalize_config",
362:     "resolve_data_loading_options",
363:     "build_argparse_overrides",
364:     "add_config_args",
365:     "load_config_with_overrides",
366: ]
````

## File: qb_env/data_loader.py
````python
 1: """qb-rl compatibility re-exports for tossup data loading."""
 2: 
 3: from qb_data.data_loader import (
 4:     QANTADatasetLoader,
 5:     TossupQuestion,
 6:     load_tossup_questions,
 7:     load_tossup_questions_from_config,
 8:     parse_row,
 9: )
10: 
11: __all__ = [
12:     "TossupQuestion",
13:     "QANTADatasetLoader",
14:     "parse_row",
15:     "load_tossup_questions",
16:     "load_tossup_questions_from_config",
17: ]
````

## File: qb_env/mc_builder.py
````python
1: """qb-rl compatibility re-exports for MC question building."""
2: 
3: from qb_data.mc_builder import MCBuilder, MCQuestion, _token_overlap
4: 
5: __all__ = ["MCQuestion", "MCBuilder", "_token_overlap"]
````

## File: qb_env/text_utils.py
````python
1: """qb-rl compatibility re-exports for text utilities."""
2: 
3: from qb_data.text_utils import normalize_answer, tokenize_text
4: 
5: __all__ = ["normalize_answer", "tokenize_text"]
````

## File: qb_env/text_wrapper.py
````python
  1: """
  2: TextObservationWrapper for converting belief features to text observations.
  3: 
  4: Wraps TossupMCEnv to provide text-formatted observations (clues + choices)
  5: instead of numeric belief feature vectors. This bridges the gap between
  6: the environment's native observation space (Box(K+6,)) and T5PolicyModel's
  7: text input requirement.
  8: 
  9: The underlying environment still operates on beliefs internally for reward
 10: computation -- the wrapper only transforms what the agent SEES, not how the
 11: environment computes rewards or transitions.
 12: 
 13: Text format matches T5PolicyModel's expected input:
 14:     "CLUES: clue1 clue2 ... | CHOICES: (1) ans1 (2) ans2 (3) ans3 (4) ans4"
 15: 
 16: Ported from qanta-buzzer's environment.py get_text_representation() method,
 17: adapted for the unified codebase's Gymnasium wrapper pattern.
 18: """
 19: 
 20: from __future__ import annotations
 21: 
 22: from typing import Any, Tuple
 23: 
 24: import gymnasium as gym
 25: import numpy as np
 26: 
 27: from qb_data.mc_builder import MCQuestion
 28: 
 29: 
 30: class TextObservationWrapper(gym.ObservationWrapper):
 31:     """Wrap TossupMCEnv to provide text observations instead of belief features.
 32: 
 33:     The underlying env still operates on beliefs internally (for reward
 34:     computation), but the agent sees text-formatted observations for T5 input.
 35:     This is a Gymnasium ObservationWrapper that intercepts the observation
 36:     returned by reset() and step() and converts it to a text string.
 37: 
 38:     The observation space is set to a placeholder Box(1,) since Gymnasium
 39:     requires a defined space, but text observations are variable-length
 40:     strings. Downstream code (T5PolicyModel) handles tokenization.
 41: 
 42:     Parameters
 43:     ----------
 44:     env : gym.Env
 45:         The underlying TossupMCEnv instance. Must have ``question``
 46:         (MCQuestion) and ``step_idx`` (int) attributes.
 47: 
 48:     Examples
 49:     --------
 50:     >>> from qb_env.tossup_env import TossupMCEnv
 51:     >>> env = TossupMCEnv(questions=qs, likelihood_model=lm, K=4)
 52:     >>> wrapped = TextObservationWrapper(env)
 53:     >>> obs, info = wrapped.reset()
 54:     >>> assert isinstance(obs, str)
 55:     >>> assert "CLUES:" in obs and "CHOICES:" in obs
 56:     """
 57: 
 58:     def __init__(self, env: gym.Env) -> None:
 59:         super().__init__(env)
 60:         # Override observation space with a placeholder.
 61:         # Text observations are variable-length strings; Gymnasium requires
 62:         # a Space object, so we use a minimal Box as a sentinel.
 63:         self.observation_space = gym.spaces.Box(
 64:             low=0, high=1, shape=(1,), dtype=np.float32
 65:         )
 66: 
 67:     def observation(self, obs: np.ndarray) -> str:
 68:         """Convert numeric belief observation to formatted text string.
 69: 
 70:         Reconstructs visible clues from the underlying environment's current
 71:         question and step index, then formats them with answer choices in the
 72:         standard T5PolicyModel input format.
 73: 
 74:         Parameters
 75:         ----------
 76:         obs : np.ndarray
 77:             Numeric belief features from the underlying environment.
 78:             Shape ``(K+6,)``. Not used directly -- the text is reconstructed
 79:             from ``env.question`` and ``env.step_idx``.
 80: 
 81:         Returns
 82:         -------
 83:         str
 84:             Formatted text observation:
 85:             ``"CLUES: <visible clue tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
 86:         """
 87:         question: MCQuestion = self.env.question
 88:         step_idx: int = self.env.step_idx
 89: 
 90:         # Build visible clue text from cumulative prefixes.
 91:         #
 92:         # TossupMCEnv step semantics:
 93:         #   - reset() sets step_idx=0, belief is uniform (no clues processed).
 94:         #   - step(WAIT) calls _compute_belief(step_idx), THEN increments step_idx.
 95:         #   - The observation returned after step() has step_idx ALREADY incremented.
 96:         #
 97:         # So step_idx tells us how many WAIT actions have been taken:
 98:         #   step_idx=0: No WAITs yet; no clues processed; show minimal context
 99:         #   step_idx=N: N WAITs taken; beliefs from cumulative_prefixes[0..N-1]
100:         #
101:         # cumulative_prefixes[i] = text of tokens[0..run_indices[i]].
102:         # After N WAITs, the agent has seen information up to
103:         # cumulative_prefixes[N-1], so that is what the text obs shows.
104:         if step_idx == 0:
105:             # No clues processed yet; show question start as minimal context
106:             # (matches initial observation having some textual content for T5)
107:             clues_text = question.tokens[0] if question.tokens else ""
108:         elif step_idx <= len(question.cumulative_prefixes):
109:             clues_text = question.cumulative_prefixes[step_idx - 1]
110:         else:
111:             # Past all clues (truncated episode); show all text
112:             clues_text = question.cumulative_prefixes[-1]
113: 
114:         # Format answer choices
115:         choices_parts = [
116:             f"({i + 1}) {opt}" for i, opt in enumerate(question.options)
117:         ]
118:         choices_text = " ".join(choices_parts)
119: 
120:         return f"CLUES: {clues_text} | CHOICES: {choices_text}"
121: 
122:     def reset(
123:         self, *, seed: int | None = None, options: dict[str, Any] | None = None
124:     ) -> Tuple[str, dict[str, Any]]:
125:         """Reset the environment and return a text observation.
126: 
127:         Parameters
128:         ----------
129:         seed : int or None
130:             Random seed passed to underlying environment.
131:         options : dict or None
132:             Options passed to underlying environment.
133: 
134:         Returns
135:         -------
136:         observation : str
137:             Text-formatted initial observation.
138:         info : dict[str, Any]
139:             Episode metadata from underlying environment.
140:         """
141:         obs, info = self.env.reset(seed=seed, options=options)
142:         return self.observation(obs), info
143: 
144:     def step(
145:         self, action: int
146:     ) -> Tuple[str, float, bool, bool, dict[str, Any]]:
147:         """Execute one step and return text observation.
148: 
149:         Parameters
150:         ----------
151:         action : int
152:             Action to take. 0 = WAIT, 1..K = BUZZ with answer (action-1).
153: 
154:         Returns
155:         -------
156:         observation : str
157:             Text-formatted observation after the step.
158:         reward : float
159:             Scalar reward for this step.
160:         terminated : bool
161:             True if the agent buzzed (natural episode end).
162:         truncated : bool
163:             True if all clues exhausted (forced termination).
164:         info : dict[str, Any]
165:             Step metadata from underlying environment.
166:         """
167:         obs, reward, terminated, truncated, info = self.env.step(action)
168:         return self.observation(obs), reward, terminated, truncated, info
169: 
170:     @property
171:     def unwrapped_env(self):
172:         """Access the underlying TossupMCEnv directly.
173: 
174:         Returns
175:         -------
176:         TossupMCEnv
177:             The unwrapped environment instance.
178:         """
179:         return self.env
````

## File: scripts/manual-smoke.sh
````bash
 1: #!/usr/bin/env bash
 2: # Manual smoke pipeline -- runs the four-stage belief-feature smoke workflow.
 3: # Intended for human verification, not CI (stages are heavyweight ML runs).
 4: #
 5: # Prereqs: pip install -e .  (see AGENTS.md for full setup)
 6: # Outputs: artifacts/smoke/
 7: set -euo pipefail
 8: 
 9: REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
10: if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
11:     source "$REPO_ROOT/.venv/bin/activate"
12: fi
13: 
14: PYTHON="${PYTHON:-python3}"
15: 
16: echo "=== Stage 1/4: Build MC dataset ==="
17: $PYTHON scripts/build_mc_dataset.py --smoke
18: 
19: echo "=== Stage 2/4: Run baselines ==="
20: $PYTHON scripts/run_baselines.py --smoke
21: 
22: echo "=== Stage 3/4: Train PPO ==="
23: $PYTHON scripts/train_ppo.py --smoke
24: 
25: echo "=== Stage 4/4: Evaluate all ==="
26: $PYTHON scripts/evaluate_all.py --smoke
27: 
28: echo "=== Smoke pipeline complete. Check artifacts/smoke/ ==="
````

## File: scripts/run_smoke_pipeline.py
````python
  1: #!/usr/bin/env python3
  2: """Run the full canonical smoke pipeline end-to-end.
  3: 
  4: Stages:
  5: 1) build_mc_dataset
  6: 2) run_baselines
  7: 3) train_ppo
  8: 4) evaluate_all
  9: 
 10: Writes a summary JSON to artifacts/smoke/smoke_pipeline_summary.json.
 11: """
 12: 
 13: from __future__ import annotations
 14: 
 15: import argparse
 16: import json
 17: import subprocess
 18: import sys
 19: import time
 20: from pathlib import Path
 21: 
 22: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 23: ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "smoke"
 24: 
 25: 
 26: STAGES = [
 27:     ["scripts/build_mc_dataset.py", "--smoke"],
 28:     ["scripts/run_baselines.py", "--smoke"],
 29:     ["scripts/train_ppo.py", "--smoke"],
 30:     ["scripts/evaluate_all.py", "--smoke"],
 31: ]
 32: 
 33: 
 34: def run_stage(python_exe: str, args: list[str]) -> tuple[int, float]:
 35:     """Run one stage command and return (exit_code, seconds)."""
 36:     cmd = [python_exe, *args]
 37:     start = time.time()
 38:     proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
 39:     elapsed = time.time() - start
 40:     return proc.returncode, elapsed
 41: 
 42: 
 43: def main() -> int:
 44:     parser = argparse.ArgumentParser(description="Run full smoke pipeline")
 45:     parser.add_argument(
 46:         "--python",
 47:         default=sys.executable,
 48:         help="Python interpreter to use (default: current interpreter)",
 49:     )
 50:     ns = parser.parse_args()
 51: 
 52:     print("=" * 60)
 53:     print("Smoke Pipeline Runner")
 54:     print("=" * 60)
 55:     print(f"Python: {ns.python}")
 56:     print()
 57: 
 58:     ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
 59: 
 60:     summary: dict[str, object] = {
 61:         "python": ns.python,
 62:         "started_at_unix": time.time(),
 63:         "stages": [],
 64:     }
 65: 
 66:     pipeline_start = time.time()
 67:     for stage_args in STAGES:
 68:         stage_name = stage_args[0]
 69:         print(f"Running: {stage_name} {' '.join(stage_args[1:])}")
 70:         code, seconds = run_stage(ns.python, stage_args)
 71:         summary["stages"].append(
 72:             {
 73:                 "stage": stage_name,
 74:                 "args": stage_args[1:],
 75:                 "exit_code": code,
 76:                 "seconds": round(seconds, 3),
 77:             }
 78:         )
 79:         if code != 0:
 80:             summary["status"] = "failed"
 81:             summary["failed_stage"] = stage_name
 82:             summary["total_seconds"] = round(time.time() - pipeline_start, 3)
 83:             out_path = ARTIFACT_DIR / "smoke_pipeline_summary.json"
 84:             out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
 85:             print(f"\nFAILED at {stage_name} (exit={code})")
 86:             print(f"Summary written: {out_path}")
 87:             return code
 88:         print(f"✓ {stage_name} completed in {seconds:.1f}s\n")
 89: 
 90:     summary["status"] = "ok"
 91:     summary["total_seconds"] = round(time.time() - pipeline_start, 3)
 92:     out_path = ARTIFACT_DIR / "smoke_pipeline_summary.json"
 93:     out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
 94: 
 95:     print("=" * 60)
 96:     print("Smoke pipeline completed successfully")
 97:     print(f"Summary written: {out_path}")
 98:     print("=" * 60)
 99:     return 0
100: 
101: 
102: if __name__ == "__main__":
103:     raise SystemExit(main())
````

## File: scripts/sweep_reward_shaping.py
````python
  1: #!/usr/bin/env python3
  2: """Sweep PPO smoke reward-shaping settings and record results.
  3: 
  4: Runs `scripts/train_ppo.py` in smoke mode across a small grid of:
  5: - environment.wait_penalty
  6: - environment.early_buzz_penalty
  7: 
  8: Collects metrics from artifacts/smoke/ppo_summary.json after each run and writes:
  9: - artifacts/smoke/reward_sweep_results.json
 10: - artifacts/smoke/reward_sweep_results.csv
 11: """
 12: 
 13: from __future__ import annotations
 14: 
 15: import argparse
 16: import csv
 17: import json
 18: import subprocess
 19: import sys
 20: import time
 21: from pathlib import Path
 22: 
 23: import yaml
 24: 
 25: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 26: SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"
 27: ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "smoke"
 28: TMP_CONFIG = ARTIFACT_DIR / "_tmp_sweep_smoke.yaml"
 29: PPO_SUMMARY = ARTIFACT_DIR / "ppo_summary.json"
 30: 
 31: WAIT_PENALTIES = [0.0, 0.02, 0.05]
 32: EARLY_BUZZ_PENALTIES = [0.2, 0.5, 0.8]
 33: SEEDS = [13, 42, 123]
 34: 
 35: 
 36: def run_cmd(cmd: list[str]) -> int:
 37:     proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
 38:     return proc.returncode
 39: 
 40: 
 41: def load_yaml(path: Path) -> dict:
 42:     with path.open("r", encoding="utf-8") as f:
 43:         return yaml.safe_load(f)
 44: 
 45: 
 46: def save_yaml(path: Path, data: dict) -> None:
 47:     path.parent.mkdir(parents=True, exist_ok=True)
 48:     with path.open("w", encoding="utf-8") as f:
 49:         yaml.safe_dump(data, f, sort_keys=False)
 50: 
 51: 
 52: def load_json(path: Path) -> dict:
 53:     with path.open("r", encoding="utf-8") as f:
 54:         return json.load(f)
 55: 
 56: 
 57: def parse_args() -> argparse.Namespace:
 58:     parser = argparse.ArgumentParser(description="Sweep PPO reward shaping")
 59:     parser.add_argument(
 60:         "--seeds",
 61:         type=str,
 62:         default=",".join(str(s) for s in SEEDS),
 63:         help="Comma-separated seeds, e.g. 13,42,123",
 64:     )
 65:     parser.add_argument(
 66:         "--timesteps",
 67:         type=int,
 68:         default=None,
 69:         help="Optional timesteps override for train_ppo during sweep",
 70:     )
 71:     return parser.parse_args()
 72: 
 73: 
 74: def main() -> int:
 75:     args = parse_args()
 76:     ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
 77: 
 78:     base_cfg = load_yaml(SMOKE_CONFIG)
 79: 
 80:     python_exe = sys.executable
 81:     results = []
 82: 
 83:     seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
 84:     grid = [(w, e) for w in WAIT_PENALTIES for e in EARLY_BUZZ_PENALTIES]
 85: 
 86:     print("=" * 72)
 87:     print(f"Reward sweep: {len(grid)} configs x {len(seeds)} seeds")
 88:     print("=" * 72)
 89: 
 90:     for idx, (wait_penalty, early_buzz_penalty) in enumerate(grid, start=1):
 91:         per_seed = []
 92:         print(f"[{idx}/{len(grid)}] wait_penalty={wait_penalty}, early_buzz_penalty={early_buzz_penalty}")
 93: 
 94:         for seed in seeds:
 95:             cfg = dict(base_cfg)
 96:             cfg.setdefault("environment", {})
 97:             cfg["environment"] = dict(cfg["environment"])
 98:             cfg["environment"]["wait_penalty"] = float(wait_penalty)
 99:             cfg["environment"]["early_buzz_penalty"] = float(early_buzz_penalty)
100:             cfg["environment"]["seed"] = int(seed)
101: 
102:             cfg.setdefault("ppo", {})
103:             cfg["ppo"] = dict(cfg["ppo"])
104:             cfg["ppo"]["seed"] = int(seed)
105:             save_yaml(TMP_CONFIG, cfg)
106: 
107:             cmd = [python_exe, "scripts/train_ppo.py", "--config", str(TMP_CONFIG), "--smoke", "--seed", str(seed)]
108:             if args.timesteps is not None:
109:                 cmd.extend(["--timesteps", str(args.timesteps)])
110: 
111:             start = time.time()
112:             code = run_cmd(cmd)
113:             elapsed = time.time() - start
114: 
115:             if code != 0 or not PPO_SUMMARY.exists():
116:                 per_seed.append({"seed": seed, "status": "failed", "seconds": round(elapsed, 3)})
117:                 continue
118: 
119:             summary = load_json(PPO_SUMMARY)
120:             per_seed.append(
121:                 {
122:                     "seed": seed,
123:                     "status": "ok",
124:                     "seconds": round(elapsed, 3),
125:                     "buzz_accuracy": float(summary.get("buzz_accuracy", 0.0)),
126:                     "mean_sq": float(summary.get("mean_sq", 0.0)),
127:                     "mean_buzz_step": float(summary.get("mean_buzz_step", 0.0)),
128:                     "ece": float(summary.get("ece", 0.0)),
129:                     "brier": float(summary.get("brier", 0.0)),
130:                 }
131:             )
132: 
133:         ok = [r for r in per_seed if r.get("status") == "ok"]
134:         if not ok:
135:             results.append(
136:                 {
137:                     "wait_penalty": wait_penalty,
138:                     "early_buzz_penalty": early_buzz_penalty,
139:                     "status": "failed",
140:                     "num_ok": 0,
141:                     "num_total": len(per_seed),
142:                     "per_seed": per_seed,
143:                 }
144:             )
145:             continue
146: 
147:         mean_acc = sum(r["buzz_accuracy"] for r in ok) / len(ok)
148:         mean_sq = sum(r["mean_sq"] for r in ok) / len(ok)
149:         mean_step = sum(r["mean_buzz_step"] for r in ok) / len(ok)
150:         mean_ece = sum(r["ece"] for r in ok) / len(ok)
151:         mean_brier = sum(r["brier"] for r in ok) / len(ok)
152:         mean_seconds = sum(r["seconds"] for r in ok) / len(ok)
153: 
154:         # Balanced objective: maximize accuracy + S_q while penalizing calibration error.
155:         objective = mean_acc + mean_sq - 0.5 * mean_ece
156: 
157:         results.append(
158:             {
159:                 "wait_penalty": wait_penalty,
160:                 "early_buzz_penalty": early_buzz_penalty,
161:                 "status": "ok",
162:                 "num_ok": len(ok),
163:                 "num_total": len(per_seed),
164:                 "seconds": round(mean_seconds, 3),
165:                 "buzz_accuracy": mean_acc,
166:                 "mean_sq": mean_sq,
167:                 "mean_buzz_step": mean_step,
168:                 "ece": mean_ece,
169:                 "brier": mean_brier,
170:                 "objective": objective,
171:                 "per_seed": per_seed,
172:             }
173:         )
174: 
175:     # cleanup temp config
176:     if TMP_CONFIG.exists():
177:         TMP_CONFIG.unlink()
178: 
179:     out_json = ARTIFACT_DIR / "reward_sweep_results.json"
180:     out_csv = ARTIFACT_DIR / "reward_sweep_results.csv"
181: 
182:     with out_json.open("w", encoding="utf-8") as f:
183:         json.dump(results, f, indent=2)
184: 
185:     fields = [
186:         "wait_penalty",
187:         "early_buzz_penalty",
188:         "status",
189:         "num_ok",
190:         "num_total",
191:         "seconds",
192:         "buzz_accuracy",
193:         "mean_sq",
194:         "mean_buzz_step",
195:         "ece",
196:         "brier",
197:         "objective",
198:     ]
199:     with out_csv.open("w", newline="", encoding="utf-8") as f:
200:         writer = csv.DictWriter(f, fieldnames=fields)
201:         writer.writeheader()
202:         for row in results:
203:             flat = {k: row.get(k, "") for k in fields}
204:             writer.writerow(flat)
205: 
206:     ok_runs = [r for r in results if r.get("status") == "ok"]
207:     if not ok_runs:
208:         print("No successful runs.")
209:         return 1
210: 
211:     best = max(ok_runs, key=lambda r: float(r.get("objective", 0.0)))
212: 
213:     print("\nBest run:")
214:     print(best)
215:     print(f"Wrote: {out_json}")
216:     print(f"Wrote: {out_csv}")
217:     return 0
218: 
219: 
220: if __name__ == "__main__":
221:     raise SystemExit(main())
````

## File: scripts/test_mc_builder.py
````python
  1: #!/usr/bin/env python
  2: """Test script to verify MC construction with anti-artifact guards."""
  3: 
  4: import os
  5: import sys
  6: from pathlib import Path
  7: 
  8: # Add parent directory to path for imports
  9: sys.path.insert(0, str(Path(__file__).parent.parent))
 10: 
 11: from qb_data.data_loader import QANTADatasetLoader
 12: from qb_data.answer_profiles import AnswerProfileBuilder
 13: from qb_data.mc_builder import MCBuilder, MCQuestion
 14: from qb_data.config import load_config
 15: 
 16: 
 17: def main():
 18:     """Test MC question construction with guards."""
 19:     print("Testing MC Builder with Anti-Artifact Guards")
 20:     print("=" * 50)
 21: 
 22:     # Load configuration
 23:     config = load_config("configs/default.yaml")
 24: 
 25:     # Load test questions
 26:     data_path = "data/test_questions.csv"
 27:     if not os.path.exists(data_path):
 28:         print(f"Error: Test data not found at {data_path}")
 29:         print("Please ensure test_questions.csv exists")
 30:         return 1
 31: 
 32:     # Load questions
 33:     questions = QANTADatasetLoader.load_from_csv(data_path)
 34:     print(f"\nLoaded {len(questions)} test questions")
 35: 
 36:     # Create answer profile builder
 37:     profile_builder = AnswerProfileBuilder(
 38:         max_tokens_per_profile=config["answer_profiles"]["max_tokens_per_profile"],
 39:         min_questions_per_answer=config["answer_profiles"]["min_questions_per_answer"]
 40:     )
 41:     profile_builder.fit(questions)
 42:     print(f"Built profiles for {len(profile_builder._grouped)} unique answers")
 43: 
 44:     # Create MC builder with guards from config
 45:     mc_builder = MCBuilder(
 46:         K=config["data"]["K"],
 47:         strategy="tfidf_profile",  # Use TF-IDF since it doesn't require embeddings
 48:         alias_edit_distance_threshold=config["mc_guards"]["alias_edit_distance_threshold"],
 49:         duplicate_token_overlap_threshold=config["mc_guards"]["duplicate_token_overlap_threshold"],
 50:         max_length_ratio=config["mc_guards"]["max_length_ratio"],
 51:         random_seed=config["data"]["shuffle_seed"]
 52:     )
 53: 
 54:     # Build MC questions
 55:     print(f"\nBuilding MC questions with K={config['data']['K']} options...")
 56:     mc_questions = mc_builder.build(questions, profile_builder)
 57:     print(f"Created {len(mc_questions)} MC questions (from {len(questions)} originals)")
 58: 
 59:     # Calculate rejection rate
 60:     rejection_rate = 1.0 - (len(mc_questions) / len(questions))
 61:     print(f"Rejection rate: {rejection_rate:.1%} (due to guard violations)")
 62: 
 63:     # Print sample MC questions
 64:     print("\n" + "=" * 50)
 65:     print("Sample MC Questions:")
 66:     print("=" * 50)
 67: 
 68:     for i, mc_q in enumerate(mc_questions[:3]):  # Show first 3
 69:         print(f"\n[Question {i+1}]")
 70:         print(f"Category: {mc_q.category or 'Unknown'}")
 71:         print(f"Question ID: {mc_q.qid}")
 72: 
 73:         # Show first clue (truncated)
 74:         first_clue = mc_q.tokens[0] if mc_q.tokens else mc_q.question[:100]
 75:         print(f"First clue: {first_clue[:150]}...")
 76: 
 77:         print(f"\nOptions:")
 78:         for j, option in enumerate(mc_q.options):
 79:             marker = " [CORRECT]" if j == mc_q.gold_index else ""
 80:             print(f"  {j+1}. {option}{marker}")
 81: 
 82:         print(f"\nDistractor strategy: {mc_q.distractor_strategy}")
 83: 
 84:         # Check guards for this question
 85:         print("\nGuard checks:")
 86: 
 87:         # Check alias collision
 88:         gold_aliases = [mc_q.answer_primary] + list(mc_q.clean_answers)
 89:         alias_violations = []
 90:         for j, option in enumerate(mc_q.options):
 91:             if j != mc_q.gold_index:
 92:                 for alias in gold_aliases:
 93:                     from difflib import SequenceMatcher
 94:                     dist = 1.0 - SequenceMatcher(None, option.lower(), alias.lower()).ratio()
 95:                     if dist < 0.2:
 96:                         alias_violations.append((option, alias, dist))
 97: 
 98:         if alias_violations:
 99:             print(f"  ✗ Alias collision detected: {alias_violations}")
100:         else:
101:             print("  ✓ No alias collisions")
102: 
103:         # Check token overlap between options
104:         from qb_data.mc_builder import _token_overlap
105:         high_overlaps = []
106:         for j in range(len(mc_q.options)):
107:             for k in range(j+1, len(mc_q.options)):
108:                 overlap = _token_overlap(mc_q.options[j], mc_q.options[k])
109:                 if overlap > 0.8:
110:                     high_overlaps.append((mc_q.options[j], mc_q.options[k], overlap))
111: 
112:         if high_overlaps:
113:             print(f"  ✗ High token overlap: {high_overlaps}")
114:         else:
115:             print("  ✓ No high token overlaps")
116: 
117:         # Check length ratio
118:         lengths = [len(o.split()) for o in mc_q.options]
119:         ratio = max(lengths) / max(1, min(lengths))
120:         if ratio > 3.0:
121:             print(f"  ✗ Length ratio violation: {ratio:.2f} (max: {max(lengths)}, min: {min(lengths)})")
122:         else:
123:             print(f"  ✓ Length ratio OK: {ratio:.2f}")
124: 
125:         # Check question overlap
126:         from qb_data.text_utils import normalize_answer
127:         q_norm = normalize_answer(mc_q.question).lower()
128:         overlaps = []
129:         for option in mc_q.options:
130:             o_norm = normalize_answer(option).lower()
131:             if o_norm and o_norm in q_norm:
132:                 overlaps.append(option)
133: 
134:         if overlaps:
135:             print(f"  ✗ Options appear in question: {overlaps}")
136:         else:
137:             print("  ✓ No options in question text")
138: 
139:     # Print statistics
140:     print("\n" + "=" * 50)
141:     print("Statistics:")
142:     print("=" * 50)
143:     print(f"Total questions processed: {len(questions)}")
144:     print(f"MC questions built: {len(mc_questions)}")
145:     print(f"Questions rejected by guards: {len(questions) - len(mc_questions)}")
146: 
147:     # Analyze rejection reasons (would need to track in MCBuilder for full details)
148:     if len(mc_questions) < len(questions):
149:         print("\nNote: Some questions were rejected due to guard violations.")
150:         print("Common reasons include:")
151:         print("  - Not enough valid distractors after alias/duplicate filtering")
152:         print("  - Length ratio violations between options")
153:         print("  - Answer text appearing in question")
154: 
155:     print("\n✓ MC questions built successfully with guards active")
156:     return 0
157: 
158: 
159: if __name__ == "__main__":
160:     exit(main())
````

## File: tests/conftest.py
````python
  1: """Shared pytest fixtures for test suites.
  2: 
  3: Provides reusable test data for environment, likelihood, features,
  4: factory, and agent test suites. All fixtures create minimal but complete
  5: data structures that satisfy the interfaces expected by the codebase modules.
  6: 
  7: Fixtures
  8: --------
  9: sample_mc_question
 10:     A single MCQuestion with 4 options (gold_index=0), 6 clue steps,
 11:     and pre-computed cumulative prefixes. Suitable for environment and
 12:     feature extraction tests.
 13: 
 14: sample_config
 15:     A minimal config dict matching the YAML structure expected by
 16:     ``make_env_from_config`` and ``build_likelihood_from_config``.
 17:     Uses "simple" reward mode for predictable test outcomes.
 18: 
 19: sample_corpus
 20:     A list of 10 short text strings about US presidents and historical
 21:     events. Suitable for fitting TF-IDF vectorizers in tests.
 22: 
 23: sample_tfidf_env
 24:     A TossupMCEnv with TF-IDF likelihood and 3 sample MCQuestions.
 25:     Fast to construct, suitable for agent and PPO tests.
 26: """
 27: 
 28: from __future__ import annotations
 29: 
 30: from typing import TYPE_CHECKING
 31: 
 32: import pytest
 33: 
 34: from qb_data.mc_builder import MCQuestion
 35: 
 36: if TYPE_CHECKING:
 37:     from qb_env.tossup_env import TossupMCEnv
 38: 
 39: 
 40: @pytest.fixture
 41: def sample_mc_question() -> MCQuestion:
 42:     """Return a minimal MCQuestion for testing.
 43: 
 44:     The question is about the first US president with 4 answer options.
 45:     Gold answer is "George Washington" at index 0. Six clue steps are
 46:     defined via run_indices with pre-computed cumulative prefixes.
 47: 
 48:     Returns
 49:     -------
 50:     MCQuestion
 51:         A complete MCQuestion suitable for environment testing.
 52:     """
 53:     tokens = [
 54:         "Who", "was", "the", "first", "president",
 55:         "of", "the", "United", "States", "?",
 56:     ]
 57:     run_indices = [0, 2, 4, 6, 8, 9]
 58:     cumulative_prefixes = [
 59:         "Who",
 60:         "Who was the",
 61:         "Who was the first president",
 62:         "Who was the first president of the",
 63:         "Who was the first president of the United States",
 64:         "Who was the first president of the United States ?",
 65:     ]
 66:     return MCQuestion(
 67:         qid="test_q1",
 68:         question="Who was the first president of the United States?",
 69:         tokens=tokens,
 70:         answer_primary="George Washington",
 71:         clean_answers=["George Washington", "Washington"],
 72:         run_indices=run_indices,
 73:         human_buzz_positions=[],
 74:         category="History",
 75:         cumulative_prefixes=cumulative_prefixes,
 76:         options=[
 77:             "George Washington",
 78:             "Thomas Jefferson",
 79:             "John Adams",
 80:             "Benjamin Franklin",
 81:         ],
 82:         gold_index=0,
 83:         option_profiles=[
 84:             "George Washington first president commander revolutionary war continental army",
 85:             "Thomas Jefferson third president declaration independence Virginia",
 86:             "John Adams second president Massachusetts diplomat",
 87:             "Benjamin Franklin inventor diplomat Philadelphia printing press",
 88:         ],
 89:         option_answer_primary=[
 90:             "George Washington",
 91:             "Thomas Jefferson",
 92:             "John Adams",
 93:             "Benjamin Franklin",
 94:         ],
 95:         distractor_strategy="test",
 96:     )
 97: 
 98: 
 99: @pytest.fixture
100: def sample_config() -> dict:
101:     """Return a minimal config dict for factory tests.
102: 
103:     Matches the YAML structure expected by ``make_env_from_config`` and
104:     ``build_likelihood_from_config``. Uses "simple" reward mode and
105:     "from_scratch" belief mode for predictable test outcomes.
106: 
107:     Returns
108:     -------
109:     dict
110:         Config dict with data, environment, and likelihood sections.
111:     """
112:     return {
113:         "data": {"K": 4},
114:         "environment": {
115:             "reward": "simple",
116:             "wait_penalty": 0.0,
117:             "buzz_correct": 1.0,
118:             "buzz_incorrect": -1.0,
119:             "belief_mode": "from_scratch",
120:         },
121:         "likelihood": {
122:             "model": "sbert",
123:             "beta": 5.0,
124:         },
125:     }
126: 
127: 
128: @pytest.fixture
129: def sample_corpus() -> list[str]:
130:     """Return a list of 10 short text strings for TF-IDF fitting.
131: 
132:     Topics cover US presidents and major historical events, providing
133:     sufficient vocabulary variety for TF-IDF vectorizer tests.
134: 
135:     Returns
136:     -------
137:     list[str]
138:         Ten text strings suitable for corpus fitting.
139:     """
140:     return [
141:         "George Washington was the first president of the United States",
142:         "Thomas Jefferson wrote the Declaration of Independence",
143:         "John Adams served as the second president after Washington",
144:         "Benjamin Franklin was an inventor and diplomat in Philadelphia",
145:         "Abraham Lincoln freed the slaves during the Civil War",
146:         "Alexander Hamilton established the national banking system",
147:         "James Madison authored the Bill of Rights and Constitution",
148:         "Andrew Jackson was a military hero and populist president",
149:         "The American Revolution established independence from Britain",
150:         "The Constitution created a federal system of government",
151:     ]
152: 
153: 
154: @pytest.fixture(scope="module")
155: def sample_t5_model():
156:     """Return a T5Likelihood model for testing.
157: 
158:     Uses t5-small (60M params) for fast test execution. Scoped to module
159:     level so the model is loaded once per test file, not per test function.
160: 
161:     Returns
162:     -------
163:     T5Likelihood
164:         A T5 likelihood model suitable for testing semantic scoring.
165: 
166:     Notes
167:     -----
168:     This fixture may take 5-10 seconds on first run to download the model
169:     from HuggingFace. Subsequent runs use cached weights.
170:     """
171:     from models.likelihoods import T5Likelihood
172: 
173:     return T5Likelihood(model_name="t5-small")
174: 
175: 
176: @pytest.fixture
177: def sample_tfidf_env(sample_mc_question: MCQuestion) -> "TossupMCEnv":
178:     """Return a TossupMCEnv with TF-IDF likelihood and 3 sample questions.
179: 
180:     Creates a lightweight environment suitable for PPOBuzzer and agent
181:     tests. Uses TF-IDF likelihood for fast execution (< 1ms per score).
182:     Three copies of the sample question are used to provide enough data
183:     for environment sampling.
184: 
185:     Returns
186:     -------
187:     TossupMCEnv
188:         A configured environment with simple reward mode.
189:     """
190:     from models.likelihoods import TfIdfLikelihood
191:     from qb_env.tossup_env import TossupMCEnv
192: 
193:     corpus = sample_mc_question.option_profiles[:]
194:     model = TfIdfLikelihood(corpus_texts=corpus)
195: 
196:     # Use 3 copies for variety in sampling
197:     questions = [sample_mc_question] * 3
198:     return TossupMCEnv(
199:         questions=questions,
200:         likelihood_model=model,
201:         K=4,
202:         reward_mode="simple",
203:         wait_penalty=0.0,
204:         buzz_correct=1.0,
205:         buzz_incorrect=-1.0,
206:         belief_mode="from_scratch",
207:         beta=5.0,
208:     )
````

## File: tests/test_ppo_t5.py
````python
  1: """Unit tests for custom PPO trainer for T5PolicyModel.
  2: 
  3: Tests cover RolloutStep dataclass, RolloutBuffer with GAE computation,
  4: rollout collection with memory management, dynamic padding, and PPO update.
  5: 
  6: Uses t5-small (60M params) and TF-IDF likelihood for fast execution.
  7: The T5 model fixture is module-scoped (loaded once per test file).
  8: """
  9: 
 10: from __future__ import annotations
 11: 
 12: import pytest
 13: import torch
 14: import numpy as np
 15: 
 16: from training.train_ppo_t5 import RolloutStep, RolloutBuffer, PPOTrainer
 17: 
 18: 
 19: # ---------------------------------------------------------------------------
 20: # Fixtures
 21: # ---------------------------------------------------------------------------
 22: 
 23: 
 24: @pytest.fixture(scope="module")
 25: def t5_ppo_config() -> dict:
 26:     """Minimal PPO config for testing."""
 27:     return {
 28:         "model_name": "t5-small",
 29:         "device": "cpu",
 30:         "max_input_length": 64,
 31:         "num_choices": 4,
 32:         "ppo_lr": 1e-4,
 33:         "ppo_iterations": 2,
 34:         "ppo_batch_size": 4,
 35:         "ppo_epochs_per_iter": 2,
 36:         "ppo_gamma": 0.99,
 37:         "ppo_gae_lambda": 0.95,
 38:         "ppo_clip_ratio": 0.2,
 39:         "ppo_value_coef": 0.5,
 40:         "ppo_entropy_coef": 0.01,
 41:         "ppo_max_grad_norm": 0.5,
 42:         "ppo_episodes_per_iter": 2,
 43:         "eval_interval": 1,
 44:         "save_interval": 100,
 45:         "checkpoint_dir": "/tmp/test_ppo_t5_checkpoints",
 46:         "reward_time_penalty": 0.01,
 47:     }
 48: 
 49: 
 50: @pytest.fixture(scope="module")
 51: def t5_ppo_model(t5_ppo_config):
 52:     """Load T5PolicyModel with t5-small once per test module."""
 53:     from models.t5_policy import T5PolicyModel
 54: 
 55:     model = T5PolicyModel(t5_ppo_config)
 56:     return model
 57: 
 58: 
 59: @pytest.fixture
 60: def sample_rollout_steps() -> list:
 61:     """Create sample RolloutStep instances for testing GAE computation."""
 62:     # Simulate a 4-step episode: WAIT, WAIT, WAIT, BUZZ(correct)
 63:     steps = [
 64:         RolloutStep(
 65:             observation_text="CLUES: Who | CHOICES: (1) A (2) B (3) C (4) D",
 66:             action=0,
 67:             reward=-0.01,
 68:             done=False,
 69:             value=0.2,
 70:             log_prob=-0.8,
 71:             input_ids=torch.randint(0, 100, (1, 10)),
 72:             attention_mask=torch.ones(1, 10, dtype=torch.long),
 73:         ),
 74:         RolloutStep(
 75:             observation_text="CLUES: Who was | CHOICES: (1) A (2) B (3) C (4) D",
 76:             action=0,
 77:             reward=-0.01,
 78:             done=False,
 79:             value=0.4,
 80:             log_prob=-0.7,
 81:             input_ids=torch.randint(0, 100, (1, 12)),
 82:             attention_mask=torch.ones(1, 12, dtype=torch.long),
 83:         ),
 84:         RolloutStep(
 85:             observation_text="CLUES: Who was the first | CHOICES: (1) A (2) B (3) C (4) D",
 86:             action=0,
 87:             reward=-0.01,
 88:             done=False,
 89:             value=0.6,
 90:             log_prob=-0.5,
 91:             input_ids=torch.randint(0, 100, (1, 15)),
 92:             attention_mask=torch.ones(1, 15, dtype=torch.long),
 93:         ),
 94:         RolloutStep(
 95:             observation_text="CLUES: Who was the first president | CHOICES: (1) A (2) B (3) C (4) D",
 96:             action=1,
 97:             reward=1.0,
 98:             done=True,
 99:             value=0.8,
100:             log_prob=-0.3,
101:             input_ids=torch.randint(0, 100, (1, 18)),
102:             attention_mask=torch.ones(1, 18, dtype=torch.long),
103:         ),
104:     ]
105:     return steps
106: 
107: 
108: # ---------------------------------------------------------------------------
109: # RolloutStep Tests
110: # ---------------------------------------------------------------------------
111: 
112: 
113: class TestRolloutStep:
114:     """Tests for the RolloutStep dataclass."""
115: 
116:     def test_rollout_step_dataclass(self):
117:         """RolloutStep stores all required fields."""
118:         step = RolloutStep(
119:             observation_text="test",
120:             action=0,
121:             reward=1.0,
122:             done=True,
123:             value=0.5,
124:             log_prob=-0.3,
125:         )
126:         assert step.observation_text == "test"
127:         assert step.action == 0
128:         assert step.reward == 1.0
129:         assert step.done is True
130:         assert step.value == 0.5
131:         assert step.log_prob == -0.3
132:         assert step.input_ids is None
133:         assert step.attention_mask is None
134:         assert step.return_ == 0.0
135:         assert step.advantage == 0.0
136: 
137:     def test_rollout_step_with_tensors(self):
138:         """RolloutStep stores tensor fields on CPU."""
139:         ids = torch.randint(0, 100, (1, 10))
140:         mask = torch.ones(1, 10, dtype=torch.long)
141:         step = RolloutStep(
142:             observation_text="test",
143:             action=1,
144:             reward=0.5,
145:             done=False,
146:             value=0.3,
147:             log_prob=-0.5,
148:             input_ids=ids,
149:             attention_mask=mask,
150:         )
151:         assert step.input_ids is not None
152:         assert step.input_ids.device.type == "cpu"
153:         assert step.attention_mask.device.type == "cpu"
154:         assert step.input_ids.shape == (1, 10)
155: 
156: 
157: # ---------------------------------------------------------------------------
158: # RolloutBuffer Tests
159: # ---------------------------------------------------------------------------
160: 
161: 
162: class TestRolloutBuffer:
163:     """Tests for the RolloutBuffer class."""
164: 
165:     def test_rollout_buffer_add(self, sample_rollout_steps):
166:         """Buffer accumulates rollouts correctly."""
167:         buffer = RolloutBuffer()
168:         assert len(buffer) == 0
169: 
170:         buffer.add_rollout(sample_rollout_steps)
171:         assert len(buffer) == 1
172: 
173:         buffer.add_rollout(sample_rollout_steps[:2])
174:         assert len(buffer) == 2
175: 
176:     def test_rollout_buffer_get_all_steps(self, sample_rollout_steps):
177:         """get_all_steps returns flat list of all steps."""
178:         buffer = RolloutBuffer()
179:         buffer.add_rollout(sample_rollout_steps)
180:         buffer.add_rollout(sample_rollout_steps[:2])
181: 
182:         all_steps = buffer.get_all_steps()
183:         assert len(all_steps) == 6  # 4 + 2
184: 
185:     def test_rollout_buffer_reset(self, sample_rollout_steps):
186:         """reset() clears all rollouts."""
187:         buffer = RolloutBuffer()
188:         buffer.add_rollout(sample_rollout_steps)
189:         assert len(buffer) == 1
190: 
191:         buffer.reset()
192:         assert len(buffer) == 0
193:         assert len(buffer.get_all_steps()) == 0
194: 
195:     def test_gae_computation(self, sample_rollout_steps):
196:         """GAE advantages match hand-calculated values.
197: 
198:         Episode: 4 steps with rewards [-0.01, -0.01, -0.01, 1.0]
199:         and values [0.2, 0.4, 0.6, 0.8].
200:         """
201:         buffer = RolloutBuffer()
202:         buffer.add_rollout(sample_rollout_steps)
203: 
204:         gamma = 0.99
205:         gae_lambda = 0.95
206: 
207:         buffer.compute_returns_and_advantages(gamma, gae_lambda)
208: 
209:         all_steps = buffer.get_all_steps()
210: 
211:         # Verify terminal step (t=3): done=True
212:         # delta_3 = r_3 + gamma * 0 - v_3 = 1.0 + 0 - 0.8 = 0.2
213:         # gae_3 = delta_3 = 0.2 (reset because done=True)
214:         assert abs(all_steps[3].advantage - 0.2) < 1e-6
215:         assert abs(all_steps[3].return_ - (0.2 + 0.8)) < 1e-6  # adv + value
216: 
217:         # Step t=2: not done
218:         # delta_2 = r_2 + gamma * v_3 - v_2 = -0.01 + 0.99 * 0.8 - 0.6 = 0.182
219:         # gae_2 = delta_2 + gamma * lambda * gae_3 = 0.182 + 0.99 * 0.95 * 0.2
220:         delta_2 = -0.01 + gamma * 0.8 - 0.6
221:         gae_2 = delta_2 + gamma * gae_lambda * 0.2
222:         assert abs(all_steps[2].advantage - gae_2) < 1e-6
223: 
224:         # Step t=1:
225:         # delta_1 = r_1 + gamma * v_2 - v_1 = -0.01 + 0.99 * 0.6 - 0.4
226:         delta_1 = -0.01 + gamma * 0.6 - 0.4
227:         gae_1 = delta_1 + gamma * gae_lambda * gae_2
228:         assert abs(all_steps[1].advantage - gae_1) < 1e-6
229: 
230:         # Step t=0:
231:         delta_0 = -0.01 + gamma * 0.4 - 0.2
232:         gae_0 = delta_0 + gamma * gae_lambda * gae_1
233:         assert abs(all_steps[0].advantage - gae_0) < 1e-6
234: 
235:     def test_gae_multiple_episodes(self, sample_rollout_steps):
236:         """GAE handles multiple episodes independently."""
237:         buffer = RolloutBuffer()
238: 
239:         # Two episodes
240:         buffer.add_rollout(sample_rollout_steps)
241:         buffer.add_rollout(sample_rollout_steps[:2] + [
242:             RolloutStep(
243:                 observation_text="end",
244:                 action=2,
245:                 reward=-1.0,
246:                 done=True,
247:                 value=0.1,
248:                 log_prob=-1.0,
249:             )
250:         ])
251: 
252:         buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
253: 
254:         all_steps = buffer.get_all_steps()
255:         # All steps should have return_ and advantage set
256:         for step in all_steps:
257:             assert isinstance(step.return_, float)
258:             assert isinstance(step.advantage, float)
259: 
260: 
261: # ---------------------------------------------------------------------------
262: # Dynamic Padding Tests
263: # ---------------------------------------------------------------------------
264: 
265: 
266: class TestDynamicPadding:
267:     """Tests for dynamic batch padding."""
268: 
269:     def test_dynamic_padding(self, t5_ppo_model, t5_ppo_config, sample_mc_question):
270:         """Padding works with variable-length sequences."""
271:         trainer = PPOTrainer(
272:             model=t5_ppo_model,
273:             train_questions=[sample_mc_question] * 3,
274:             val_questions=[sample_mc_question] * 2,
275:             config=t5_ppo_config,
276:         )
277: 
278:         # Create steps with different sequence lengths
279:         steps = [
280:             RolloutStep(
281:                 observation_text="short",
282:                 action=0,
283:                 reward=0.0,
284:                 done=False,
285:                 value=0.1,
286:                 log_prob=-0.5,
287:                 input_ids=torch.randint(0, 100, (1, 5)),
288:                 attention_mask=torch.ones(1, 5, dtype=torch.long),
289:             ),
290:             RolloutStep(
291:                 observation_text="this is a longer sequence",
292:                 action=1,
293:                 reward=1.0,
294:                 done=True,
295:                 value=0.8,
296:                 log_prob=-0.2,
297:                 input_ids=torch.randint(0, 100, (1, 15)),
298:                 attention_mask=torch.ones(1, 15, dtype=torch.long),
299:             ),
300:             RolloutStep(
301:                 observation_text="medium",
302:                 action=0,
303:                 reward=0.0,
304:                 done=False,
305:                 value=0.3,
306:                 log_prob=-0.6,
307:                 input_ids=torch.randint(0, 100, (1, 10)),
308:                 attention_mask=torch.ones(1, 10, dtype=torch.long),
309:             ),
310:         ]
311: 
312:         input_ids, attention_mask = trainer._pad_batch(steps)
313: 
314:         # All padded to max length in batch (15)
315:         assert input_ids.shape == (3, 15)
316:         assert attention_mask.shape == (3, 15)
317: 
318:         # First sequence (len 5) should have 10 padding tokens
319:         assert attention_mask[0, :5].sum() == 5
320:         assert attention_mask[0, 5:].sum() == 0
321: 
322:         # Second sequence (len 15) should have no padding
323:         assert attention_mask[1].sum() == 15
324: 
325:         # Third sequence (len 10) should have 5 padding tokens
326:         assert attention_mask[2, :10].sum() == 10
327:         assert attention_mask[2, 10:].sum() == 0
328: 
329: 
330: # ---------------------------------------------------------------------------
331: # Memory Management Tests
332: # ---------------------------------------------------------------------------
333: 
334: 
335: class TestMemoryManagement:
336:     """Tests for memory-safe tensor handling."""
337: 
338:     def test_memory_management_cpu_storage(self, sample_rollout_steps):
339:         """Rollout tensors are stored on CPU, not GPU."""
340:         for step in sample_rollout_steps:
341:             if step.input_ids is not None:
342:                 assert step.input_ids.device.type == "cpu", (
343:                     f"input_ids on {step.input_ids.device}, expected CPU"
344:                 )
345:             if step.attention_mask is not None:
346:                 assert step.attention_mask.device.type == "cpu", (
347:                     f"attention_mask on {step.attention_mask.device}, expected CPU"
348:                 )
349: 
350:     def test_rollout_tensors_are_detached(self, sample_rollout_steps):
351:         """Stored tensors do not require gradients."""
352:         for step in sample_rollout_steps:
353:             if step.input_ids is not None:
354:                 assert not step.input_ids.requires_grad
355:             if step.attention_mask is not None:
356:                 assert not step.attention_mask.requires_grad
357: 
358: 
359: # ---------------------------------------------------------------------------
360: # PPO Update Tests
361: # ---------------------------------------------------------------------------
362: 
363: 
364: class TestPPOUpdate:
365:     """Tests for PPO policy updates."""
366: 
367:     def test_ppo_update_no_oom(
368:         self, t5_ppo_model, t5_ppo_config, sample_mc_question
369:     ):
370:         """update_policy completes without OOM or errors."""
371:         trainer = PPOTrainer(
372:             model=t5_ppo_model,
373:             train_questions=[sample_mc_question] * 3,
374:             val_questions=[sample_mc_question] * 2,
375:             config=t5_ppo_config,
376:         )
377: 
378:         # Create a small buffer with tokenized steps
379:         buffer = RolloutBuffer()
380:         texts = [
381:             "CLUES: Who | CHOICES: (1) A (2) B (3) C (4) D",
382:             "CLUES: Who was | CHOICES: (1) A (2) B (3) C (4) D",
383:             "CLUES: Who was the | CHOICES: (1) A (2) B (3) C (4) D",
384:         ]
385: 
386:         rollout = []
387:         for i, text in enumerate(texts):
388:             inputs = t5_ppo_model.tokenizer(
389:                 text,
390:                 return_tensors="pt",
391:                 padding=True,
392:                 truncation=True,
393:                 max_length=64,
394:             )
395:             is_last = i == len(texts) - 1
396:             step = RolloutStep(
397:                 observation_text=text,
398:                 action=0 if not is_last else 1,
399:                 reward=-0.01 if not is_last else 1.0,
400:                 done=is_last,
401:                 value=0.1 * (i + 1),
402:                 log_prob=-0.5,
403:                 input_ids=inputs["input_ids"].detach().cpu(),
404:                 attention_mask=inputs["attention_mask"].detach().cpu(),
405:             )
406:             rollout.append(step)
407: 
408:         buffer.add_rollout(rollout)
409: 
410:         # Should complete without errors
411:         metrics = trainer.update_policy(buffer)
412: 
413:         assert "policy_loss" in metrics
414:         assert "value_loss" in metrics
415:         assert "entropy" in metrics
416:         assert metrics["num_updates"] > 0
417: 
418:     def test_ppo_update_empty_buffer(
419:         self, t5_ppo_model, t5_ppo_config, sample_mc_question
420:     ):
421:         """update_policy handles empty buffer gracefully."""
422:         trainer = PPOTrainer(
423:             model=t5_ppo_model,
424:             train_questions=[sample_mc_question] * 3,
425:             val_questions=[sample_mc_question] * 2,
426:             config=t5_ppo_config,
427:         )
428: 
429:         buffer = RolloutBuffer()
430:         metrics = trainer.update_policy(buffer)
431: 
432:         assert metrics["num_updates"] == 0
433:         assert metrics["policy_loss"] == 0.0
434: 
435: 
436: # ---------------------------------------------------------------------------
437: # Rollout Collection Tests
438: # ---------------------------------------------------------------------------
439: 
440: 
441: class TestRolloutCollection:
442:     """Tests for rollout collection."""
443: 
444:     def test_rollout_collection(
445:         self, t5_ppo_model, t5_ppo_config, sample_mc_question
446:     ):
447:         """collect_rollouts returns buffer with episodes."""
448:         trainer = PPOTrainer(
449:             model=t5_ppo_model,
450:             train_questions=[sample_mc_question] * 3,
451:             val_questions=[sample_mc_question] * 2,
452:             config=t5_ppo_config,
453:         )
454: 
455:         buffer = trainer.collect_rollouts(num_episodes=2)
456: 
457:         assert len(buffer) == 2  # 2 episodes collected
458:         all_steps = buffer.get_all_steps()
459:         assert len(all_steps) > 0  # At least some steps
460: 
461:         # Each step should have text, action, reward, tensors
462:         for step in all_steps:
463:             assert isinstance(step.observation_text, str)
464:             assert isinstance(step.action, int)
465:             assert 0 <= step.action <= 4  # WAIT or SELECT
466:             assert step.input_ids is not None
467:             assert step.attention_mask is not None
468:             # Tensors should be on CPU
469:             assert step.input_ids.device.type == "cpu"
470:             assert step.attention_mask.device.type == "cpu"
471: 
472:     def test_rollout_episodes_terminate(
473:         self, t5_ppo_model, t5_ppo_config, sample_mc_question
474:     ):
475:         """All collected episodes properly terminate."""
476:         trainer = PPOTrainer(
477:             model=t5_ppo_model,
478:             train_questions=[sample_mc_question] * 3,
479:             val_questions=[sample_mc_question] * 2,
480:             config=t5_ppo_config,
481:         )
482: 
483:         buffer = trainer.collect_rollouts(num_episodes=3)
484: 
485:         for rollout in buffer.rollouts:
486:             # Last step should be done
487:             assert rollout[-1].done, "Episode should terminate"
488:             # Non-terminal steps should not be done
489:             for step in rollout[:-1]:
490:                 assert not step.done, "Non-terminal step should not be done"
````

## File: tests/test_qb_rl_bridge.py
````python
  1: """Compatibility bridge tests for qb-rl surfaces ported into qanta-buzzer."""
  2: 
  3: from __future__ import annotations
  4: 
  5: import sys
  6: import types
  7: 
  8: import numpy as np
  9: import pytest
 10: 
 11: import agents.bayesian_buzzer as bayesian_buzzer
 12: import models.answer_profiles as compat_answer_profiles
 13: import models.likelihoods as likelihoods
 14: import qb_data.answer_profiles as qb_answer_profiles
 15: import qb_data.data_loader as qb_data_loader
 16: import qb_env.data_loader as compat_data_loader
 17: import qb_env.mc_builder as compat_mc_builder
 18: import qb_env.text_utils as compat_text_utils
 19: from agents.softmax_profile_buzzer import (
 20:     SequentialBayesBuzzer as CompatSequentialBayesBuzzer,
 21: )
 22: from agents.softmax_profile_buzzer import (
 23:     SoftmaxEpisodeResult as CompatSoftmaxEpisodeResult,
 24: )
 25: from agents.softmax_profile_buzzer import (
 26:     SoftmaxProfileBuzzer as CompatSoftmaxProfileBuzzer,
 27: )
 28: from models.likelihoods import OpenAILikelihood, build_likelihood_from_config
 29: from qb_data.mc_builder import MCBuilder
 30: 
 31: 
 32: def _install_fake_openai(monkeypatch, vectors: dict[str, list[float]], calls: list[tuple[str, tuple[str, ...]]]) -> None:
 33:     """Install a fake ``openai`` module that serves deterministic embeddings."""
 34: 
 35:     class FakeEmbeddingsClient:
 36:         def create(self, model: str, input: list[str]):
 37:             calls.append((model, tuple(input)))
 38:             return types.SimpleNamespace(
 39:                 data=[
 40:                     types.SimpleNamespace(embedding=vectors[text])
 41:                     for text in input
 42:                 ]
 43:             )
 44: 
 45:     class FakeOpenAI:
 46:         def __init__(self, api_key: str):
 47:             self.api_key = api_key
 48:             self.embeddings = FakeEmbeddingsClient()
 49: 
 50:     monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
 51: 
 52: 
 53: class TestOpenAILikelihood:
 54:     """Tests for optional OpenAI embedding support."""
 55: 
 56:     def test_openai_likelihood_requires_api_key(self, monkeypatch) -> None:
 57:         monkeypatch.delenv("OPENAI_API_KEY", raising=False)
 58:         with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
 59:             OpenAILikelihood()
 60: 
 61:     def test_openai_likelihood_scores_and_reuses_cache(self, monkeypatch) -> None:
 62:         calls: list[tuple[str, tuple[str, ...]]] = []
 63:         vectors = {
 64:             "first president": [2.0, 0.0],
 65:             "george washington": [3.0, 0.0],
 66:             "albert einstein": [0.0, 4.0],
 67:         }
 68:         _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
 69:         monkeypatch.setenv("OPENAI_API_KEY", "test-key")
 70: 
 71:         model = OpenAILikelihood(model="fake-embedding-model")
 72: 
 73:         embeddings = model._embed_batch(["first president", "george washington"])
 74:         norms = np.linalg.norm(embeddings, axis=1)
 75:         np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)
 76:         calls_before_score = len(calls)
 77: 
 78:         scores_1 = model.score(
 79:             "first president",
 80:             ["george washington", "albert einstein"],
 81:         )
 82:         assert scores_1[0] > scores_1[1]
 83:         assert len(calls) == calls_before_score + 2, (
 84:             "first score should call the embeddings API twice"
 85:         )
 86: 
 87:         scores_2 = model.score(
 88:             "first president",
 89:             ["george washington", "albert einstein"],
 90:         )
 91:         np.testing.assert_allclose(scores_1, scores_2, atol=1e-6)
 92:         assert len(calls) == calls_before_score + 2, "second score should be served from cache"
 93: 
 94:     def test_likelihood_factory_openai(self, monkeypatch) -> None:
 95:         calls: list[tuple[str, tuple[str, ...]]] = []
 96:         vectors = {"a": [1.0, 0.0]}
 97:         _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
 98:         monkeypatch.setenv("OPENAI_API_KEY", "test-key")
 99: 
100:         config = {"likelihood": {"model": "openai", "openai_model": "fake-openai"}}
101:         model = build_likelihood_from_config(config)
102: 
103:         assert isinstance(model, OpenAILikelihood)
104:         assert model.model == "fake-openai"
105: 
106: 
107: class TestOpenAIProfileStrategy:
108:     """Tests for OpenAI-backed distractor ranking."""
109: 
110:     def test_openai_profile_uses_openai_embeddings(self, monkeypatch) -> None:
111:         calls: list[str] = []
112:         embeddings = {
113:             "gold profile": np.array([1.0, 0.0], dtype=np.float32),
114:             "near distractor": np.array([0.9, 0.1], dtype=np.float32),
115:             "far distractor": np.array([0.0, 1.0], dtype=np.float32),
116:         }
117: 
118:         class FakeOpenAILikelihood:
119:             def __init__(self, model: str = "unused") -> None:
120:                 calls.append(model)
121: 
122:             def embed_and_cache(self, texts: list[str]) -> np.ndarray:
123:                 return np.stack([embeddings[text] for text in texts]).astype(np.float32)
124: 
125:         monkeypatch.setattr(likelihoods, "OpenAILikelihood", FakeOpenAILikelihood)
126: 
127:         builder = MCBuilder(strategy="openai_profile", openai_model="fake-openai")
128:         rankings = builder._compute_rankings(
129:             answers=["gold", "near", "far"],
130:             answer_profiles={
131:                 "gold": "gold profile",
132:                 "near": "near distractor",
133:                 "far": "far distractor",
134:             },
135:             answer_to_category={},
136:         )
137: 
138:         assert calls == ["fake-openai"]
139:         assert rankings["gold"][0] == "near"
140:         assert rankings["gold"][1] == "far"
141: 
142: 
143: class TestQBRLCompatibilityModules:
144:     """Tests for qb-rl import-path shims."""
145: 
146:     def test_module_aliases_resolve_expected_symbols(self) -> None:
147:         assert compat_answer_profiles.AnswerProfileBuilder is qb_answer_profiles.AnswerProfileBuilder
148:         assert compat_data_loader.parse_row is qb_data_loader.parse_row
149:         assert compat_mc_builder.MCBuilder.__name__ == "MCBuilder"
150:         assert compat_text_utils.normalize_answer("The Answer") == "answer"
151:         assert CompatSoftmaxProfileBuzzer is bayesian_buzzer.SoftmaxProfileBuzzer
152:         assert CompatSequentialBayesBuzzer is bayesian_buzzer.SequentialBayesBuzzer
153:         assert CompatSoftmaxEpisodeResult is bayesian_buzzer.SoftmaxEpisodeResult
154: 
155:     def test_parse_row_supports_qb_rl_metadata(self) -> None:
156:         question = compat_data_loader.parse_row(
157:             {
158:                 "qid": "q-1",
159:                 "question": "alpha beta gamma",
160:                 "answer_primary": "George Washington",
161:                 "clean_answers": ["George Washington", "Washington"],
162:                 "run_indices": [1, 2],
163:                 "metadata": {
164:                     "category": "History",
165:                     "human_buzz_positions": [{"position": 4, "count": 2}],
166:                 },
167:             }
168:         )
169: 
170:         assert question.qid == "q-1"
171:         assert question.category == "History"
172:         assert question.human_buzz_positions == [(4, 2)]
173:         assert question.cumulative_prefixes == ["alpha beta", "alpha beta gamma"]
174: 
175:     def test_load_tossup_questions_from_config_prefers_dataset_smoke(
176:         self, monkeypatch
177:     ) -> None:
178:         captured: dict[str, object] = {}
179:         sample_question = compat_data_loader.TossupQuestion(
180:             qid="hf-1",
181:             question="alpha beta",
182:             tokens=["alpha", "beta"],
183:             answer_primary="Answer",
184:             clean_answers=["Answer"],
185:             run_indices=[1],
186:             human_buzz_positions=None,
187:             category="History",
188:             cumulative_prefixes=["alpha beta"],
189:         )
190: 
191:         def fake_load_tossup_questions(
192:             dataset: str,
193:             dataset_config: str | None = None,
194:             split: str = "eval",
195:             limit: int | None = None,
196:         ):
197:             captured["dataset"] = dataset
198:             captured["dataset_config"] = dataset_config
199:             captured["split"] = split
200:             captured["limit"] = limit
201:             return [sample_question]
202: 
203:         monkeypatch.setattr(qb_data_loader, "load_tossup_questions", fake_load_tossup_questions)
204: 
205:         config = {
206:             "data": {
207:                 "dataset": "main-dataset",
208:                 "dataset_config": "main-config",
209:                 "dataset_smoke": "smoke-dataset",
210:                 "dataset_smoke_config": "smoke-config",
211:                 "split": "train",
212:             }
213:         }
214: 
215:         questions = compat_data_loader.load_tossup_questions_from_config(config, smoke=True)
216: 
217:         assert len(questions) == 1
218:         assert captured == {
219:             "dataset": "smoke-dataset",
220:             "dataset_config": "smoke-config",
221:             "split": "train",
222:             "limit": None,
223:         }
````

## File: tests/test_supervised_t5.py
````python
  1: """Unit tests for SupervisedTrainer and supervised training utilities.
  2: 
  3: Tests cover batch preparation, training epochs, gradient accumulation,
  4: checkpoint save/load, best model selection, and the run_supervised_training
  5: entry point.
  6: 
  7: Uses t5-small (60M params) for speed. The model fixture is module-scoped
  8: to load t5-small only once per test file.
  9: """
 10: 
 11: from __future__ import annotations
 12: 
 13: import json
 14: import os
 15: import tempfile
 16: 
 17: import pytest
 18: import torch
 19: 
 20: from models.t5_policy import T5PolicyModel
 21: from qb_data.mc_builder import MCQuestion
 22: from training.train_supervised_t5 import (
 23:     SupervisedTrainer,
 24:     format_question_text,
 25: )
 26: 
 27: 
 28: # ---------------------------------------------------------------------------
 29: # Fixtures
 30: # ---------------------------------------------------------------------------
 31: 
 32: 
 33: def _make_question(qid: str, gold_index: int = 0) -> MCQuestion:
 34:     """Create a minimal MCQuestion for testing."""
 35:     tokens = ["Who", "was", "the", "first", "president"]
 36:     return MCQuestion(
 37:         qid=qid,
 38:         question="Who was the first president",
 39:         tokens=tokens,
 40:         answer_primary="George Washington",
 41:         clean_answers=["George Washington"],
 42:         run_indices=[0, 2, 4],
 43:         human_buzz_positions=[],
 44:         category="History",
 45:         cumulative_prefixes=[
 46:             "Who",
 47:             "Who was the",
 48:             "Who was the first president",
 49:         ],
 50:         options=[
 51:             "George Washington",
 52:             "Thomas Jefferson",
 53:             "John Adams",
 54:             "Benjamin Franklin",
 55:         ],
 56:         gold_index=gold_index,
 57:         option_profiles=[
 58:             "George Washington first president",
 59:             "Thomas Jefferson third president",
 60:             "John Adams second president",
 61:             "Benjamin Franklin inventor diplomat",
 62:         ],
 63:         option_answer_primary=[
 64:             "George Washington",
 65:             "Thomas Jefferson",
 66:             "John Adams",
 67:             "Benjamin Franklin",
 68:         ],
 69:         distractor_strategy="test",
 70:     )
 71: 
 72: 
 73: @pytest.fixture(scope="module")
 74: def t5_small_model() -> T5PolicyModel:
 75:     """Load T5PolicyModel with t5-small once per test module."""
 76:     model = T5PolicyModel(
 77:         {
 78:             "model_name": "t5-small",
 79:             "device": "cpu",
 80:             "max_input_length": 64,
 81:             "num_choices": 4,
 82:         }
 83:     )
 84:     return model
 85: 
 86: 
 87: @pytest.fixture
 88: def train_questions() -> list[MCQuestion]:
 89:     """Return 8 training questions with varied gold indices."""
 90:     return [_make_question(f"train_{i}", i % 4) for i in range(8)]
 91: 
 92: 
 93: @pytest.fixture
 94: def val_questions() -> list[MCQuestion]:
 95:     """Return 4 validation questions."""
 96:     return [_make_question(f"val_{i}", i % 4) for i in range(4)]
 97: 
 98: 
 99: @pytest.fixture
100: def trainer_config(tmp_path) -> dict:
101:     """Return a minimal supervised trainer config using temp directory."""
102:     return {
103:         "model_name": "t5-small",
104:         "device": "cpu",
105:         "num_choices": 4,
106:         "supervised_lr": 1e-3,
107:         "supervised_epochs": 2,
108:         "supervised_batch_size": 2,
109:         "supervised_grad_accum_steps": 2,
110:         "max_input_length": 64,
111:         "max_grad_norm": 1.0,
112:         "weight_decay": 0.01,
113:         "checkpoint_dir": str(tmp_path / "checkpoints"),
114:     }
115: 
116: 
117: @pytest.fixture
118: def trainer(
119:     t5_small_model: T5PolicyModel,
120:     train_questions: list[MCQuestion],
121:     val_questions: list[MCQuestion],
122:     trainer_config: dict,
123: ) -> SupervisedTrainer:
124:     """Return a configured SupervisedTrainer instance."""
125:     return SupervisedTrainer(
126:         model=t5_small_model,
127:         train_questions=train_questions,
128:         val_questions=val_questions,
129:         config=trainer_config,
130:     )
131: 
132: 
133: # ---------------------------------------------------------------------------
134: # Format Tests
135: # ---------------------------------------------------------------------------
136: 
137: 
138: class TestFormatQuestionText:
139:     """Tests for the format_question_text utility."""
140: 
141:     def test_format_includes_all_tokens(self):
142:         """Formatted text includes all question tokens as clues."""
143:         q = _make_question("q1")
144:         text = format_question_text(q)
145:         assert "Who was the first president" in text
146: 
147:     def test_format_includes_all_choices(self):
148:         """Formatted text includes all 4 answer choices."""
149:         q = _make_question("q1")
150:         text = format_question_text(q)
151:         assert "(1) George Washington" in text
152:         assert "(2) Thomas Jefferson" in text
153:         assert "(3) John Adams" in text
154:         assert "(4) Benjamin Franklin" in text
155: 
156:     def test_format_structure(self):
157:         """Formatted text has CLUES: ... | CHOICES: ... structure."""
158:         q = _make_question("q1")
159:         text = format_question_text(q)
160:         assert text.startswith("CLUES: ")
161:         assert " | CHOICES: " in text
162: 
163: 
164: # ---------------------------------------------------------------------------
165: # Batch Preparation Tests
166: # ---------------------------------------------------------------------------
167: 
168: 
169: class TestPrepareBatch:
170:     """Tests for SupervisedTrainer.prepare_batch."""
171: 
172:     def test_prepare_batch_format(self, trainer: SupervisedTrainer):
173:         """Batch preparation produces correct tensor types and shapes."""
174:         questions = [_make_question(f"q{i}", i % 4) for i in range(3)]
175:         input_ids, attention_mask, labels = trainer.prepare_batch(questions)
176: 
177:         assert isinstance(input_ids, torch.Tensor)
178:         assert isinstance(attention_mask, torch.Tensor)
179:         assert isinstance(labels, torch.Tensor)
180:         assert input_ids.shape[0] == 3  # batch_size
181:         assert attention_mask.shape == input_ids.shape
182:         assert labels.shape == (3,)
183: 
184:     def test_prepare_batch_complete_questions(self, trainer: SupervisedTrainer):
185:         """Batch shows complete questions (all clues), not incremental."""
186:         q = _make_question("q1")
187:         input_ids, _, _ = trainer.prepare_batch([q])
188: 
189:         # Decode tokens to verify all clues are included
190:         decoded = trainer.model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
191:         # All tokens should be present in the decoded text
192:         assert "first" in decoded.lower()
193:         assert "president" in decoded.lower()
194: 
195:     def test_prepare_batch_labels_correct(self, trainer: SupervisedTrainer):
196:         """Labels match gold_index of each question."""
197:         questions = [
198:             _make_question("q0", gold_index=0),
199:             _make_question("q1", gold_index=2),
200:             _make_question("q2", gold_index=3),
201:         ]
202:         _, _, labels = trainer.prepare_batch(questions)
203:         assert labels.tolist() == [0, 2, 3]
204: 
205: 
206: # ---------------------------------------------------------------------------
207: # Training Tests
208: # ---------------------------------------------------------------------------
209: 
210: 
211: class TestTrainEpoch:
212:     """Tests for SupervisedTrainer.train_epoch."""
213: 
214:     def test_training_epoch_completes(self, trainer: SupervisedTrainer):
215:         """One epoch completes without errors."""
216:         loss, acc = trainer.train_epoch()
217: 
218:         assert isinstance(loss, float)
219:         assert isinstance(acc, float)
220:         assert loss > 0, "Loss should be positive"
221:         assert 0 <= acc <= 1, "Accuracy should be in [0, 1]"
222: 
223:     def test_gradient_accumulation(
224:         self,
225:         t5_small_model: T5PolicyModel,
226:         train_questions: list[MCQuestion],
227:         val_questions: list[MCQuestion],
228:         tmp_path,
229:     ):
230:         """Optimizer updates only on accumulation steps (not every batch)."""
231:         config = {
232:             "supervised_lr": 1e-3,
233:             "supervised_epochs": 1,
234:             "supervised_batch_size": 2,
235:             "supervised_grad_accum_steps": 4,  # Update every 4 batches
236:             "max_input_length": 64,
237:             "checkpoint_dir": str(tmp_path / "checkpoints"),
238:         }
239: 
240:         trainer = SupervisedTrainer(
241:             model=t5_small_model,
242:             train_questions=train_questions,
243:             val_questions=val_questions,
244:             config=config,
245:         )
246: 
247:         # Record initial params
248:         initial_params = {
249:             name: param.clone()
250:             for name, param in t5_small_model.policy_head.named_parameters()
251:         }
252: 
253:         # Run one epoch
254:         trainer.train_epoch()
255: 
256:         # Check that params changed (at least some should update)
257:         any_changed = False
258:         for name, param in t5_small_model.policy_head.named_parameters():
259:             if not torch.equal(initial_params[name], param):
260:                 any_changed = True
261:                 break
262: 
263:         assert any_changed, "Policy head parameters should change after training"
264: 
265: 
266: # ---------------------------------------------------------------------------
267: # Validation Tests
268: # ---------------------------------------------------------------------------
269: 
270: 
271: class TestValidation:
272:     """Tests for SupervisedTrainer.validate."""
273: 
274:     def test_validate_returns_metrics(self, trainer: SupervisedTrainer):
275:         """Validation returns loss and accuracy."""
276:         val_loss, val_acc = trainer.validate()
277: 
278:         assert isinstance(val_loss, float)
279:         assert isinstance(val_acc, float)
280:         assert val_loss > 0
281:         assert 0 <= val_acc <= 1
282: 
283: 
284: # ---------------------------------------------------------------------------
285: # Checkpoint Tests
286: # ---------------------------------------------------------------------------
287: 
288: 
289: class TestCheckpoint:
290:     """Tests for checkpoint save/load functionality."""
291: 
292:     def test_checkpoint_save_load(self, trainer: SupervisedTrainer):
293:         """Save then load produces identical model outputs."""
294:         trainer.model.eval()
295: 
296:         # Get output before save
297:         q = _make_question("test_checkpoint")
298:         input_ids, attention_mask, _ = trainer.prepare_batch([q])
299:         with torch.no_grad():
300:             logits_before, preds_before = trainer.model.predict_answer(
301:                 input_ids, attention_mask
302:             )
303: 
304:         # Save checkpoint
305:         save_path = trainer.save_checkpoint(is_best=True)
306:         assert save_path.exists()
307:         assert (save_path / "policy_head.pt").exists()
308:         assert (save_path / "training_state.pt").exists()
309: 
310:         # Load checkpoint
311:         trainer.model.load(str(save_path))
312: 
313:         # Get output after load
314:         with torch.no_grad():
315:             logits_after, preds_after = trainer.model.predict_answer(
316:                 input_ids, attention_mask
317:             )
318: 
319:         assert torch.allclose(logits_before, logits_after, atol=1e-5)
320: 
321:     def test_best_model_selection(
322:         self,
323:         t5_small_model: T5PolicyModel,
324:         train_questions: list[MCQuestion],
325:         val_questions: list[MCQuestion],
326:         tmp_path,
327:     ):
328:         """Best model saved by validation accuracy (best_model/ dir exists)."""
329:         config = {
330:             "supervised_lr": 1e-3,
331:             "supervised_epochs": 2,
332:             "supervised_batch_size": 4,
333:             "supervised_grad_accum_steps": 1,
334:             "max_input_length": 64,
335:             "checkpoint_dir": str(tmp_path / "checkpoints"),
336:         }
337: 
338:         trainer = SupervisedTrainer(
339:             model=t5_small_model,
340:             train_questions=train_questions,
341:             val_questions=val_questions,
342:             config=config,
343:         )
344: 
345:         result = trainer.train()
346: 
347:         # Best model directory should exist
348:         best_model_path = trainer.checkpoint_dir / "best_model"
349:         assert best_model_path.exists(), "best_model/ directory should exist"
350:         assert (best_model_path / "policy_head.pt").exists()
351:         assert result["best_val_acc"] >= 0
352: 
353:     def test_history_saved(self, trainer: SupervisedTrainer):
354:         """Training history saved to history.json with correct structure."""
355:         # Run a quick training
356:         trainer.config["supervised_epochs"] = 1
357:         trainer.epochs = 1
358:         trainer.train()
359: 
360:         history_path = trainer.checkpoint_dir / "history.json"
361:         assert history_path.exists()
362: 
363:         with open(history_path) as f:
364:             history = json.load(f)
365: 
366:         assert "train" in history
367:         assert "val" in history
368:         assert "config" in history
369:         assert len(history["train"]) >= 1
370:         assert "loss" in history["train"][0]
371:         assert "accuracy" in history["train"][0]
````

## File: training/__init__.py
````python
1: """
2: Training Package
3: 
4: Supervised warm-start and PPO fine-tuning for T5 policy models.
5: """
````

## File: training/train_supervised_t5.py
````python
  1: """
  2: Supervised warm-start training for T5PolicyModel.
  3: 
  4: Trains answer selection on complete questions using cross-entropy loss. All
  5: clues are shown at once (not incremental), providing a strong initialization
  6: before PPO fine-tuning on partial observations.
  7: 
  8: The training loop uses gradient accumulation (default 4 steps, effective
  9: batch = 32) for stable training without exceeding GPU memory. Best model
 10: is saved by validation accuracy to checkpoints/supervised/best_model/.
 11: 
 12: Ported from qanta-buzzer reference implementation (train_supervised.py)
 13: with these changes:
 14:     - Accepts list of MCQuestion objects instead of QuizBowlDataset class
 15:     - Config dict interface instead of qanta-buzzer's Config class
 16:     - Direct text formatting from MCQuestion (no QuizBowlEnvironment needed)
 17:     - NumPy-style docstrings added throughout
 18: 
 19: Usage
 20: -----
 21: From Python::
 22: 
 23:     from training.train_supervised_t5 import SupervisedTrainer, run_supervised_training
 24:     from models.t5_policy import T5PolicyModel
 25:     from qb_data.mc_builder import MCQuestion
 26: 
 27:     model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
 28:     trainer = SupervisedTrainer(model, train_qs, val_qs, config)
 29:     trainer.train()
 30: 
 31: From command line::
 32: 
 33:     python -m training.train_supervised_t5 --config configs/t5_policy.yaml
 34: """
 35: 
 36: from __future__ import annotations
 37: 
 38: import json
 39: import random
 40: from pathlib import Path
 41: from typing import Any, Dict, List, Optional, Tuple
 42: 
 43: import numpy as np
 44: import torch
 45: import torch.nn as nn
 46: import torch.optim as optim
 47: 
 48: from models.t5_policy import T5PolicyModel
 49: from qb_data.mc_builder import MCQuestion
 50: 
 51: 
 52: def format_question_text(question: MCQuestion) -> str:
 53:     """Format a complete question as text for supervised training.
 54: 
 55:     Shows ALL clues (complete question) since supervised training is the
 56:     easier task of answer selection on full information. PPO later trains
 57:     on incremental clues.
 58: 
 59:     Parameters
 60:     ----------
 61:     question : MCQuestion
 62:         Question with tokens, options, and gold_index.
 63: 
 64:     Returns
 65:     -------
 66:     str
 67:         Formatted text: ``"CLUES: <all tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
 68:     """
 69:     clues_text = " ".join(question.tokens)
 70:     choices_parts = [f"({i + 1}) {opt}" for i, opt in enumerate(question.options)]
 71:     choices_text = " ".join(choices_parts)
 72:     return f"CLUES: {clues_text} | CHOICES: {choices_text}"
 73: 
 74: 
 75: class SupervisedTrainer:
 76:     """Trainer for supervised warm-start of T5PolicyModel.
 77: 
 78:     Trains the answer head using cross-entropy loss on complete questions
 79:     (all clues shown at once). Uses gradient accumulation for stable training
 80:     with large effective batch sizes without exceeding GPU memory.
 81: 
 82:     The training loop:
 83:     1. Shuffles training data each epoch
 84:     2. Iterates over mini-batches
 85:     3. Computes cross-entropy loss on answer logits
 86:     4. Accumulates gradients for ``grad_accum_steps`` batches
 87:     5. Clips gradients and updates optimizer
 88:     6. Validates after each epoch
 89:     7. Saves best model by validation accuracy
 90: 
 91:     Parameters
 92:     ----------
 93:     model : T5PolicyModel
 94:         Model to train. Must have ``predict_answer`` and ``tokenizer``.
 95:     train_questions : list[MCQuestion]
 96:         Training set questions.
 97:     val_questions : list[MCQuestion]
 98:         Validation set questions.
 99:     config : dict[str, Any]
100:         Configuration dictionary with keys:
101: 
102:         - ``supervised_lr`` (float): Learning rate. Default 3e-4.
103:         - ``supervised_epochs`` (int): Number of epochs. Default 10.
104:         - ``supervised_batch_size`` (int): Batch size. Default 8.
105:         - ``supervised_grad_accum_steps`` (int): Gradient accumulation. Default 4.
106:         - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
107:         - ``max_input_length`` (int): Max token length. Default 512.
108:         - ``max_grad_norm`` (float): Gradient clip norm. Default 1.0.
109:         - ``weight_decay`` (float): AdamW weight decay. Default 0.01.
110: 
111:     Attributes
112:     ----------
113:     model : T5PolicyModel
114:         The model being trained.
115:     optimizer : torch.optim.AdamW
116:         Optimizer with weight decay.
117:     criterion : nn.CrossEntropyLoss
118:         Loss function for answer classification.
119:     best_val_acc : float
120:         Best validation accuracy seen so far.
121:     train_history : list[dict]
122:         Per-epoch training metrics.
123:     val_history : list[dict]
124:         Per-epoch validation metrics.
125:     checkpoint_dir : Path
126:         Directory for saving checkpoints.
127:     """
128: 
129:     def __init__(
130:         self,
131:         model: T5PolicyModel,
132:         train_questions: List[MCQuestion],
133:         val_questions: List[MCQuestion],
134:         config: Dict[str, Any],
135:     ) -> None:
136:         self.model = model
137:         self.train_questions = list(train_questions)
138:         self.val_questions = list(val_questions)
139:         self.config = config
140: 
141:         self.device = model.device
142: 
143:         # Hyperparameters with defaults
144:         self.lr = float(config.get("supervised_lr", 3e-4))
145:         self.epochs = int(config.get("supervised_epochs", 10))
146:         self.batch_size = int(config.get("supervised_batch_size", 8))
147:         self.grad_accum_steps = int(config.get("supervised_grad_accum_steps", 4))
148:         self.max_input_length = int(config.get("max_input_length", 512))
149:         self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
150:         self.weight_decay = float(config.get("weight_decay", 0.01))
151: 
152:         # Optimizer
153:         self.optimizer = optim.AdamW(
154:             model.parameters(), lr=self.lr, weight_decay=self.weight_decay
155:         )
156: 
157:         # Loss function
158:         self.criterion = nn.CrossEntropyLoss()
159: 
160:         # Training state
161:         self.current_epoch = 0
162:         self.best_val_acc = 0.0
163:         self.train_history: List[Dict[str, Any]] = []
164:         self.val_history: List[Dict[str, Any]] = []
165: 
166:         # Checkpoint directory
167:         self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")) / "supervised"
168:         self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
169: 
170:     def prepare_batch(
171:         self, questions: List[MCQuestion]
172:     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
173:         """Format a batch of complete questions as tokenized tensors.
174: 
175:         Each question is formatted with ALL clues visible (supervised training
176:         shows complete information). Text is tokenized using the model's
177:         T5TokenizerFast.
178: 
179:         Parameters
180:         ----------
181:         questions : list[MCQuestion]
182:             Batch of questions to format.
183: 
184:         Returns
185:         -------
186:         input_ids : torch.Tensor
187:             Token IDs of shape ``[batch_size, seq_len]``, on device.
188:         attention_mask : torch.Tensor
189:             Attention mask of shape ``[batch_size, seq_len]``, on device.
190:         labels : torch.Tensor
191:             Gold answer indices of shape ``[batch_size]``, on device.
192:         """
193:         texts = [format_question_text(q) for q in questions]
194:         labels = [q.gold_index for q in questions]
195: 
196:         # Tokenize
197:         inputs = self.model.tokenizer(
198:             texts,
199:             return_tensors="pt",
200:             padding=True,
201:             truncation=True,
202:             max_length=self.max_input_length,
203:         )
204: 
205:         input_ids = inputs["input_ids"].to(self.device)
206:         attention_mask = inputs["attention_mask"].to(self.device)
207:         labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
208: 
209:         return input_ids, attention_mask, labels_tensor
210: 
211:     def train_epoch(self) -> Tuple[float, float]:
212:         """Train for one epoch with gradient accumulation.
213: 
214:         Shuffles training data, iterates over mini-batches, and updates
215:         the optimizer every ``grad_accum_steps`` batches. Gradients are
216:         clipped to ``max_grad_norm`` before each optimizer step.
217: 
218:         Returns
219:         -------
220:         epoch_loss : float
221:             Average loss over all batches in the epoch.
222:         epoch_acc : float
223:             Average accuracy over all batches in the epoch.
224:         """
225:         self.model.train()
226: 
227:         # Shuffle training data
228:         shuffled = self.train_questions[:]
229:         random.shuffle(shuffled)
230: 
231:         total_loss = 0.0
232:         total_correct = 0
233:         total_samples = 0
234:         num_batches = max(1, len(shuffled) // self.batch_size)
235: 
236:         # Zero gradients at start
237:         self.optimizer.zero_grad()
238: 
239:         for batch_idx in range(num_batches):
240:             # Get batch
241:             start = batch_idx * self.batch_size
242:             end = min(start + self.batch_size, len(shuffled))
243:             batch_questions = shuffled[start:end]
244: 
245:             if not batch_questions:
246:                 continue
247: 
248:             # Prepare batch
249:             input_ids, attention_mask, labels = self.prepare_batch(batch_questions)
250: 
251:             # Forward pass
252:             answer_logits, predictions = self.model.predict_answer(
253:                 input_ids, attention_mask
254:             )
255: 
256:             # Compute loss (scaled by accumulation steps for correct gradient magnitude)
257:             loss = self.criterion(answer_logits, labels)
258:             scaled_loss = loss / self.grad_accum_steps
259:             scaled_loss.backward()
260: 
261:             # Track metrics (use unscaled loss for logging)
262:             total_loss += loss.item()
263:             total_correct += (predictions == labels).sum().item()
264:             total_samples += len(labels)
265: 
266:             # Gradient accumulation: update every N batches
267:             if (batch_idx + 1) % self.grad_accum_steps == 0:
268:                 torch.nn.utils.clip_grad_norm_(
269:                     self.model.parameters(), self.max_grad_norm
270:                 )
271:                 self.optimizer.step()
272:                 self.optimizer.zero_grad()
273: 
274:         # Handle remaining accumulated gradients (if num_batches not divisible by accum_steps)
275:         remaining = num_batches % self.grad_accum_steps
276:         if remaining > 0:
277:             torch.nn.utils.clip_grad_norm_(
278:                 self.model.parameters(), self.max_grad_norm
279:             )
280:             self.optimizer.step()
281:             self.optimizer.zero_grad()
282: 
283:         epoch_loss = total_loss / max(1, num_batches)
284:         epoch_acc = total_correct / max(1, total_samples)
285: 
286:         return epoch_loss, epoch_acc
287: 
288:     def validate(self) -> Tuple[float, float]:
289:         """Validate on the validation set.
290: 
291:         Runs the model in eval mode on all validation questions, computing
292:         accuracy and loss without gradient computation.
293: 
294:         Returns
295:         -------
296:         val_loss : float
297:             Average cross-entropy loss on validation set.
298:         val_acc : float
299:             Accuracy on validation set (fraction correct).
300:         """
301:         self.model.eval()
302: 
303:         total_loss = 0.0
304:         total_correct = 0
305:         total_samples = 0
306:         num_batches = max(1, len(self.val_questions) // self.batch_size)
307: 
308:         with torch.no_grad():
309:             for batch_idx in range(num_batches):
310:                 start = batch_idx * self.batch_size
311:                 end = min(start + self.batch_size, len(self.val_questions))
312:                 batch_questions = self.val_questions[start:end]
313: 
314:                 if not batch_questions:
315:                     continue
316: 
317:                 input_ids, attention_mask, labels = self.prepare_batch(batch_questions)
318:                 answer_logits, predictions = self.model.predict_answer(
319:                     input_ids, attention_mask
320:                 )
321: 
322:                 loss = self.criterion(answer_logits, labels)
323:                 total_loss += loss.item()
324:                 total_correct += (predictions == labels).sum().item()
325:                 total_samples += len(labels)
326: 
327:         val_loss = total_loss / max(1, num_batches)
328:         val_acc = total_correct / max(1, total_samples)
329: 
330:         return val_loss, val_acc
331: 
332:     def train(self) -> Dict[str, Any]:
333:         """Run full supervised training loop.
334: 
335:         Iterates over epochs, training and validating each epoch. Saves the
336:         best model by validation accuracy to ``checkpoint_dir/best_model/``.
337:         Training history is saved to ``checkpoint_dir/history.json``.
338: 
339:         Returns
340:         -------
341:         dict[str, Any]
342:             Training summary with keys: ``best_val_acc``, ``final_train_acc``,
343:             ``final_train_loss``, ``total_epochs``.
344:         """
345:         print(f"Starting supervised training for {self.epochs} epochs")
346:         print(f"  Training samples: {len(self.train_questions)}")
347:         print(f"  Validation samples: {len(self.val_questions)}")
348:         print(f"  Batch size: {self.batch_size}")
349:         print(f"  Gradient accumulation: {self.grad_accum_steps} (effective batch = {self.batch_size * self.grad_accum_steps})")
350:         print(f"  Learning rate: {self.lr}")
351:         print(f"  Device: {self.device}")
352:         print()
353: 
354:         final_train_loss = 0.0
355:         final_train_acc = 0.0
356: 
357:         for epoch in range(self.epochs):
358:             self.current_epoch = epoch
359: 
360:             # Train epoch
361:             train_loss, train_acc = self.train_epoch()
362:             final_train_loss = train_loss
363:             final_train_acc = train_acc
364: 
365:             # Validate
366:             val_loss, val_acc = self.validate()
367: 
368:             # Log results
369:             print(
370:                 f"Epoch {epoch + 1}/{self.epochs} - "
371:                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
372:                 f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
373:             )
374: 
375:             # Save history
376:             self.train_history.append(
377:                 {"epoch": epoch + 1, "loss": train_loss, "accuracy": train_acc}
378:             )
379:             self.val_history.append(
380:                 {"epoch": epoch + 1, "loss": val_loss, "accuracy": val_acc}
381:             )
382: 
383:             # Save best model
384:             if val_acc > self.best_val_acc:
385:                 self.best_val_acc = val_acc
386:                 self.save_checkpoint(is_best=True)
387:                 print(f"  -> New best validation accuracy: {val_acc:.4f}")
388: 
389:         print(f"\nSupervised training completed!")
390:         print(f"  Best validation accuracy: {self.best_val_acc:.4f}")
391: 
392:         # Save training history
393:         self.save_history()
394: 
395:         return {
396:             "best_val_acc": self.best_val_acc,
397:             "final_train_acc": final_train_acc,
398:             "final_train_loss": final_train_loss,
399:             "total_epochs": self.epochs,
400:         }
401: 
402:     def save_checkpoint(self, is_best: bool = False) -> Path:
403:         """Save model checkpoint to disk.
404: 
405:         Saves the model (T5 encoder + policy head) and optimizer state.
406:         Best model is saved to ``checkpoint_dir/best_model/``, epoch
407:         checkpoints to ``checkpoint_dir/epoch_N/``.
408: 
409:         Parameters
410:         ----------
411:         is_best : bool
412:             If True, save to ``best_model/`` directory.
413: 
414:         Returns
415:         -------
416:         Path
417:             Path to the saved checkpoint directory.
418:         """
419:         if is_best:
420:             save_path = self.checkpoint_dir / "best_model"
421:         else:
422:             save_path = self.checkpoint_dir / f"epoch_{self.current_epoch + 1}"
423: 
424:         # Use T5PolicyModel's save() method
425:         self.model.save(str(save_path))
426: 
427:         # Save training state
428:         state = {
429:             "epoch": self.current_epoch + 1,
430:             "best_val_acc": self.best_val_acc,
431:             "optimizer_state_dict": self.optimizer.state_dict(),
432:         }
433:         torch.save(state, save_path / "training_state.pt")
434: 
435:         return save_path
436: 
437:     def save_history(self) -> Path:
438:         """Save training history to JSON.
439: 
440:         Converts numpy types to native Python types for JSON serialization.
441: 
442:         Returns
443:         -------
444:         Path
445:             Path to the saved history file.
446:         """
447:         history = {
448:             "train": _convert_to_native(self.train_history),
449:             "val": _convert_to_native(self.val_history),
450:             "config": {
451:                 "lr": self.lr,
452:                 "epochs": self.epochs,
453:                 "batch_size": self.batch_size,
454:                 "grad_accum_steps": self.grad_accum_steps,
455:             },
456:         }
457: 
458:         history_path = self.checkpoint_dir / "history.json"
459:         with open(history_path, "w") as f:
460:             json.dump(history, f, indent=2)
461: 
462:         print(f"Training history saved to {history_path}")
463:         return history_path
464: 
465: 
466: def run_supervised_training(
467:     config: Dict[str, Any],
468:     train_questions: List[MCQuestion],
469:     val_questions: List[MCQuestion],
470:     test_questions: Optional[List[MCQuestion]] = None,
471: ) -> Tuple[T5PolicyModel, SupervisedTrainer]:
472:     """Run the complete supervised training pipeline.
473: 
474:     Creates a T5PolicyModel, trains it on complete questions, and optionally
475:     evaluates on a test set. This is the main entry point for supervised
476:     warm-start training.
477: 
478:     Parameters
479:     ----------
480:     config : dict[str, Any]
481:         Configuration dictionary. Must include model config keys
482:         (``model_name``, ``device``, ``num_choices``) and supervised
483:         training keys (``supervised_lr``, etc.).
484:     train_questions : list[MCQuestion]
485:         Training set questions.
486:     val_questions : list[MCQuestion]
487:         Validation set questions.
488:     test_questions : list[MCQuestion] or None
489:         Optional test set for final evaluation.
490: 
491:     Returns
492:     -------
493:     model : T5PolicyModel
494:         The trained model (with best weights loaded).
495:     trainer : SupervisedTrainer
496:         The trainer instance with training history.
497:     """
498:     print("=" * 60)
499:     print("SUPERVISED TRAINING PHASE")
500:     print("=" * 60)
501: 
502:     # Initialize model
503:     model_config = {
504:         "model_name": config.get("model_name", "t5-large"),
505:         "device": config.get("device", "cpu"),
506:         "max_input_length": config.get("max_input_length", 512),
507:         "num_choices": config.get("num_choices", 4),
508:     }
509:     model = T5PolicyModel(model_config)
510: 
511:     # Create trainer
512:     trainer = SupervisedTrainer(
513:         model=model,
514:         train_questions=train_questions,
515:         val_questions=val_questions,
516:         config=config,
517:     )
518: 
519:     # Train
520:     summary = trainer.train()
521: 
522:     # Evaluate on test set if provided
523:     if test_questions is not None:
524:         print("\n" + "=" * 60)
525:         print("FINAL EVALUATION ON TEST SET")
526:         print("=" * 60)
527: 
528:         # Load best model
529:         best_model_path = trainer.checkpoint_dir / "best_model"
530:         model.load(str(best_model_path))
531:         model.eval()
532: 
533:         # Evaluate
534:         test_loss, test_acc = _evaluate_on_questions(model, test_questions, trainer)
535:         print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
536: 
537:         # Save test results
538:         test_results = {
539:             "test_loss": test_loss,
540:             "test_accuracy": test_acc,
541:             "training_summary": summary,
542:         }
543:         results_path = trainer.checkpoint_dir / "test_results.json"
544:         with open(results_path, "w") as f:
545:             json.dump(_convert_to_native(test_results), f, indent=2)
546:         print(f"Test results saved to {results_path}")
547: 
548:     return model, trainer
549: 
550: 
551: def _evaluate_on_questions(
552:     model: T5PolicyModel,
553:     questions: List[MCQuestion],
554:     trainer: SupervisedTrainer,
555: ) -> Tuple[float, float]:
556:     """Evaluate model on a set of questions.
557: 
558:     Parameters
559:     ----------
560:     model : T5PolicyModel
561:         Model to evaluate.
562:     questions : list[MCQuestion]
563:         Questions to evaluate on.
564:     trainer : SupervisedTrainer
565:         Trainer instance (for batch preparation).
566: 
567:     Returns
568:     -------
569:     avg_loss : float
570:         Average cross-entropy loss.
571:     accuracy : float
572:         Fraction of correctly predicted answers.
573:     """
574:     model.eval()
575:     total_loss = 0.0
576:     total_correct = 0
577:     total_samples = 0
578:     batch_size = trainer.batch_size
579:     num_batches = max(1, len(questions) // batch_size)
580:     criterion = nn.CrossEntropyLoss()
581: 
582:     with torch.no_grad():
583:         for batch_idx in range(num_batches):
584:             start = batch_idx * batch_size
585:             end = min(start + batch_size, len(questions))
586:             batch_questions = questions[start:end]
587: 
588:             if not batch_questions:
589:                 continue
590: 
591:             input_ids, attention_mask, labels = trainer.prepare_batch(batch_questions)
592:             answer_logits, predictions = model.predict_answer(input_ids, attention_mask)
593: 
594:             loss = criterion(answer_logits, labels)
595:             total_loss += loss.item()
596:             total_correct += (predictions == labels).sum().item()
597:             total_samples += len(labels)
598: 
599:     return total_loss / max(1, num_batches), total_correct / max(1, total_samples)
600: 
601: 
602: def _convert_to_native(obj: Any) -> Any:
603:     """Convert numpy types to Python native types for JSON serialization.
604: 
605:     Parameters
606:     ----------
607:     obj : Any
608:         Object to convert. Handles dicts, lists, numpy scalars and arrays.
609: 
610:     Returns
611:     -------
612:     Any
613:         Object with all numpy types converted to native Python types.
614:     """
615:     if isinstance(obj, dict):
616:         return {k: _convert_to_native(v) for k, v in obj.items()}
617:     elif isinstance(obj, list):
618:         return [_convert_to_native(v) for v in obj]
619:     elif isinstance(obj, (np.integer,)):
620:         return int(obj)
621:     elif isinstance(obj, (np.floating,)):
622:         return float(obj)
623:     elif isinstance(obj, np.ndarray):
624:         return _convert_to_native(obj.tolist())
625:     else:
626:         return obj
````

## File: CLAUDE.md
````markdown
 1: # CLAUDE.md
 2: 
 3: See **AGENTS.md** for the full repo contract: setup, architecture, testing, smoke pipeline, and configuration.
 4: 
 5: ## Claude-specific notes
 6: 
 7: - `.planning/` is durable project memory; respect STATE.md decisions.
 8: - Prefer narrow verification over broad cargo-cult test runs.
 9: - Do not add dependencies unless required.
10: - Seeds: use 1, 2, 3 for multi-seed runs.
11: - NumPy/PyTorch vectorized operations over loops in ML code.
````

## File: agents/__init__.py
````python
 1: from agents.threshold_buzzer import (
 2:     ThresholdBuzzer,
 3:     AlwaysBuzzFinalBuzzer,
 4:     EpisodeResult,
 5:     sweep_thresholds,
 6:     result_to_dict,
 7: )
 8: from agents.bayesian_buzzer import (
 9:     SoftmaxProfileBuzzer,
10:     SequentialBayesBuzzer,
11:     SoftmaxEpisodeResult,
12:     sweep_sequential_thresholds,
13: )
14: 
15: # Lazy import: PPOBuzzer requires stable_baselines3 which may not be installed
16: # in all environments (e.g., baseline-only runs). Import on demand.
17: 
18: 
19: def __getattr__(name: str):
20:     if name in ("PPOBuzzer", "PPOEpisodeTrace"):
21:         from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
22:         return {"PPOBuzzer": PPOBuzzer, "PPOEpisodeTrace": PPOEpisodeTrace}[name]
23:     raise AttributeError(f"module 'agents' has no attribute {name!r}")
24: 
25: 
26: __all__ = [
27:     "ThresholdBuzzer",
28:     "AlwaysBuzzFinalBuzzer",
29:     "SoftmaxProfileBuzzer",
30:     "SequentialBayesBuzzer",
31:     "PPOBuzzer",
32:     "EpisodeResult",
33:     "SoftmaxEpisodeResult",
34:     "PPOEpisodeTrace",
35:     "sweep_thresholds",
36:     "sweep_sequential_thresholds",
37:     "result_to_dict",
38: ]
````

## File: evaluation/__init__.py
````python
 1: """
 2: Evaluation Package
 3: 
 4: Metrics computation for quiz bowl buzzer agents, including S_q scoring,
 5: calibration analysis (ECE, Brier score), and buzz timing statistics.
 6: 
 7: Ported from qb-rl reference implementation with adaptations for
 8: qanta-buzzer's EpisodeResult / SoftmaxEpisodeResult / PPOEpisodeTrace
 9: dataclass structures.
10: """
11: 
12: from evaluation.metrics import (
13:     calibration_at_buzz,
14:     expected_calibration_error,
15:     expected_wins_score,
16:     per_category_accuracy,
17:     summarize_buzz_metrics,
18:     system_score,
19: )
20: 
21: __all__ = [
22:     "system_score",
23:     "expected_wins_score",
24:     "summarize_buzz_metrics",
25:     "calibration_at_buzz",
26:     "expected_calibration_error",
27:     "per_category_accuracy",
28: ]
````

## File: evaluation/metrics.py
````python
  1: """
  2: Evaluation Metrics for Quiz Bowl Buzzer Agents
  3: 
  4: Computes buzz accuracy, S_q scoring, calibration metrics (ECE, Brier score),
  5: and buzz timing statistics from episode trace data.
  6: 
  7: Ported from qb-rl reference implementation (evaluation/metrics.py).
  8: Accepts both raw dicts and dataclass instances (EpisodeResult,
  9: SoftmaxEpisodeResult, PPOEpisodeTrace) via the _to_dict helper.
 10: 
 11: Functions
 12: ---------
 13: system_score(c_trace, g_trace)
 14:     Compute S_q = sum_t b_t * g_t where b_t = c_t * prod_{i<t} (1 - c_i).
 15: expected_calibration_error(confidences, outcomes, n_bins)
 16:     Binned ECE over confidence-outcome pairs.
 17: brier_score(confidences, outcomes)
 18:     Mean squared error between confidence and binary outcome.
 19: summarize_buzz_metrics(results)
 20:     Aggregate accuracy, buzz step, S_q, and reward across episodes.
 21: calibration_at_buzz(results)
 22:     Extract buzz-time top_p confidence and compute ECE + Brier score.
 23: expected_wins_score(c_trace, g_trace, opponent_survival_trace, ...)
 24:     Offline Expected Wins scoring over an episode.
 25: """
 26: 
 27: from __future__ import annotations
 28: 
 29: from dataclasses import asdict, is_dataclass
 30: from typing import Any
 31: 
 32: import numpy as np
 33: 
 34: 
 35: def _to_dict(item: Any) -> dict[str, Any]:
 36:     """Convert dataclass or object to dict for uniform access.
 37: 
 38:     Parameters
 39:     ----------
 40:     item : Any
 41:         A dict, dataclass instance, or object with __dict__.
 42: 
 43:     Returns
 44:     -------
 45:     dict[str, Any]
 46:         Dictionary representation of the item.
 47:     """
 48:     if isinstance(item, dict):
 49:         return item
 50:     if is_dataclass(item):
 51:         return asdict(item)
 52:     return item.__dict__
 53: 
 54: 
 55: def system_score(c_trace: list[float], g_trace: list[float]) -> float:
 56:     """Compute S_q scoring metric for a single episode.
 57: 
 58:     S_q = sum_t b_t * g_t, where b_t = c_t * prod_{i<t} (1 - c_i).
 59:     This is the expected correctness under the agent's buzz policy,
 60:     accounting for the survival probability of not having buzzed earlier.
 61: 
 62:     Parameters
 63:     ----------
 64:     c_trace : list[float]
 65:         Buzz probability at each time step (confidence proxy).
 66:     g_trace : list[float]
 67:         Correctness indicator at each time step (1.0 if top answer is
 68:         correct, 0.0 otherwise).
 69: 
 70:     Returns
 71:     -------
 72:     float
 73:         S_q score for the episode, in [0, 1].
 74:     """
 75:     c = np.array(c_trace, dtype=np.float64)
 76:     g = np.array(g_trace, dtype=np.float64)
 77:     if len(c) == 0:
 78:         return 0.0
 79:     b = np.zeros_like(c)
 80:     survival = 1.0
 81:     for t in range(len(c)):
 82:         b[t] = c[t] * survival
 83:         survival *= (1.0 - c[t])
 84:     return float(np.sum(b * g))
 85: 
 86: 
 87: def expected_wins_score(
 88:     c_trace: list[float],
 89:     g_trace: list[float],
 90:     opponent_survival_trace: list[float],
 91:     reward_correct: float = 10.0,
 92:     reward_incorrect: float = -5.0,
 93:     opponent_expected_value: float = 0.0,
 94: ) -> float:
 95:     """Compute offline Expected Wins score for a single episode.
 96: 
 97:     Uses the continuous V_self formulation::
 98: 
 99:         V_self_t = g_t * reward_correct + (1 - g_t) * reward_incorrect
100: 
101:     NOT a binary branch on ``g_t``.
102: 
103:     The full formula is::
104: 
105:         EW = sum_t  b_t * [S_t * V_self_t + (1 - S_t) * V_opp]
106: 
107:     where ``b_t = c_t * prod_{i<t}(1 - c_i)`` is the agent's buzz
108:     probability mass at step *t*, and ``S_t`` is opponent survival.
109: 
110:     Parameters
111:     ----------
112:     c_trace : list[float]
113:         Per-step buzz probability from the agent.
114:     g_trace : list[float]
115:         Per-step correctness probability (P(gold) / P(buzz) for PPO,
116:         binary 0/1 for baseline agents).
117:     opponent_survival_trace : list[float]
118:         Per-step P(opponent has not buzzed before step t).
119:     reward_correct : float
120:         Points for buzzing correctly before the opponent.
121:     reward_incorrect : float
122:         Points for buzzing incorrectly before the opponent.
123:     opponent_expected_value : float
124:         Expected score when the opponent buzzes first.
125: 
126:     Returns
127:     -------
128:     float
129:         Expected Wins score for the episode.
130:     """
131:     c = np.array(c_trace, dtype=np.float64)
132:     g = np.array(g_trace, dtype=np.float64)
133:     s = np.array(opponent_survival_trace, dtype=np.float64)
134:     if len(c) == 0:
135:         return 0.0
136:     n = min(len(c), len(g), len(s))
137:     c, g, s = c[:n], g[:n], s[:n]
138: 
139:     b = np.zeros(n, dtype=np.float64)
140:     survival = 1.0
141:     for t in range(n):
142:         b[t] = c[t] * survival
143:         survival *= 1.0 - c[t]
144: 
145:     v_self = g * reward_correct + (1.0 - g) * reward_incorrect
146:     v = s * v_self + (1.0 - s) * opponent_expected_value
147:     return float(np.sum(b * v))
148: 
149: 
150: def expected_calibration_error(
151:     confidences: list[float], outcomes: list[int], n_bins: int = 10
152: ) -> float:
153:     """Compute Expected Calibration Error (ECE) with uniform binning.
154: 
155:     ECE measures the gap between predicted confidence and actual accuracy
156:     across confidence bins. Lower ECE indicates better-calibrated predictions.
157: 
158:     Parameters
159:     ----------
160:     confidences : list[float]
161:         Predicted confidence values in [0, 1].
162:     outcomes : list[int]
163:         Binary outcomes (1 = correct, 0 = incorrect).
164:     n_bins : int
165:         Number of uniform bins for confidence bucketing.
166: 
167:     Returns
168:     -------
169:     float
170:         Expected calibration error in [0, 1]. Returns 0.0 if no data.
171:     """
172:     if not confidences:
173:         return 0.0
174:     conf = np.array(confidences, dtype=np.float64)
175:     y = np.array(outcomes, dtype=np.float64)
176:     bins = np.linspace(0.0, 1.0, n_bins + 1)
177:     ece = 0.0
178:     for i in range(n_bins):
179:         lo, hi = bins[i], bins[i + 1]
180:         mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
181:         if not mask.any():
182:             continue
183:         bin_acc = y[mask].mean()
184:         bin_conf = conf[mask].mean()
185:         ece += (mask.mean()) * abs(bin_acc - bin_conf)
186:     return float(ece)
187: 
188: 
189: def brier_score(confidences: list[float], outcomes: list[int]) -> float:
190:     """Compute Brier score (mean squared calibration error).
191: 
192:     Brier score measures the mean squared difference between predicted
193:     confidence and binary outcome. Lower is better; 0 is perfect.
194: 
195:     Parameters
196:     ----------
197:     confidences : list[float]
198:         Predicted confidence values in [0, 1].
199:     outcomes : list[int]
200:         Binary outcomes (1 = correct, 0 = incorrect).
201: 
202:     Returns
203:     -------
204:     float
205:         Brier score in [0, 1]. Returns 0.0 if no data.
206:     """
207:     if not confidences:
208:         return 0.0
209:     conf = np.array(confidences, dtype=np.float64)
210:     y = np.array(outcomes, dtype=np.float64)
211:     return float(np.mean((conf - y) ** 2))
212: 
213: 
214: def summarize_buzz_metrics(results: list[Any]) -> dict[str, float]:
215:     """Aggregate buzz metrics across a list of episode results.
216: 
217:     Computes accuracy, mean buzz step, mean S_q score, and mean reward
218:     from episode trace data. Accepts dicts or dataclass instances.
219: 
220:     Parameters
221:     ----------
222:     results : list[Any]
223:         List of episode results (dicts, EpisodeResult, SoftmaxEpisodeResult,
224:         or PPOEpisodeTrace instances). Each must have: correct, buzz_step,
225:         c_trace, g_trace. Optionally: reward_like or episode_reward.
226: 
227:     Returns
228:     -------
229:     dict[str, float]
230:         Summary metrics: n, buzz_accuracy, mean_buzz_step, mean_sq,
231:         mean_reward_like.
232:     """
233:     rows = [_to_dict(r) for r in results]
234:     if not rows:
235:         return {
236:             "n": 0.0,
237:             "buzz_accuracy": 0.0,
238:             "mean_buzz_step": 0.0,
239:             "mean_sq": 0.0,
240:             "mean_reward_like": 0.0,
241:         }
242: 
243:     correct = np.array(
244:         [1 if bool(r.get("correct", False)) else 0 for r in rows],
245:         dtype=np.float64,
246:     )
247:     buzz_steps = np.array(
248:         [int(r.get("buzz_step", 0)) for r in rows], dtype=np.float64
249:     )
250:     sq_scores = np.array(
251:         [
252:             system_score(
253:                 list(r.get("c_trace", [])),
254:                 list(r.get("g_trace", [])),
255:             )
256:             for r in rows
257:         ],
258:         dtype=np.float64,
259:     )
260:     reward_like = np.array(
261:         [
262:             float(r.get("reward_like", r.get("episode_reward", 0.0)))
263:             for r in rows
264:         ],
265:         dtype=np.float64,
266:     )
267: 
268:     return {
269:         "n": float(len(rows)),
270:         "buzz_accuracy": float(correct.mean()),
271:         "mean_buzz_step": float(buzz_steps.mean()),
272:         "mean_sq": float(sq_scores.mean()),
273:         "mean_reward_like": float(reward_like.mean()),
274:     }
275: 
276: 
277: def per_category_accuracy(
278:     results: list[Any],
279:     questions: list[Any],
280: ) -> dict[str, dict[str, float]]:
281:     """Compute accuracy and S_q metrics grouped by question category.
282: 
283:     Joins results with questions to extract category field, then groups
284:     and computes summarize_buzz_metrics per category.
285: 
286:     Parameters
287:     ----------
288:     results : list[Any]
289:         Episode results from agent evaluation (dicts or dataclasses).
290:         Must have qid field for joining.
291:     questions : list[Any]
292:         Original questions with category field (MCQuestion or similar).
293: 
294:     Returns
295:     -------
296:     dict[str, dict[str, float]]
297:         Mapping from category name to metrics dict with keys:
298:         n, buzz_accuracy, mean_buzz_step, mean_sq, mean_reward_like.
299:     """
300:     from collections import defaultdict
301: 
302:     # Build qid -> category lookup, default to "unknown" for missing
303:     qid_to_category: dict[str, str] = {}
304:     for q in questions:
305:         q_dict = _to_dict(q)
306:         cat = q_dict.get("category", "") or ""
307:         qid = q_dict.get("qid", "")
308:         qid_to_category[qid] = cat if cat else "unknown"
309: 
310:     # Group results by category
311:     by_category: dict[str, list[Any]] = defaultdict(list)
312:     for r in results:
313:         r_dict = _to_dict(r)
314:         qid = r_dict.get("qid", "")
315:         category = qid_to_category.get(qid, "unknown")
316:         by_category[category].append(r)
317: 
318:     # Compute metrics per category
319:     return {
320:         cat: summarize_buzz_metrics(rows)
321:         for cat, rows in sorted(by_category.items())
322:     }
323: 
324: 
325: def calibration_at_buzz(results: list[Any]) -> dict[str, float]:
326:     """Compute calibration metrics at the buzz decision point.
327: 
328:     Uses the belief model's top-answer probability (``top_p_trace``) at
329:     buzz time as the confidence proxy.  This measures whether the belief
330:     distribution is well-calibrated: when the model assigns 0.8
331:     probability to its top answer, that answer should be correct ~80% of
332:     the time.
333: 
334:     Falls back to ``c_trace`` (sigmoid confidence) when ``top_p_trace``
335:     is unavailable (e.g. PPO episode traces that lack per-step belief
336:     breakdowns).
337: 
338:     Parameters
339:     ----------
340:     results : list[Any]
341:         List of episode results (dicts or dataclass instances). Each must
342:         have: buzz_step, correct, and at least one of top_p_trace or
343:         c_trace.
344: 
345:     Returns
346:     -------
347:     dict[str, float]
348:         Calibration metrics: ece, brier, n_calibration.
349:     """
350:     rows = [_to_dict(r) for r in results]
351:     confidences: list[float] = []
352:     outcomes: list[int] = []
353:     for row in rows:
354:         top_p_trace = list(row.get("top_p_trace", []))
355:         c_trace = list(row.get("c_trace", []))
356:         conf_trace = top_p_trace if top_p_trace else c_trace
357:         if not conf_trace:
358:             continue
359:         buzz_step = int(row.get("buzz_step", max(0, len(conf_trace) - 1)))
360:         idx = min(max(0, buzz_step), len(conf_trace) - 1)
361:         confidences.append(float(conf_trace[idx]))
362:         outcomes.append(1 if bool(row.get("correct", False)) else 0)
363: 
364:     return {
365:         "ece": expected_calibration_error(confidences, outcomes),
366:         "brier": brier_score(confidences, outcomes),
367:         "n_calibration": float(len(confidences)),
368:     }
````

## File: models/features.py
````python
  1: """
  2: Belief Feature Extraction
  3: 
  4: Extracts derived features from belief probability distributions for use as
  5: policy observations. Given a belief vector of K probabilities (one per answer
  6: option), produces a (K + 6)-dimensional feature vector containing:
  7: 
  8:     belief[0..K-1]   raw belief probabilities
  9:     top_p             max belief probability
 10:     margin            gap between top two probabilities
 11:     entropy           Shannon entropy of the distribution
 12:     stability         L1 distance from previous belief (0 if first step)
 13:     progress          fraction of total clue steps elapsed
 14:     clue_idx_norm     normalized clue index (0 to 1 over steps)
 15: 
 16: Ported from qb-rl reference implementation (models/features.py).
 17: """
 18: 
 19: from __future__ import annotations
 20: 
 21: import numpy as np
 22: 
 23: 
 24: def entropy_of_distribution(prob: np.ndarray) -> float:
 25:     """Compute Shannon entropy of a probability distribution.
 26: 
 27:     Uses clipping for numerical stability to avoid log(0).
 28: 
 29:     Parameters
 30:     ----------
 31:     prob : np.ndarray
 32:         1D probability vector. Values should sum to ~1.0.
 33: 
 34:     Returns
 35:     -------
 36:     float
 37:         Shannon entropy H(p) = -sum(p * log(p)), non-negative.
 38: 
 39:     Examples
 40:     --------
 41:     >>> import numpy as np
 42:     >>> uniform = np.array([0.25, 0.25, 0.25, 0.25])
 43:     >>> abs(entropy_of_distribution(uniform) - 1.3863) < 0.001
 44:     True
 45:     """
 46:     clipped = np.clip(prob, 1e-12, 1.0)
 47:     return float(-(clipped * np.log(clipped)).sum())
 48: 
 49: 
 50: def extract_belief_features(
 51:     belief: np.ndarray,
 52:     prev_belief: np.ndarray | None,
 53:     step_idx: int,
 54:     total_steps: int,
 55: ) -> np.ndarray:
 56:     """Extract derived features from a belief probability vector.
 57: 
 58:     Concatenates the raw belief with 6 derived scalar features to produce
 59:     a fixed-size observation vector for the RL policy.
 60: 
 61:     Parameters
 62:     ----------
 63:     belief : np.ndarray
 64:         1D probability vector of shape (K,) over answer options.
 65:     prev_belief : np.ndarray or None
 66:         Previous step's belief vector, same shape as ``belief``.
 67:         Pass None on the first step (stability will be 0.0).
 68:     step_idx : int
 69:         Current clue step index (0-based).
 70:     total_steps : int
 71:         Total number of clue steps in the episode.
 72: 
 73:     Returns
 74:     -------
 75:     np.ndarray
 76:         Feature vector of shape (K + 6,) with dtype float32.
 77:         Layout: [belief..., top_p, margin, entropy, stability, progress, clue_idx_norm].
 78: 
 79:     Raises
 80:     ------
 81:     ValueError
 82:         If ``belief`` is not a 1D array.
 83: 
 84:     Examples
 85:     --------
 86:     >>> import numpy as np
 87:     >>> belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
 88:     >>> feats = extract_belief_features(belief, None, 2, 6)
 89:     >>> feats.shape
 90:     (10,)
 91:     >>> feats.dtype
 92:     dtype('float32')
 93:     """
 94:     belief = np.asarray(belief, dtype=np.float32)
 95:     if belief.ndim != 1:
 96:         raise ValueError("belief must be a 1D probability vector")
 97: 
 98:     top_p = float(np.max(belief))
 99:     sorted_probs = np.sort(belief)[::-1]
100:     second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
101:     margin = top_p - second
102:     ent = entropy_of_distribution(belief)
103:     stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0
104:     progress = float(step_idx / max(1, total_steps))
105:     clue_idx_norm = float(step_idx / max(1, total_steps - 1))
106: 
107:     extras = np.array([top_p, margin, ent, stability, progress, clue_idx_norm], dtype=np.float32)
108:     return np.concatenate([belief, extras]).astype(np.float32)
109: 
110: 
111: def extract_padded_belief_features(
112:     belief: np.ndarray,
113:     prev_belief: np.ndarray | None,
114:     step_idx: int,
115:     total_steps: int,
116:     max_K: int,
117: ) -> np.ndarray:
118:     """Extract belief features padded to a fixed ``max_K`` size.
119: 
120:     Identical to :func:`extract_belief_features` except the belief
121:     segment is zero-padded (or truncated) to exactly ``max_K`` elements,
122:     producing a ``(max_K + 6)``-dimensional vector regardless of the
123:     actual number of answer options.
124: 
125:     Parameters
126:     ----------
127:     belief : np.ndarray
128:         1D probability vector of shape (K_actual,).
129:     prev_belief : np.ndarray or None
130:         Previous belief vector (same shape as *belief*).
131:     step_idx : int
132:         Current clue step index (0-based).
133:     total_steps : int
134:         Total clue steps in the episode.
135:     max_K : int
136:         Target padded length for the belief segment.
137: 
138:     Returns
139:     -------
140:     np.ndarray
141:         Feature vector of shape (max_K + 6,), dtype float32.
142:     """
143:     belief = np.asarray(belief, dtype=np.float32)
144:     if belief.ndim != 1:
145:         raise ValueError("belief must be a 1D probability vector")
146: 
147:     K_actual = len(belief)
148: 
149:     top_p = float(np.max(belief))
150:     sorted_probs = np.sort(belief)[::-1]
151:     second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
152:     margin = top_p - second
153:     ent = entropy_of_distribution(belief)
154:     stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0
155:     progress = float(step_idx / max(1, total_steps))
156:     clue_idx_norm = float(step_idx / max(1, total_steps - 1))
157: 
158:     padded = np.zeros(max_K, dtype=np.float32)
159:     padded[:K_actual] = belief[:max_K]
160:     extras = np.array([top_p, margin, ent, stability, progress, clue_idx_norm], dtype=np.float32)
161:     return np.concatenate([padded, extras]).astype(np.float32)
````

## File: qb_data/data_loader.py
````python
  1: """
  2: Data structures and loaders for quiz bowl questions.
  3: """
  4: 
  5: import csv
  6: import hashlib
  7: from dataclasses import dataclass
  8: from pathlib import Path
  9: from typing import List, Optional, Tuple, Any, Dict
 10: 
 11: from qb_data.text_utils import normalize_answer
 12: 
 13: 
 14: @dataclass
 15: class TossupQuestion:
 16:     """
 17:     A quiz bowl tossup question with incremental clues.
 18: 
 19:     Attributes
 20:     ----------
 21:     qid : str
 22:         Unique question identifier
 23:     question : str
 24:         Full question text (all clues concatenated)
 25:     tokens : List[str]
 26:         Tokenized question split on whitespace
 27:     answer_primary : str
 28:         Primary answer text
 29:     clean_answers : List[str]
 30:         List of acceptable answer variants
 31:     run_indices : List[int]
 32:         Token indices where clues end (for incremental reveal)
 33:     human_buzz_positions : Optional[List[Tuple[int, int]]]
 34:         Human buzzer positions as (position, count) tuples
 35:     category : str
 36:         Question category (e.g., "History", "Literature")
 37:     cumulative_prefixes : List[str]
 38:         Precomputed text prefixes at each run_index
 39:     """
 40:     qid: str
 41:     question: str
 42:     tokens: List[str]
 43:     answer_primary: str
 44:     clean_answers: List[str]
 45:     run_indices: List[int]
 46:     human_buzz_positions: Optional[List[Tuple[int, int]]]
 47:     category: str
 48:     cumulative_prefixes: List[str]
 49: 
 50: 
 51: def _parse_clues_to_tokens(clues: List[str]) -> Tuple[List[str], List[int]]:
 52:     """
 53:     Convert list of clues to tokens and run indices.
 54: 
 55:     Parameters
 56:     ----------
 57:     clues : List[str]
 58:         List of clue strings
 59: 
 60:     Returns
 61:     -------
 62:     Tuple[List[str], List[int]]
 63:         Tokens (words) and indices where each clue ends
 64:     """
 65:     tokens = []
 66:     run_indices = []
 67: 
 68:     for clue in clues:
 69:         clue_tokens = clue.split()
 70:         tokens.extend(clue_tokens)
 71:         if clue_tokens:  # Only add index if clue has tokens
 72:             run_indices.append(len(tokens) - 1)
 73: 
 74:     return tokens, run_indices
 75: 
 76: 
 77: def _generate_qid(text: str) -> str:
 78:     """
 79:     Generate a unique question ID from question text.
 80: 
 81:     Parameters
 82:     ----------
 83:     text : str
 84:         Question text to hash
 85: 
 86:     Returns
 87:     -------
 88:     str
 89:         Unique identifier based on text hash
 90:     """
 91:     hash_obj = hashlib.md5(text.encode('utf-8'))
 92:     return f"qid-{hash_obj.hexdigest()[:12]}"
 93: 
 94: 
 95: def _coerce_human_buzz_positions(value: Any) -> Optional[List[Tuple[int, int]]]:
 96:     """Coerce various metadata formats into ``(position, count)`` tuples."""
 97:     if value is None:
 98:         return None
 99: 
100:     if isinstance(value, list):
101:         result: List[Tuple[int, int]] = []
102:         for item in value:
103:             if isinstance(item, (list, tuple)) and len(item) == 2:
104:                 try:
105:                     result.append((int(item[0]), int(item[1])))
106:                 except (TypeError, ValueError):
107:                     continue
108:             elif isinstance(item, dict):
109:                 pos = item.get("position")
110:                 count = item.get("count", 1)
111:                 if pos is None:
112:                     continue
113:                 try:
114:                     result.append((int(pos), int(count)))
115:                 except (TypeError, ValueError):
116:                     continue
117:         return result or None
118: 
119:     return None
120: 
121: 
122: def _coerce_run_indices(run_indices: Any, token_count: int) -> List[int]:
123:     """Validate and coerce run indices into a sorted unique list."""
124:     clean: List[int] = []
125:     for idx in run_indices or []:
126:         try:
127:             clean.append(int(idx))
128:         except (TypeError, ValueError):
129:             continue
130: 
131:     if not clean:
132:         if token_count <= 0:
133:             raise ValueError("question must contain at least one token")
134:         clean = list(range(token_count))
135: 
136:     clean = sorted(set(clean))
137:     if clean[0] < 0 or clean[-1] > token_count - 1:
138:         raise ValueError(
139:             f"run_indices out of bounds: min={clean[0]} max={clean[-1]} token_count={token_count}"
140:         )
141:     return clean
142: 
143: 
144: def parse_row(row: Dict[str, Any]) -> TossupQuestion:
145:     """Parse a qb-rl/HuggingFace-style row into ``TossupQuestion``."""
146:     question = str(row["question"])
147:     tokens = question.split()
148:     metadata = row.get("metadata", {}) or {}
149:     answer_primary = str(
150:         row.get("answer_primary") or (row.get("clean_answers") or [""])[0]
151:     ).strip()
152:     clean_answers = [str(x) for x in (row.get("clean_answers") or [])]
153:     if not clean_answers and answer_primary:
154:         clean_answers = [answer_primary]
155: 
156:     run_indices = _coerce_run_indices(
157:         row.get("run_indices") or [],
158:         token_count=len(tokens),
159:     )
160: 
161:     normalized_question = " ".join(question.split())
162:     normalized_tokens = " ".join(tokens)
163:     if normalized_tokens != normalized_question:
164:         raise ValueError("tokenization roundtrip mismatch")
165:     if max(run_indices) > len(tokens) - 1:
166:         raise ValueError("run_indices out of bounds")
167: 
168:     cumulative_prefixes = [" ".join(tokens[: idx + 1]) for idx in run_indices]
169:     category = str(metadata.get("category") or row.get("category") or "")
170:     human_buzz_positions = _coerce_human_buzz_positions(
171:         metadata.get("human_buzz_positions") or row.get("human_buzz_positions")
172:     )
173: 
174:     qid_raw = row.get("qid") or row.get("question_id") or row.get("id")
175:     if qid_raw is None:
176:         qid_raw = _generate_qid(question)
177: 
178:     return TossupQuestion(
179:         qid=str(qid_raw),
180:         question=question,
181:         tokens=tokens,
182:         answer_primary=answer_primary,
183:         clean_answers=clean_answers,
184:         run_indices=run_indices,
185:         human_buzz_positions=human_buzz_positions,
186:         category=category,
187:         cumulative_prefixes=cumulative_prefixes,
188:     )
189: 
190: 
191: def load_tossup_questions(
192:     dataset: str,
193:     dataset_config: Optional[str] = None,
194:     split: str = "eval",
195:     limit: Optional[int] = None,
196: ) -> List[TossupQuestion]:
197:     """Load tossup questions from Hugging Face datasets using qb-rl semantics."""
198:     try:
199:         from datasets import load_dataset
200:     except ImportError as exc:
201:         raise ImportError(
202:             "datasets is required for Hugging Face loading. Install it with: pip install datasets"
203:         ) from exc
204: 
205:     if dataset_config:
206:         ds = load_dataset(dataset, dataset_config, split=split)
207:     else:
208:         ds = load_dataset(dataset, split=split)
209: 
210:     if limit is not None:
211:         ds = ds.select(range(min(int(limit), len(ds))))
212: 
213:     return [parse_row(dict(row)) for row in ds]
214: 
215: 
216: def load_tossup_questions_from_config(
217:     config: Dict[str, Any],
218:     smoke: bool = False,
219: ) -> List[TossupQuestion]:
220:     """Load tossups from config, supporting qb-rl and qanta-buzzer keys."""
221:     from qb_data.config import resolve_data_loading_options
222: 
223:     data_opts = resolve_data_loading_options(config, smoke=smoke)
224:     csv_path = data_opts.get("csv_path")
225:     dataset = data_opts.get("dataset")
226:     dataset_config = data_opts.get("dataset_config")
227:     split = data_opts.get("split", "eval")
228:     limit = data_opts.get("max_questions")
229: 
230:     if csv_path and Path(csv_path).exists():
231:         questions = QANTADatasetLoader.load_from_csv(str(csv_path))
232:     elif dataset:
233:         questions = load_tossup_questions(
234:             dataset=str(dataset),
235:             dataset_config=str(dataset_config) if dataset_config else None,
236:             split=str(split),
237:             limit=int(limit) if limit is not None else None,
238:         )
239:     elif csv_path and data_opts.get("use_huggingface"):
240:         from qb_data.huggingface_loader import try_huggingface_fallback
241: 
242:         questions = try_huggingface_fallback(str(csv_path))
243:         if questions is None:
244:             raise FileNotFoundError(
245:                 f"Could not load questions from missing CSV path {csv_path} via Hugging Face fallback"
246:             )
247:     else:
248:         raise FileNotFoundError(
249:             "No valid data source configured. Provide data.csv_path or "
250:             "data.dataset/data.dataset_config for qb-rl compatibility."
251:         )
252: 
253:     if limit is not None:
254:         questions = questions[: int(limit)]
255: 
256:     return questions
257: 
258: 
259: class QANTADatasetLoader:
260:     """
261:     Loader for QANTA-format quiz bowl CSV files.
262: 
263:     The QANTA format has questions with clues separated by ||| delimiters.
264:     """
265: 
266:     @classmethod
267:     def load_from_csv(cls, filepath: str) -> List[TossupQuestion]:
268:         """
269:         Load questions from a QANTA-format CSV file.
270: 
271:         Parameters
272:         ----------
273:         filepath : str
274:             Path to the CSV file
275: 
276:         Returns
277:         -------
278:         List[TossupQuestion]
279:             List of parsed questions
280: 
281:         Raises
282:         ------
283:         FileNotFoundError
284:             If the CSV file doesn't exist
285:         ValueError
286:             If required columns are missing or data is malformed
287:         """
288:         filepath = Path(filepath)
289:         if not filepath.exists():
290:             raise FileNotFoundError(f"CSV file not found: {filepath}")
291: 
292:         questions = []
293: 
294:         with open(filepath, 'r', encoding='utf-8') as f:
295:             reader = csv.DictReader(f)
296: 
297:             # Validate required columns
298:             actual_columns = set(reader.fieldnames or [])
299: 
300:             # Handle alternate column names
301:             if 'Text' in actual_columns and 'question' not in actual_columns:
302:                 # QANTA format uses 'Text' instead of 'question'
303:                 text_col = 'Text'
304:             elif 'question' in actual_columns:
305:                 text_col = 'question'
306:             else:
307:                 raise ValueError(f"Missing required column 'question' or 'Text'. Found columns: {actual_columns}")
308: 
309:             if 'Answer' in actual_columns and 'answer' not in actual_columns:
310:                 answer_col = 'Answer'
311:             elif 'answer' in actual_columns:
312:                 answer_col = 'answer'
313:             else:
314:                 raise ValueError(f"Missing required column 'answer' or 'Answer'. Found columns: {actual_columns}")
315: 
316:             # Check for optional columns
317:             category_col = None
318:             if 'Category' in actual_columns:
319:                 category_col = 'Category'
320:             elif 'category' in actual_columns:
321:                 category_col = 'category'
322: 
323:             qid_col = None
324:             if 'Question ID' in actual_columns:
325:                 qid_col = 'Question ID'
326:             elif 'qid' in actual_columns:
327:                 qid_col = 'qid'
328:             elif 'question_id' in actual_columns:
329:                 qid_col = 'question_id'
330: 
331:             # Parse each row
332:             for row_idx, row in enumerate(reader):
333:                 try:
334:                     # Get question text and parse clues
335:                     question_text = row[text_col]
336:                     if not question_text or not question_text.strip():
337:                         continue  # Skip empty questions
338: 
339:                     # Split on ||| delimiter
340:                     if '|||' in question_text:
341:                         clues = [clue.strip() for clue in question_text.split('|||')]
342:                         clues = [c for c in clues if c]  # Remove empty clues
343:                     else:
344:                         # Treat entire text as single clue if no delimiter
345:                         clues = [question_text.strip()]
346: 
347:                     if not clues:
348:                         continue  # Skip if no valid clues
349: 
350:                     # Get answer
351:                     answer = row[answer_col].strip()
352:                     if not answer:
353:                         continue  # Skip questions without answers
354: 
355:                     # Get category (optional)
356:                     category = ""
357:                     if category_col:
358:                         category = row.get(category_col, "").strip()
359: 
360:                     # Get or generate question ID
361:                     if qid_col and row.get(qid_col):
362:                         qid = row[qid_col].strip()
363:                     else:
364:                         qid = _generate_qid(question_text)
365: 
366:                     # Parse clues into tokens and run indices
367:                     tokens, run_indices = _parse_clues_to_tokens(clues)
368: 
369:                     # Build cumulative prefixes
370:                     cumulative_prefixes = []
371:                     for idx in run_indices:
372:                         prefix = " ".join(tokens[:idx + 1])
373:                         cumulative_prefixes.append(prefix)
374: 
375:                     # Create clean answers list
376:                     clean_answers = [normalize_answer(answer)]
377: 
378:                     # Full question is all clues joined
379:                     full_question = " ".join(clues)
380: 
381:                     # Create TossupQuestion
382:                     question = TossupQuestion(
383:                         qid=qid,
384:                         question=full_question,
385:                         tokens=tokens,
386:                         answer_primary=answer,
387:                         clean_answers=clean_answers,
388:                         run_indices=run_indices,
389:                         human_buzz_positions=None,  # Not available in basic CSV
390:                         category=category,
391:                         cumulative_prefixes=cumulative_prefixes
392:                     )
393: 
394:                     questions.append(question)
395: 
396:                 except Exception as e:
397:                     print(f"Warning: Failed to parse row {row_idx + 1}: {e}")
398:                     continue
399: 
400:         if not questions:
401:             raise ValueError(f"No valid questions found in {filepath}")
402: 
403:         return questions
````

## File: qb_data/huggingface_loader.py
````python
  1: """
  2: HuggingFace dataset loader for quiz bowl data.
  3: 
  4: This module provides fallback loading from HuggingFace Hub when local CSV files
  5: are not available.
  6: """
  7: 
  8: from typing import List, Optional, Dict, Any
  9: 
 10: from qb_data.data_loader import TossupQuestion
 11: from qb_data.text_utils import tokenize_text, normalize_answer
 12: 
 13: 
 14: def load_from_huggingface(
 15:     dataset_name: str,
 16:     config_name: Optional[str] = None,
 17:     split: str = "eval"
 18: ) -> List[TossupQuestion]:
 19:     """
 20:     Load quiz bowl dataset from HuggingFace Hub.
 21: 
 22:     Parameters
 23:     ----------
 24:     dataset_name : str
 25:         Name of the HuggingFace dataset (e.g., "qanta-challenge/acf-co24-tossups")
 26:     config_name : Optional[str]
 27:         Configuration name for the dataset (e.g., "questions", "tossup")
 28:     split : str
 29:         Dataset split to load (default: "eval")
 30: 
 31:     Returns
 32:     -------
 33:     List[TossupQuestion]
 34:         List of parsed questions
 35: 
 36:     Raises
 37:     ------
 38:     ImportError
 39:         If datasets library is not installed
 40:     ValueError
 41:         If dataset not found or required fields missing
 42:     """
 43:     try:
 44:         from datasets import load_dataset
 45:     except ImportError:
 46:         print("Warning: datasets library not installed. Falling back to CSV loader.")
 47:         print("Install with: pip install datasets")
 48:         raise ImportError("HuggingFace datasets library not available. Please use CSV fallback.")
 49: 
 50:     # Known dataset configurations from qb-rl
 51:     known_configs = {
 52:         "qanta-challenge/acf-co24-tossups": "questions",
 53:         "qanta-challenge/qanta25-playground": "tossup"
 54:     }
 55: 
 56:     # Use known config if not provided
 57:     if config_name is None and dataset_name in known_configs:
 58:         config_name = known_configs[dataset_name]
 59:         print(f"Using known config '{config_name}' for {dataset_name}")
 60: 
 61:     # Try to load dataset
 62:     try:
 63:         print(f"Loading {dataset_name} from HuggingFace Hub...")
 64:         if config_name:
 65:             dataset = load_dataset(dataset_name, config_name, split=split)
 66:         else:
 67:             dataset = load_dataset(dataset_name, split=split)
 68:         print(f"Successfully loaded {len(dataset)} questions")
 69:     except Exception as e:
 70:         error_msg = f"Failed to load dataset {dataset_name}: {e}"
 71:         print(f"Error: {error_msg}")
 72:         print("Falling back to local CSV loader...")
 73:         raise ValueError(error_msg)
 74: 
 75:     # Parse dataset rows into TossupQuestion format
 76:     questions = []
 77:     for idx, row in enumerate(dataset):
 78:         try:
 79:             question = parse_huggingface_row(row, idx)
 80:             questions.append(question)
 81:         except KeyError as e:
 82:             print(f"Warning: Skipping row {idx} due to missing field: {e}")
 83:             continue
 84:         except Exception as e:
 85:             print(f"Warning: Failed to parse row {idx}: {e}")
 86:             continue
 87: 
 88:     if not questions:
 89:         raise ValueError(f"No valid questions parsed from {dataset_name}")
 90: 
 91:     print(f"Parsed {len(questions)} questions from HuggingFace dataset")
 92:     return questions
 93: 
 94: 
 95: def parse_huggingface_row(row: Dict[str, Any], idx: int = 0) -> TossupQuestion:
 96:     """
 97:     Parse a HuggingFace dataset row into TossupQuestion format.
 98: 
 99:     Parameters
100:     ----------
101:     row : Dict[str, Any]
102:         Single row from HuggingFace dataset
103:     idx : int
104:         Row index for generating IDs
105: 
106:     Returns
107:     -------
108:     TossupQuestion
109:         Parsed question object
110: 
111:     Raises
112:     ------
113:     KeyError
114:         If required fields are missing
115:     """
116:     # Field mapping for different dataset formats
117:     # Primary fields
118:     question_fields = ["question", "text", "question_text", "tossup_text"]
119:     answer_fields = ["answer_primary", "answer", "clean_answer", "clean_answers", "page"]
120:     category_fields = ["category", "topic", "subject"]
121: 
122:     # Extract question text
123:     question_text = None
124:     for field in question_fields:
125:         if field in row:
126:             question_text = row[field]
127:             break
128: 
129:     if not question_text:
130:         raise KeyError(f"No question field found. Available fields: {list(row.keys())}")
131: 
132:     # Extract answer
133:     answer_text = None
134:     for field in answer_fields:
135:         if field in row:
136:             value = row[field]
137:             # Handle list of answers
138:             if isinstance(value, list) and value:
139:                 answer_text = value[0]
140:             elif isinstance(value, str):
141:                 answer_text = value
142:             break
143: 
144:     if not answer_text:
145:         raise KeyError(f"No answer field found. Available fields: {list(row.keys())}")
146: 
147:     # Extract category (with default)
148:     category = "General"
149:     for field in category_fields:
150:         if field in row and row[field]:
151:             category = str(row[field])
152:             break
153: 
154:     # Generate ID if not present
155:     qid = row.get("qid") or row.get("id") or row.get("qanta_id") or f"hf_{idx:06d}"
156: 
157:     # Handle clues that may be separated by ||| or in a list
158:     if "|||" in question_text:
159:         # QANTA format with ||| separators
160:         clues = question_text.split("|||")
161:         question_text = " ".join(clues)
162:     elif isinstance(question_text, list):
163:         # List of clues
164:         clues = question_text
165:         question_text = " ".join(clues)
166:     else:
167:         # Single text, split by sentences as approximation
168:         import re
169:         sentences = re.split(r'(?<=[.!?])\s+', question_text)
170:         clues = sentences if len(sentences) > 1 else [question_text]
171: 
172:     # Tokenize text
173:     tokens = tokenize_text(question_text)
174: 
175:     # Build run indices (boundaries between clues)
176:     run_indices = []
177:     current_pos = 0
178:     for clue in clues:
179:         clue_tokens = tokenize_text(clue)
180:         current_pos += len(clue_tokens)
181:         if current_pos > 0:
182:             run_indices.append(current_pos - 1)  # Index is 0-based
183: 
184:     # Build cumulative prefixes
185:     cumulative_prefixes = []
186:     for idx in run_indices:
187:         prefix = " ".join(tokens[:idx + 1])
188:         cumulative_prefixes.append(prefix)
189: 
190:     # Normalize answer for matching
191:     clean_answers = [normalize_answer(answer_text)]
192: 
193:     return TossupQuestion(
194:         qid=qid,
195:         question=question_text,
196:         tokens=tokens,
197:         answer_primary=answer_text,  # Keep original answer as primary
198:         clean_answers=clean_answers,  # Normalized version for matching
199:         run_indices=run_indices,
200:         human_buzz_positions=None,  # Not available from HuggingFace
201:         category=category,
202:         cumulative_prefixes=cumulative_prefixes
203:     )
204: 
205: 
206: def try_huggingface_fallback(csv_path: str) -> Optional[List[TossupQuestion]]:
207:     """
208:     Attempt to load from HuggingFace if CSV is missing.
209: 
210:     Parameters
211:     ----------
212:     csv_path : str
213:         Path to missing CSV file
214: 
215:     Returns
216:     -------
217:     Optional[List[TossupQuestion]]
218:         Questions if HuggingFace load succeeds, None otherwise
219:     """
220:     print(f"CSV file {csv_path} not found. Attempting HuggingFace fallback...")
221: 
222:     # Try known datasets in order
223:     fallback_datasets = [
224:         ("qanta-challenge/acf-co24-tossups", "questions"),
225:         ("qanta-challenge/qanta25-playground", "tossup")
226:     ]
227: 
228:     for dataset_name, config_name in fallback_datasets:
229:         try:
230:             questions = load_from_huggingface(dataset_name, config_name)
231:             if questions:
232:                 print(f"Successfully loaded {len(questions)} questions from {dataset_name}")
233:                 return questions
234:         except Exception as e:
235:             print(f"Failed to load {dataset_name}: {e}")
236:             continue
237: 
238:     print("All HuggingFace fallback attempts failed")
239:     return None
````

## File: qb_data/text_utils.py
````python
 1: """
 2: Text utilities for quiz bowl answer normalization and tokenization.
 3: """
 4: 
 5: import re
 6: from typing import Optional, List
 7: 
 8: 
 9: def tokenize_text(text: str) -> List[str]:
10:     """
11:     Tokenize text by splitting on whitespace.
12: 
13:     Parameters
14:     ----------
15:     text : str
16:         Text to tokenize
17: 
18:     Returns
19:     -------
20:     List[str]
21:         List of tokens (words)
22:     """
23:     if not text:
24:         return []
25:     return text.split()
26: 
27: 
28: def normalize_answer(answer: str) -> str:
29:     """
30:     Normalize an answer string for comparison.
31: 
32:     Removes articles (a, an, the) from the beginning, converts to lowercase,
33:     strips punctuation and extra whitespace, and handles edge cases.
34: 
35:     Parameters
36:     ----------
37:     answer : str
38:         The answer string to normalize
39: 
40:     Returns
41:     -------
42:     str
43:         The normalized answer string
44: 
45:     Examples
46:     --------
47:     >>> normalize_answer("The Great Gatsby")
48:     'great gatsby'
49:     >>> normalize_answer("A Tale of Two Cities!")
50:     'tale of two cities'
51:     >>> normalize_answer("   An    Example   ")
52:     'example'
53:     >>> normalize_answer("")
54:     ''
55:     """
56:     if not answer:
57:         return ""
58: 
59:     # Convert to lowercase
60:     normalized = answer.lower()
61: 
62:     # Remove leading/trailing whitespace
63:     normalized = normalized.strip()
64: 
65:     # Remove leading articles (a, an, the)
66:     # Use \b word boundary to ensure we match complete words
67:     normalized = re.sub(r'^(a|an|the)\b\s*', '', normalized)
68: 
69:     # Remove punctuation
70:     # Keep alphanumeric characters and spaces
71:     normalized = re.sub(r'[^\w\s]', '', normalized)
72: 
73:     # Normalize whitespace (collapse multiple spaces to single space)
74:     normalized = re.sub(r'\s+', ' ', normalized)
75: 
76:     # Final strip in case punctuation removal left spaces
77:     normalized = normalized.strip()
78: 
79:     return normalized
````

## File: qb_env/__init__.py
````python
 1: """Quiz Bowl Environment Package.
 2: 
 3: Gymnasium-compliant POMDP environment for quiz bowl question answering,
 4: plus thin qb-rl compatibility exports for the old `qb_env.*` import paths.
 5: """
 6: 
 7: from qb_env.data_loader import (
 8:     QANTADatasetLoader,
 9:     TossupQuestion,
10:     load_tossup_questions,
11:     load_tossup_questions_from_config,
12:     parse_row,
13: )
14: from qb_env.mc_builder import MCBuilder, MCQuestion
15: from qb_env.stop_only_env import StopOnlyEnv
16: from qb_env.text_utils import normalize_answer, tokenize_text
17: from qb_env.tossup_env import TossupMCEnv, make_env_from_config
18: from qb_env.text_wrapper import TextObservationWrapper
19: 
20: __all__ = [
21:     "TossupMCEnv",
22:     "make_env_from_config",
23:     "TextObservationWrapper",
24:     "TossupQuestion",
25:     "QANTADatasetLoader",
26:     "parse_row",
27:     "load_tossup_questions",
28:     "load_tossup_questions_from_config",
29:     "MCQuestion",
30:     "MCBuilder",
31:     "StopOnlyEnv",
32:     "normalize_answer",
33:     "tokenize_text",
34: ]
````

## File: qb_env/opponent_models.py
````python
  1: """Opponent buzz-position models for Expected Wins reward computation.
  2: 
  3: Provides pluggable opponent models that estimate the probability an
  4: opponent has buzzed before a given step.  Used by the ``expected_wins``
  5: reward mode in :class:`TossupMCEnv`.
  6: 
  7: Three built-in models:
  8: 
  9: * :class:`EmpiricalHistogramOpponentModel` — derives CDF from
 10:   ``MCQuestion.human_buzz_positions`` data.
 11: * :class:`LogisticOpponentModel` — parametric sigmoid CDF for
 12:   questions that lack empirical data.
 13: * :func:`build_opponent_model_from_config` — factory with fallback
 14:   hierarchy: question-level empirical → global empirical → logistic.
 15: 
 16: The ``expected_wins`` reward mode is disabled by default.  To enable,
 17: set ``environment.reward_mode: expected_wins`` and optionally configure
 18: ``environment.opponent_buzz_model`` in the YAML config.
 19: """
 20: 
 21: from __future__ import annotations
 22: 
 23: import math
 24: from typing import Any, Protocol, runtime_checkable
 25: 
 26: import numpy as np
 27: 
 28: from qb_data.mc_builder import MCQuestion
 29: 
 30: 
 31: @runtime_checkable
 32: class OpponentBuzzModel(Protocol):
 33:     """Protocol for opponent buzz-position models."""
 34: 
 35:     def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
 36:         """Cumulative probability that the opponent has buzzed before *step_idx*.
 37: 
 38:         Parameters
 39:         ----------
 40:         question : MCQuestion
 41:             Current question (may carry ``human_buzz_positions``).
 42:         step_idx : int
 43:             0-based clue step.
 44: 
 45:         Returns
 46:         -------
 47:         float
 48:             P(opponent buzzed before step_idx), in [0, 1].
 49:         """
 50:         ...
 51: 
 52:     def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
 53:         """Probability that the opponent has NOT buzzed by *step_idx*.
 54: 
 55:         Complement of :meth:`prob_buzzed_before_step`.
 56:         """
 57:         ...
 58: 
 59: 
 60: class LogisticOpponentModel:
 61:     """Parametric logistic CDF opponent model.
 62: 
 63:     Models the opponent's cumulative buzz probability at step *t* as::
 64: 
 65:         P(buzzed before t) = 1 / (1 + exp(-steepness * (t/total - midpoint)))
 66: 
 67:     Parameters
 68:     ----------
 69:     midpoint : float
 70:         Fraction of total steps at which the CDF reaches 0.5.
 71:     steepness : float
 72:         Controls how sharply the probability increases around the
 73:         midpoint.  Higher values → sharper transition.
 74:     """
 75: 
 76:     def __init__(self, midpoint: float = 0.6, steepness: float = 6.0) -> None:
 77:         self.midpoint = midpoint
 78:         self.steepness = steepness
 79: 
 80:     def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
 81:         total = len(question.cumulative_prefixes)
 82:         if total <= 1:
 83:             return 0.0
 84:         frac = step_idx / total
 85:         x = self.steepness * (frac - self.midpoint)
 86:         if x >= 0:
 87:             return 1.0 / (1.0 + math.exp(-x))
 88:         z = math.exp(x)
 89:         return z / (1.0 + z)
 90: 
 91:     def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
 92:         return 1.0 - self.prob_buzzed_before_step(question, step_idx)
 93: 
 94: 
 95: class EmpiricalHistogramOpponentModel:
 96:     """Opponent model derived from empirical human buzz-position data.
 97: 
 98:     Builds a per-step CDF from the ``human_buzz_positions`` field on
 99:     each question.  Falls back to a :class:`LogisticOpponentModel`
100:     when a question has no empirical data.
101: 
102:     Parameters
103:     ----------
104:     fallback : LogisticOpponentModel or None
105:         Model to use when a question lacks empirical data.
106:     global_positions : list of (int, int) or None
107:         Pooled (position, count) pairs from the entire dataset.
108:         Used when a question has no per-question data but a global
109:         distribution is available.
110:     """
111: 
112:     def __init__(
113:         self,
114:         fallback: LogisticOpponentModel | None = None,
115:         global_positions: list[tuple[int, int]] | None = None,
116:     ) -> None:
117:         self.fallback = fallback or LogisticOpponentModel()
118:         self._global_cdf: np.ndarray | None = None
119:         if global_positions:
120:             self._global_cdf = self._build_cdf(global_positions)
121: 
122:     @staticmethod
123:     def _build_cdf(positions: list[tuple[int, int]]) -> np.ndarray:
124:         """Build a CDF array from (position, count) pairs.
125: 
126:         Returns an array where ``cdf[i]`` is the cumulative probability
127:         that a buzz has occurred at or before position *i*.
128:         """
129:         if not positions:
130:             return np.array([], dtype=np.float64)
131:         max_pos = max(p for p, _ in positions)
132:         counts = np.zeros(max_pos + 1, dtype=np.float64)
133:         for pos, count in positions:
134:             counts[pos] += count
135:         total = counts.sum()
136:         if total <= 0:
137:             return np.zeros(max_pos + 1, dtype=np.float64)
138:         return np.cumsum(counts) / total
139: 
140:     def _cdf_at_step(
141:         self, cdf: np.ndarray, question: MCQuestion, step_idx: int
142:     ) -> float:
143:         """Look up cumulative probability at a token position."""
144:         if cdf.size == 0:
145:             return 0.0
146:         if not question.run_indices:
147:             token_pos = step_idx
148:         elif step_idx < len(question.run_indices):
149:             token_pos = question.run_indices[step_idx]
150:         else:
151:             token_pos = question.run_indices[-1] if question.run_indices else step_idx
152:         idx = min(token_pos, len(cdf) - 1)
153:         return float(cdf[idx])
154: 
155:     def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
156:         if question.human_buzz_positions:
157:             cdf = self._build_cdf(question.human_buzz_positions)
158:             return self._cdf_at_step(cdf, question, step_idx)
159:         if self._global_cdf is not None and self._global_cdf.size > 0:
160:             return self._cdf_at_step(self._global_cdf, question, step_idx)
161:         return self.fallback.prob_buzzed_before_step(question, step_idx)
162: 
163:     def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
164:         return 1.0 - self.prob_buzzed_before_step(question, step_idx)
165: 
166: 
167: def build_opponent_model_from_config(
168:     questions: list[MCQuestion] | None = None,
169:     config: dict[str, Any] | None = None,
170: ) -> OpponentBuzzModel | None:
171:     """Build an opponent model from YAML configuration.
172: 
173:     Returns ``None`` when the opponent model is disabled (the default).
174: 
175:     Parameters
176:     ----------
177:     questions : list[MCQuestion] or None
178:         Dataset questions for building global empirical distribution.
179:     config : dict or None
180:         Full YAML config dict.
181: 
182:     Returns
183:     -------
184:     OpponentBuzzModel or None
185:     """
186:     if config is None:
187:         return None
188:     env_cfg = config.get("environment", {})
189:     opp_cfg = env_cfg.get("opponent_buzz_model", {})
190:     if not opp_cfg or opp_cfg.get("type", "none") == "none":
191:         return None
192: 
193:     model_type = opp_cfg.get("type", "logistic")
194: 
195:     if model_type == "logistic":
196:         return LogisticOpponentModel(
197:             midpoint=float(opp_cfg.get("midpoint", 0.6)),
198:             steepness=float(opp_cfg.get("steepness", 6.0)),
199:         )
200: 
201:     if model_type == "empirical":
202:         global_positions: list[tuple[int, int]] = []
203:         if questions:
204:             for q in questions:
205:                 if q.human_buzz_positions:
206:                     global_positions.extend(q.human_buzz_positions)
207:         fallback = LogisticOpponentModel(
208:             midpoint=float(opp_cfg.get("midpoint", 0.6)),
209:             steepness=float(opp_cfg.get("steepness", 6.0)),
210:         )
211:         return EmpiricalHistogramOpponentModel(
212:             fallback=fallback,
213:             global_positions=global_positions if global_positions else None,
214:         )
215: 
216:     raise ValueError(f"Unknown opponent_buzz_model type: {model_type}")
````

## File: qb_env/stop_only_env.py
````python
 1: """Stop-only action-space wrapper for the quiz bowl environment."""
 2: 
 3: from __future__ import annotations
 4: 
 5: from typing import Any
 6: 
 7: import gymnasium as gym
 8: import numpy as np
 9: from gymnasium import spaces
10: 
11: from qb_env.tossup_env import TossupMCEnv
12: 
13: 
14: class StopOnlyEnv(gym.Wrapper):
15:     """Wrap TossupMCEnv with a binary WAIT/BUZZ action space.
16: 
17:     Action mapping:
18:     - 0 -> WAIT
19:     - 1 -> BUZZ using the current answer-selection strategy
20: 
21:     The default answer-selection strategy commits to the current belief argmax.
22:     """
23: 
24:     def __init__(self, env: TossupMCEnv, answer_mode: str = "argmax_belief") -> None:
25:         super().__init__(env)
26:         self.answer_mode = answer_mode
27:         self.action_space = spaces.Discrete(2)
28: 
29:     def reset(
30:         self,
31:         *,
32:         seed: int | None = None,
33:         options: dict[str, Any] | None = None,
34:     ) -> tuple[np.ndarray, dict[str, Any]]:
35:         return self.env.reset(seed=seed, options=options)
36: 
37:     def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
38:         if not self.action_space.contains(action):
39:             raise ValueError(f"Invalid action: {action}")
40:         if action == 0:
41:             return self.env.step(0)
42: 
43:         if self.answer_mode != "argmax_belief":
44:             raise ValueError(f"Unknown answer_mode: {self.answer_mode}")
45: 
46:         chosen_idx = int(np.argmax(self.env.belief))
47:         return self.env.step(1 + chosen_idx)
````

## File: tests/test_action_space_alignment.py
````python
 1: """Integration-style guards for the PR1 feature-port subset."""
 2: 
 3: from __future__ import annotations
 4: 
 5: import pytest
 6: import torch
 7: 
 8: from models.t5_policy import T5PolicyModel
 9: from models.likelihoods import TfIdfLikelihood
10: from qb_env import StopOnlyEnv, TossupMCEnv
11: 
12: 
13: @pytest.fixture(scope="module")
14: def t5_small_model():
15:     try:
16:         model = T5PolicyModel(
17:             {
18:                 "model_name": "t5-small",
19:                 "device": "cpu",
20:                 "max_input_length": 128,
21:                 "num_choices": 4,
22:             }
23:         )
24:     except OSError as exc:
25:         pytest.skip(f"t5-small unavailable in test environment: {exc}")
26:     model.eval()
27:     return model
28: 
29: 
30: def test_t5_wait_log_prob_does_not_depend_on_answer_logits(t5_small_model):
31:     """WAIT log-prob is independent of answer-head mass."""
32:     model = t5_small_model
33:     joint_log_prob = getattr(model, "_joint_action_log_prob")
34:     wait_logits = torch.tensor([[1.5, -0.5]], dtype=torch.float32, device=model.device)
35:     answer_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32, device=model.device)
36:     actions = torch.tensor([0], dtype=torch.long, device=model.device)
37: 
38:     lp1 = joint_log_prob(wait_logits, answer_logits, actions)
39:     lp2 = joint_log_prob(wait_logits, answer_logits.flip(dims=[-1]), actions)
40:     assert torch.allclose(lp1, lp2, atol=1e-6)
41: 
42: 
43: def test_t5_entropy_uses_chain_rule(t5_small_model):
44:     """Joint entropy follows H(wait) + p_buzz * H(answer)."""
45:     model = t5_small_model
46:     joint_entropy = getattr(model, "_joint_entropy")
47:     wait_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=model.device)
48:     answer_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], dtype=torch.float32, device=model.device)
49: 
50:     entropy = joint_entropy(wait_logits, answer_logits)
51:     wait_probs = torch.softmax(wait_logits, dim=-1)
52:     wait_log_probs = torch.log_softmax(wait_logits, dim=-1)
53:     answer_probs = torch.softmax(answer_logits, dim=-1)
54:     answer_log_probs = torch.log_softmax(answer_logits, dim=-1)
55:     expected = (
56:         -(wait_probs * wait_log_probs).sum(dim=-1)
57:         + wait_probs[:, 1] * (-(answer_probs * answer_log_probs).sum(dim=-1))
58:     )
59:     assert torch.allclose(entropy, expected, atol=1e-6)
60: 
61: 
62: def test_stop_only_env_has_discrete_2_action_space(sample_tfidf_env):
63:     env = StopOnlyEnv(sample_tfidf_env)
64:     assert env.action_space.n == 2
65: 
66: 
67: def test_flat_kplus1_mode_still_available(sample_tfidf_env):
68:     assert sample_tfidf_env.action_space.n == 5
69: 
70: 
71: def test_no_buzz_end_mode_does_not_force_choice(sample_mc_question):
72:     corpus = sample_mc_question.option_profiles[:]
73:     model = TfIdfLikelihood(corpus_texts=corpus)
74:     env = TossupMCEnv(
75:         questions=[sample_mc_question],
76:         likelihood_model=model,
77:         K=4,
78:         reward_mode="simple",
79:         end_mode="no_buzz",
80:         no_buzz_reward=0.0,
81:     )
82:     _obs, _info = env.reset(seed=0)
83:     while True:
84:         _obs, _reward, _term, truncated, info = env.step(0)
85:         if truncated:
86:             break
87:     assert info.get("no_buzz") is True
88:     assert info.get("forced_choice") == -1
89:     assert info.get("forced_correct") is False
````

## File: tests/test_answer_profile_cache.py
````python
  1: """Tests for AnswerProfileBuilder._cache memoization.
  2: 
  3: Verifies that:
  4: 1. Distractor profiles (exclude_qid=None) are cached and return identical results
  5: 2. Leave-one-out profiles (answer, qid) are cached and return identical results
  6: 3. Cache is invalidated on fit() with new data
  7: 4. Cached distractor profile is byte-identical to freshly computed profile
  8: 5. Cached leave-one-out profile is byte-identical to freshly computed profile
  9: 6. Cache reduces actual computation (single entry per unique key)
 10: """
 11: 
 12: from __future__ import annotations
 13: 
 14: import pytest
 15: 
 16: from qb_data.answer_profiles import AnswerProfileBuilder
 17: from qb_data.data_loader import TossupQuestion
 18: 
 19: 
 20: def _make_question(
 21:     qid: str,
 22:     answer: str,
 23:     text: str,
 24:     category: str = "History",
 25: ) -> TossupQuestion:
 26:     """Create a minimal TossupQuestion for cache testing."""
 27:     tokens = text.split()
 28:     return TossupQuestion(
 29:         qid=qid,
 30:         question=text,
 31:         tokens=tokens,
 32:         answer_primary=answer,
 33:         clean_answers=[answer],
 34:         run_indices=[len(tokens) - 1],
 35:         human_buzz_positions=[],
 36:         category=category,
 37:         cumulative_prefixes=[text],
 38:     )
 39: 
 40: 
 41: @pytest.fixture
 42: def sample_questions() -> list[TossupQuestion]:
 43:     """Five questions with 3 shared answers for exercising cache hits."""
 44:     return [
 45:         _make_question("q1", "Washington", "first president commander in chief"),
 46:         _make_question("q2", "Washington", "led the continental army to victory"),
 47:         _make_question("q3", "Jefferson", "wrote the declaration of independence"),
 48:         _make_question("q4", "Jefferson", "third president and diplomat to France"),
 49:         _make_question("q5", "Lincoln", "preserved the union during civil war"),
 50:     ]
 51: 
 52: 
 53: @pytest.fixture
 54: def builder(sample_questions: list[TossupQuestion]) -> AnswerProfileBuilder:
 55:     """Return a fitted AnswerProfileBuilder."""
 56:     b = AnswerProfileBuilder(max_tokens_per_profile=2000, min_questions_per_answer=1)
 57:     b.fit(sample_questions)
 58:     return b
 59: 
 60: 
 61: class TestProfileCacheHits:
 62:     """Repeated calls with the same args return the same cached result."""
 63: 
 64:     def test_distractor_profile_cached(
 65:         self, builder: AnswerProfileBuilder
 66:     ) -> None:
 67:         """profile_for_answer returns identical string on repeated (answer, None)."""
 68:         first = builder.profile_for_answer("Washington", exclude_qid=None)
 69:         second = builder.profile_for_answer("Washington", exclude_qid=None)
 70:         assert first is second  # same object, not just equal
 71: 
 72:     def test_leave_one_out_profile_cached(
 73:         self, builder: AnswerProfileBuilder
 74:     ) -> None:
 75:         """profile_for_answer returns identical string on repeated (answer, qid)."""
 76:         first = builder.profile_for_answer("Washington", exclude_qid="q1")
 77:         second = builder.profile_for_answer("Washington", exclude_qid="q1")
 78:         assert first is second  # same object from cache
 79: 
 80: 
 81: class TestCacheInvalidation:
 82:     """fit() with new data clears the cache."""
 83: 
 84:     def test_fit_clears_cache(
 85:         self, builder: AnswerProfileBuilder, sample_questions: list[TossupQuestion]
 86:     ) -> None:
 87:         """After fit() with new data, cache is empty and profiles reflect new data."""
 88:         # Populate cache
 89:         builder.profile_for_answer("Washington", exclude_qid=None)
 90:         assert len(builder._cache) > 0
 91: 
 92:         # Re-fit with different data
 93:         new_questions = [
 94:             _make_question("q99", "Washington", "completely different text about cherry trees"),
 95:         ]
 96:         builder.fit(new_questions)
 97:         assert len(builder._cache) == 0
 98: 
 99:         # New profile should reflect new data
100:         profile = builder.profile_for_answer("Washington", exclude_qid=None)
101:         assert "cherry" in profile
102: 
103: 
104: class TestCacheEquivalence:
105:     """Cached profiles are byte-identical to freshly computed profiles."""
106: 
107:     def test_distractor_cache_equivalence(
108:         self, sample_questions: list[TossupQuestion]
109:     ) -> None:
110:         """Cached (answer, None) profile is byte-identical to a fresh computation."""
111:         # Build fresh (uncached) profile
112:         fresh_builder = AnswerProfileBuilder(
113:             max_tokens_per_profile=2000, min_questions_per_answer=1
114:         )
115:         fresh_builder.fit(sample_questions)
116:         fresh_profile = fresh_builder._profile_text("Jefferson", exclude_qid=None)
117: 
118:         # Build cached profile
119:         cached_builder = AnswerProfileBuilder(
120:             max_tokens_per_profile=2000, min_questions_per_answer=1
121:         )
122:         cached_builder.fit(sample_questions)
123:         _ = cached_builder._profile_text("Jefferson", exclude_qid=None)  # populate cache
124:         cached_profile = cached_builder._profile_text("Jefferson", exclude_qid=None)  # from cache
125: 
126:         assert fresh_profile == cached_profile
127: 
128:     def test_leave_one_out_cache_equivalence(
129:         self, sample_questions: list[TossupQuestion]
130:     ) -> None:
131:         """Cached (answer, qid) profile is byte-identical to a fresh computation."""
132:         fresh_builder = AnswerProfileBuilder(
133:             max_tokens_per_profile=2000, min_questions_per_answer=1
134:         )
135:         fresh_builder.fit(sample_questions)
136:         fresh_profile = fresh_builder._profile_text("Washington", exclude_qid="q1")
137: 
138:         cached_builder = AnswerProfileBuilder(
139:             max_tokens_per_profile=2000, min_questions_per_answer=1
140:         )
141:         cached_builder.fit(sample_questions)
142:         _ = cached_builder._profile_text("Washington", exclude_qid="q1")
143:         cached_profile = cached_builder._profile_text("Washington", exclude_qid="q1")
144: 
145:         assert fresh_profile == cached_profile
146: 
147: 
148: class TestCacheEfficiency:
149:     """Cache reduces computation to one real call per unique key."""
150: 
151:     def test_cache_stores_one_entry_per_unique_key(
152:         self, builder: AnswerProfileBuilder
153:     ) -> None:
154:         """Calling _profile_text N times with same args results in 1 cache entry."""
155:         for _ in range(10):
156:             builder.profile_for_answer("Lincoln", exclude_qid=None)
157: 
158:         # Only one cache entry for (Lincoln, None)
159:         assert ("Lincoln", None) in builder._cache
160:         assert len([k for k in builder._cache if k[0] == "Lincoln"]) == 1
````

## File: tests/test_build_mc_dataset.py
````python
  1: """Regression tests for scripts/build_mc_dataset.py CLI defaults."""
  2: 
  3: from __future__ import annotations
  4: 
  5: from pathlib import Path
  6: 
  7: from qb_data.config import load_config as load_yaml_config, merge_overrides
  8: from scripts.build_mc_dataset import parse_args, parse_overrides, resolve_output_dir
  9: 
 10: 
 11: class TestBuildMcDatasetArgs:
 12:     """Tests for smoke-aware argument resolution."""
 13: 
 14:     def test_parse_args_smoke_uses_dynamic_defaults(self) -> None:
 15:         args = parse_args(["--smoke"])
 16: 
 17:         assert args.smoke is True
 18:         assert args.config is None
 19:         assert args.output_dir is None
 20:         assert args.overrides == []
 21: 
 22:     def test_parse_args_explicit_overrides_win(self) -> None:
 23:         args = parse_args(
 24:             [
 25:                 "--smoke",
 26:                 "--config",
 27:                 "configs/custom.yaml",
 28:                 "--output-dir",
 29:                 "custom/output",
 30:                 "data.K=5",
 31:             ]
 32:         )
 33: 
 34:         assert args.smoke is True
 35:         assert args.config == "configs/custom.yaml"
 36:         assert args.output_dir == "custom/output"
 37:         assert args.overrides == ["data.K=5"]
 38: 
 39:     def test_resolve_output_dir_defaults_to_smoke_artifacts(self) -> None:
 40:         assert resolve_output_dir(None, smoke=True) == Path("artifacts/smoke")
 41: 
 42:     def test_resolve_output_dir_defaults_to_processed_data(self) -> None:
 43:         assert resolve_output_dir(None, smoke=False) == Path("data/processed")
 44: 
 45:     def test_resolve_output_dir_preserves_explicit_override(self) -> None:
 46:         assert resolve_output_dir("custom/output", smoke=True) == Path("custom/output")
 47: 
 48:     def test_load_config_smoke_without_explicit_path(self) -> None:
 49:         cfg = load_yaml_config(None, smoke=True)
 50: 
 51:         assert cfg["data"]["max_questions"] == 50
 52:         assert cfg["ppo"]["total_timesteps"] == 3000
 53: 
 54: 
 55: class TestParseOverrides:
 56:     """Tests for the fixed flat-key override parsing."""
 57: 
 58:     def test_returns_dotted_keys(self) -> None:
 59:         """parse_overrides must return flat dotted keys, not nested dicts."""
 60:         args = parse_args(["data.K=5", "environment.reward_mode=simple"])
 61:         overrides = parse_overrides(args)
 62:         assert "data.K" in overrides
 63:         assert overrides["data.K"] == 5
 64:         assert "environment.reward_mode" in overrides
 65:         assert overrides["environment.reward_mode"] == "simple"
 66:         assert "data" not in overrides, "Must not nest into a 'data' sub-dict"
 67: 
 68:     def test_preserves_sibling_sections(self) -> None:
 69:         """Overriding data.K must not clobber data.csv_path."""
 70:         base = {
 71:             "data": {"K": 4, "csv_path": "questions.csv", "distractor_strategy": "sbert_profile"},
 72:             "environment": {"reward_mode": "time_penalty", "seed": 13},
 73:         }
 74:         args = parse_args(["data.K=5"])
 75:         overrides = parse_overrides(args)
 76:         merged = merge_overrides(dict(base), overrides)
 77:         assert merged["data"]["K"] == 5
 78:         assert merged["data"]["csv_path"] == "questions.csv"
 79:         assert merged["data"]["distractor_strategy"] == "sbert_profile"
 80:         assert merged["environment"]["reward_mode"] == "time_penalty"
 81: 
 82:     def test_value_types(self) -> None:
 83:         """Values are parsed as int, float, bool, or string."""
 84:         args = parse_args(["data.K=5", "likelihood.beta=3.5", "data.shuffle=true", "data.name=foo"])
 85:         overrides = parse_overrides(args)
 86:         assert overrides["data.K"] == 5
 87:         assert isinstance(overrides["data.K"], int)
 88:         assert overrides["likelihood.beta"] == 3.5
 89:         assert isinstance(overrides["likelihood.beta"], float)
 90:         assert overrides["data.shuffle"] is True
 91:         assert overrides["data.name"] == "foo"
 92: 
 93:     def test_no_overrides_returns_empty(self) -> None:
 94:         args = parse_args(["--smoke"])
 95:         overrides = parse_overrides(args)
 96:         assert overrides == {}
 97: 
 98:     def test_merge_overrides_leaf_only(self) -> None:
 99:         """merge_overrides with dotted keys updates only targeted leaves."""
100:         config = {
101:             "data": {"K": 4, "csv_path": "q.csv"},
102:             "environment": {"reward_mode": "simple"},
103:         }
104:         result = merge_overrides(config, {"data.K": 6, "environment.reward_mode": "time_penalty"})
105:         assert result["data"]["K"] == 6
106:         assert result["data"]["csv_path"] == "q.csv"
107:         assert result["environment"]["reward_mode"] == "time_penalty"
````

## File: tests/test_dataset_splits.py
````python
  1: """Tests for stratified dataset splitting reproducibility.
  2: 
  3: Verifies that splits are deterministic across invocations and do not
  4: depend on Python's hash randomization (PYTHONHASHSEED).
  5: """
  6: 
  7: import subprocess
  8: import sys
  9: 
 10: import pytest
 11: 
 12: from qb_data.data_loader import TossupQuestion
 13: from qb_data.dataset_splits import create_stratified_splits
 14: 
 15: 
 16: def _make_questions(n: int, categories: list[str]) -> list[TossupQuestion]:
 17:     """Create n dummy TossupQuestion instances cycling through categories."""
 18:     questions = []
 19:     for i in range(n):
 20:         cat = categories[i % len(categories)]
 21:         questions.append(
 22:             TossupQuestion(
 23:                 qid=f"q{i:04d}",
 24:                 question=f"Question {i}",
 25:                 tokens=[f"token{i}"],
 26:                 answer_primary=f"Answer {i}",
 27:                 clean_answers=[f"Answer {i}"],
 28:                 run_indices=[0],
 29:                 human_buzz_positions=[],
 30:                 category=cat,
 31:                 cumulative_prefixes=[f"token{i}"],
 32:             )
 33:         )
 34:     return questions
 35: 
 36: 
 37: def test_splits_deterministic_same_process():
 38:     """Same seed produces identical splits within one process."""
 39:     questions = _make_questions(60, ["History", "Science", "Literature"])
 40:     train1, val1, test1 = create_stratified_splits(questions, seed=42)
 41:     train2, val2, test2 = create_stratified_splits(questions, seed=42)
 42:     assert [q.qid for q in train1] == [q.qid for q in train2]
 43:     assert [q.qid for q in val1] == [q.qid for q in val2]
 44:     assert [q.qid for q in test1] == [q.qid for q in test2]
 45: 
 46: 
 47: def test_splits_deterministic_across_processes():
 48:     """Splits must be identical even with different PYTHONHASHSEED values.
 49: 
 50:     Runs the split in two subprocesses with different PYTHONHASHSEED and
 51:     checks that they produce identical qid orderings.
 52:     """
 53:     script = (
 54:         "import json, sys, io; sys.path.insert(0, '.'); "
 55:         "sys.stdout = io.StringIO(); "
 56:         "from qb_data.data_loader import TossupQuestion; "
 57:         "from qb_data.dataset_splits import create_stratified_splits; "
 58:         "qs = [TossupQuestion(qid=f'q{i:04d}', question=f'Q{i}', tokens=[f't{i}'], "
 59:         "answer_primary=f'A{i}', clean_answers=[f'A{i}'], run_indices=[0], "
 60:         "human_buzz_positions=[], category=['History','Science','Lit'][i%3], "
 61:         "cumulative_prefixes=[f't{i}']) for i in range(60)]; "
 62:         "tr,va,te = create_stratified_splits(qs, seed=42); "
 63:         "sys.stdout = sys.__stdout__; "
 64:         "print(json.dumps([q.qid for q in tr]))"
 65:     )
 66:     import json
 67:     import os
 68: 
 69:     base_env = {k: v for k, v in os.environ.items()}
 70:     repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
 71:     results = []
 72:     for hashseed in ["0", "12345"]:
 73:         env = {**base_env, "PYTHONHASHSEED": hashseed}
 74:         proc = subprocess.run(
 75:             [sys.executable, "-c", script],
 76:             capture_output=True,
 77:             text=True,
 78:             env=env,
 79:             cwd=repo_root,
 80:             timeout=30,
 81:         )
 82:         assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
 83:         results.append(json.loads(proc.stdout.strip()))
 84:     assert results[0] == results[1], (
 85:         "Splits differ across PYTHONHASHSEED values — hash(category) is not deterministic"
 86:     )
 87: 
 88: 
 89: def test_splits_different_seeds_differ():
 90:     """Different seeds should produce different splits."""
 91:     questions = _make_questions(60, ["History", "Science", "Literature"])
 92:     train1, _, _ = create_stratified_splits(questions, seed=42)
 93:     train2, _, _ = create_stratified_splits(questions, seed=99)
 94:     assert [q.qid for q in train1] != [q.qid for q in train2]
 95: 
 96: 
 97: def test_splits_all_questions_assigned():
 98:     """Every question must appear in exactly one split."""
 99:     questions = _make_questions(100, ["A", "B", "C", "D"])
100:     train, val, test = create_stratified_splits(questions, seed=1)
101:     all_qids = {q.qid for q in train} | {q.qid for q in val} | {q.qid for q in test}
102:     assert len(all_qids) == 100
103:     assert len(train) + len(val) + len(test) == 100
````

## File: tests/test_features.py
````python
  1: """Test suite for models/features.py — belief feature extraction.
  2: 
  3: Covers ENV-03: Belief feature extraction produces (K+6)-dimensional vectors
  4: with correct derived features (entropy, margin, stability, progress).
  5: """
  6: 
  7: from __future__ import annotations
  8: 
  9: import numpy as np
 10: import pytest
 11: 
 12: from models.features import entropy_of_distribution, extract_belief_features
 13: 
 14: 
 15: # ------------------------------------------------------------------ #
 16: # Tests for entropy_of_distribution
 17: # ------------------------------------------------------------------ #
 18: 
 19: 
 20: class TestEntropyOfDistribution:
 21:     """Tests for Shannon entropy computation."""
 22: 
 23:     def test_entropy_uniform(self) -> None:
 24:         """Uniform distribution over 4 options has maximum entropy ln(4)."""
 25:         belief = np.array([0.25, 0.25, 0.25, 0.25])
 26:         ent = entropy_of_distribution(belief)
 27:         # ln(4) ~ 1.3863
 28:         assert 1.35 < ent < 1.40, f"Uniform entropy {ent} not near ln(4)=1.3863"
 29: 
 30:     def test_entropy_peaked(self) -> None:
 31:         """Peaked distribution has low entropy."""
 32:         belief = np.array([0.9, 0.05, 0.03, 0.02])
 33:         ent = entropy_of_distribution(belief)
 34:         assert ent < 0.5, f"Peaked entropy {ent} should be < 0.5"
 35: 
 36:     def test_entropy_deterministic_no_nan(self) -> None:
 37:         """Deterministic distribution [1, 0, 0, 0] produces no NaN/inf."""
 38:         belief = np.array([1.0, 0.0, 0.0, 0.0])
 39:         ent = entropy_of_distribution(belief)
 40:         assert np.isfinite(ent), f"Entropy {ent} should be finite"
 41:         assert ent >= 0.0, f"Entropy {ent} should be non-negative"
 42: 
 43:     def test_entropy_deterministic_last(self) -> None:
 44:         """Deterministic distribution [0, 0, 0, 1] produces no NaN/inf."""
 45:         belief = np.array([0.0, 0.0, 0.0, 1.0])
 46:         ent = entropy_of_distribution(belief)
 47:         assert np.isfinite(ent), f"Entropy {ent} should be finite"
 48:         assert ent >= 0.0, f"Entropy {ent} should be non-negative"
 49: 
 50:     def test_entropy_binary(self) -> None:
 51:         """Binary uniform distribution has entropy ln(2)."""
 52:         belief = np.array([0.5, 0.5])
 53:         ent = entropy_of_distribution(belief)
 54:         assert abs(ent - np.log(2)) < 0.01, f"Binary entropy {ent} != ln(2)={np.log(2):.4f}"
 55: 
 56: 
 57: # ------------------------------------------------------------------ #
 58: # Tests for extract_belief_features
 59: # ------------------------------------------------------------------ #
 60: 
 61: 
 62: class TestExtractBeliefFeatures:
 63:     """Tests for belief feature vector extraction."""
 64: 
 65:     def test_feature_shape(self) -> None:
 66:         """Output shape is (K+6,) for K=4 belief vector."""
 67:         belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
 68:         features = extract_belief_features(belief, None, 0, 6)
 69:         assert features.shape == (10,), f"Expected (10,), got {features.shape}"
 70: 
 71:     def test_feature_shape_k3(self) -> None:
 72:         """Output shape adapts to K=3."""
 73:         belief = np.array([0.4, 0.3, 0.3], dtype=np.float32)
 74:         features = extract_belief_features(belief, None, 0, 5)
 75:         assert features.shape == (9,), f"Expected (9,), got {features.shape}"
 76: 
 77:     def test_feature_contents_belief_prefix(self) -> None:
 78:         """First K elements of feature vector are the raw belief."""
 79:         belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
 80:         features = extract_belief_features(belief, None, 2, 6)
 81:         np.testing.assert_array_almost_equal(
 82:             features[:4], belief, decimal=5,
 83:             err_msg="First K elements should match input belief",
 84:         )
 85: 
 86:     def test_derived_top_p(self) -> None:
 87:         """top_p is max(belief)."""
 88:         belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
 89:         features = extract_belief_features(belief, None, 2, 6)
 90:         assert abs(features[4] - 0.5) < 1e-5, f"top_p={features[4]}, expected 0.5"
 91: 
 92:     def test_derived_margin(self) -> None:
 93:         """margin is top_p - second_highest."""
 94:         belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
 95:         features = extract_belief_features(belief, None, 2, 6)
 96:         expected_margin = 0.5 - 0.3
 97:         assert abs(features[5] - expected_margin) < 1e-5, (
 98:             f"margin={features[5]}, expected {expected_margin}"
 99:         )
100: 
101:     def test_derived_entropy_in_range(self) -> None:
102:         """Entropy is in a reasonable range for a non-uniform distribution."""
103:         belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
104:         features = extract_belief_features(belief, None, 2, 6)
105:         ent = features[6]
106:         assert 0 < ent < np.log(4) + 0.01, f"Entropy {ent} out of range"
107: 
108:     def test_stability_none_prev(self) -> None:
109:         """Stability is 0.0 when prev_belief is None (first step)."""
110:         belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
111:         features = extract_belief_features(belief, None, 0, 6)
112:         assert features[7] == 0.0, f"Stability={features[7]}, expected 0.0 for first step"
113: 
114:     def test_stability_computation(self) -> None:
115:         """Stability tracks L1 distance between consecutive beliefs."""
116:         prev_belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
117:         belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
118:         features = extract_belief_features(belief, prev_belief, 1, 6)
119:         expected_stability = float(np.abs(belief - prev_belief).sum())
120:         assert abs(features[7] - expected_stability) < 1e-5, (
121:             f"Stability={features[7]}, expected {expected_stability}"
122:         )
123: 
124:     def test_progress(self) -> None:
125:         """progress = step_idx / total_steps."""
126:         belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
127:         features = extract_belief_features(belief, None, 3, 6)
128:         expected_progress = 3.0 / 6.0
129:         assert abs(features[8] - expected_progress) < 1e-5, (
130:             f"progress={features[8]}, expected {expected_progress}"
131:         )
132: 
133:     def test_clue_idx_norm(self) -> None:
134:         """clue_idx_norm = step_idx / (total_steps - 1)."""
135:         belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
136:         features = extract_belief_features(belief, None, 3, 6)
137:         expected_norm = 3.0 / 5.0
138:         assert abs(features[9] - expected_norm) < 1e-5, (
139:             f"clue_idx_norm={features[9]}, expected {expected_norm}"
140:         )
141: 
142:     def test_dtype_float32(self) -> None:
143:         """Output dtype is float32."""
144:         belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
145:         features = extract_belief_features(belief, None, 0, 6)
146:         assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
147: 
148:     def test_invalid_2d_belief_raises(self) -> None:
149:         """Passing a 2D belief array raises ValueError."""
150:         belief = np.array([[0.5, 0.5]], dtype=np.float32)
151:         with pytest.raises(ValueError, match="1D"):
152:             extract_belief_features(belief, None, 0, 1)
153: 
154: 
155: class TestPaddedBeliefFeatures:
156:     """Tests for extract_padded_belief_features."""
157: 
158:     def test_shape_equals_max_k_plus_6(self) -> None:
159:         from models.features import extract_padded_belief_features
160: 
161:         belief = np.array([0.5, 0.3, 0.2], dtype=np.float32)
162:         feats = extract_padded_belief_features(belief, None, 0, 6, max_K=8)
163:         assert feats.shape == (8 + 6,)
164: 
165:     def test_valid_slots_preserved(self) -> None:
166:         from models.features import extract_padded_belief_features
167: 
168:         belief = np.array([0.6, 0.3, 0.1], dtype=np.float32)
169:         feats = extract_padded_belief_features(belief, None, 0, 6, max_K=5)
170:         np.testing.assert_allclose(feats[:3], belief, atol=1e-7)
171: 
172:     def test_padded_slots_zero(self) -> None:
173:         from models.features import extract_padded_belief_features
174: 
175:         belief = np.array([0.5, 0.5], dtype=np.float32)
176:         feats = extract_padded_belief_features(belief, None, 0, 6, max_K=6)
177:         np.testing.assert_array_equal(feats[2:6], [0.0, 0.0, 0.0, 0.0])
178: 
179:     def test_extras_match_unpadded(self) -> None:
180:         from models.features import extract_padded_belief_features
181: 
182:         belief = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
183:         unpadded = extract_belief_features(belief, None, 2, 6)
184:         padded = extract_padded_belief_features(belief, None, 2, 6, max_K=4)
185:         np.testing.assert_allclose(unpadded[4:], padded[4:], atol=1e-7)
186: 
187:     def test_dtype(self) -> None:
188:         from models.features import extract_padded_belief_features
189: 
190:         belief = np.array([0.5, 0.5], dtype=np.float32)
191:         feats = extract_padded_belief_features(belief, None, 0, 4, max_K=4)
192:         assert feats.dtype == np.float32
````

## File: tests/test_hazard_pretrain.py
````python
 1: """Tests for the hazard pretraining bridge utilities and CLI guard."""
 2: 
 3: from __future__ import annotations
 4: 
 5: import argparse
 6: import importlib
 7: 
 8: import pytest
 9: import torch
10: 
11: 
12: def test_compute_survival_terms_simple_case() -> None:
13:     """compute_survival_terms returns expected survival and stop masses."""
14:     compute_survival_terms = importlib.import_module(
15:         "training.hazard_pretrain"
16:     ).compute_survival_terms
17: 
18:     stop_probs = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
19:     survival, stop_mass = compute_survival_terms(stop_probs)
20: 
21:     expected_survival = torch.tensor([[1.0, 0.8, 0.4]], dtype=torch.float32)
22:     expected_stop_mass = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
23:     assert torch.allclose(survival, expected_survival, atol=1e-6)
24:     assert torch.allclose(stop_mass, expected_stop_mass, atol=1e-6)
25: 
26: 
27: def test_hazard_expected_nll_loss_uses_terminal_penalty() -> None:
28:     """hazard_expected_nll_loss returns a scalar with beta_terminal applied."""
29:     hazard_expected_nll_loss = importlib.import_module(
30:         "training.hazard_pretrain"
31:     ).hazard_expected_nll_loss
32: 
33:     stop_probs = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
34:     nll_per_prefix = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
35: 
36:     loss = hazard_expected_nll_loss(
37:         stop_probs=stop_probs,
38:         nll_per_prefix=nll_per_prefix,
39:         beta_terminal=1.5,
40:     )
41: 
42:     assert loss.ndim == 0
43:     assert loss.item() == pytest.approx(1.6)
44: 
45: 
46: def test_hazard_pretrain_flag_raises_not_implemented() -> None:
47:     """CLI rejects hazard-pretrain until the training loop exists."""
48:     validate_args = importlib.import_module("scripts.train_t5_policy").validate_args
49: 
50:     args = argparse.Namespace(
51:         config="configs/t5_policy.yaml",
52:         smoke=False,
53:         skip_supervised=False,
54:         model_path=None,
55:         mc_path=None,
56:         ppo_iterations=None,
57:         hazard_pretrain=True,
58:         beta_terminal=1.0,
59:         freeze_answer_head=False,
60:     )
61: 
62:     with pytest.raises(NotImplementedError, match="Hazard pretraining loop not yet implemented"):
63:         validate_args(args)
````

## File: tests/test_mc_builder_topk.py
````python
  1: """Regression tests for top-M distractor ranking in MCBuilder._compute_rankings.
  2: 
  3: Validates that the argpartition-based top-M retrieval produces the same top
  4: distractors as a full argsort, truncates ranking lists correctly, degrades
  5: gracefully when N is small, and leaves category_random strategy unchanged.
  6: """
  7: 
  8: from __future__ import annotations
  9: 
 10: import numpy as np
 11: from sklearn.feature_extraction.text import TfidfVectorizer
 12: from sklearn.metrics.pairwise import cosine_similarity
 13: 
 14: from qb_data.mc_builder import MCBuilder
 15: 
 16: 
 17: # ---------------------------------------------------------------------------
 18: # Helpers
 19: # ---------------------------------------------------------------------------
 20: 
 21: def _make_synthetic_answers(n: int) -> tuple[list[str], dict[str, str]]:
 22:     """Create *n* synthetic answers with distinct TF-IDF profiles.
 23: 
 24:     Each answer is a short phrase and its profile is a sentence containing
 25:     unique vocabulary so TF-IDF can discriminate between them.
 26:     """
 27:     topics = [
 28:         ("George Washington", "first president commander revolutionary war continental army"),
 29:         ("Thomas Jefferson", "third president declaration independence Virginia Monticello"),
 30:         ("John Adams", "second president Massachusetts diplomat federalist"),
 31:         ("Benjamin Franklin", "inventor diplomat Philadelphia printing press electricity"),
 32:         ("Abraham Lincoln", "sixteenth president civil war emancipation slavery"),
 33:         ("Alexander Hamilton", "treasury secretary banking system federalist papers"),
 34:         ("James Madison", "bill rights constitution fourth president Virginia"),
 35:         ("Andrew Jackson", "military hero populist president battle New Orleans"),
 36:         ("Theodore Roosevelt", "progressive trust buster national parks rough riders"),
 37:         ("Ulysses Grant", "civil war general eighteenth president reconstruction"),
 38:         ("Woodrow Wilson", "world war one league nations progressive president"),
 39:         ("Franklin Roosevelt", "new deal world war two great depression fireside"),
 40:         ("Harry Truman", "atomic bomb cold war Korean conflict fair deal"),
 41:         ("Dwight Eisenhower", "supreme commander NATO interstate highway system"),
 42:         ("John Kennedy", "space race Cuban missile crisis new frontier"),
 43:         ("Lyndon Johnson", "great society civil rights Vietnam escalation"),
 44:         ("Richard Nixon", "detente China opening Watergate resignation"),
 45:         ("Ronald Reagan", "cold war end conservative revolution economic growth"),
 46:         ("Barack Obama", "affordable care act first African American president"),
 47:         ("Jimmy Carter", "Camp David accords energy crisis human rights"),
 48:     ]
 49:     answers = [t[0] for t in topics[:n]]
 50:     profiles = {t[0]: t[1] for t in topics[:n]}
 51:     return answers, profiles
 52: 
 53: 
 54: def _full_sort_rankings(
 55:     answers: list[str], profiles: dict[str, str]
 56: ) -> dict[str, list[str]]:
 57:     """Compute rankings via full argsort (reference implementation)."""
 58:     docs = [profiles[a] for a in answers]
 59:     answer_idx = {a: i for i, a in enumerate(answers)}
 60:     vectorizer = TfidfVectorizer(stop_words="english")
 61:     matrix = vectorizer.fit_transform(docs)
 62:     sim = cosine_similarity(matrix, matrix)
 63:     rankings: dict[str, list[str]] = {}
 64:     for answer in answers:
 65:         idx = answer_idx[answer]
 66:         order = np.argsort(-sim[idx]).tolist()
 67:         rankings[answer] = [answers[i] for i in order if answers[i] != answer]
 68:     return rankings
 69: 
 70: 
 71: # ---------------------------------------------------------------------------
 72: # Tests
 73: # ---------------------------------------------------------------------------
 74: 
 75: class TestTopMRanking:
 76:     """Tests for top-M argpartition ranking in MCBuilder._compute_rankings."""
 77: 
 78:     def test_top_m_truncation(self) -> None:
 79:         """Rankings should have length <= min(M, N-1)."""
 80:         answers, profiles = _make_synthetic_answers(20)
 81:         builder = MCBuilder(K=4, strategy="tfidf_profile")
 82:         categories: dict[str, str] = {}
 83: 
 84:         rankings = builder._compute_rankings(answers, profiles, categories)
 85: 
 86:         M = min(max(5 * 4, 30), len(answers) - 1)  # min(30, 19) = 19
 87:         for answer, ranked in rankings.items():
 88:             assert len(ranked) <= min(M, len(answers) - 1), (
 89:                 f"Answer '{answer}' has {len(ranked)} distractors, "
 90:                 f"expected <= {min(M, len(answers) - 1)}"
 91:             )
 92: 
 93:     def test_order_preservation(self) -> None:
 94:         """Top-3 distractors must match the full-sort reference."""
 95:         answers, profiles = _make_synthetic_answers(20)
 96:         builder = MCBuilder(K=4, strategy="tfidf_profile")
 97:         categories: dict[str, str] = {}
 98: 
 99:         rankings = builder._compute_rankings(answers, profiles, categories)
100:         reference = _full_sort_rankings(answers, profiles)
101: 
102:         for answer in answers:
103:             actual_top3 = rankings[answer][:3]
104:             expected_top3 = reference[answer][:3]
105:             assert actual_top3 == expected_top3, (
106:                 f"Answer '{answer}': top-3 mismatch.\n"
107:                 f"  actual:   {actual_top3}\n"
108:                 f"  expected: {expected_top3}"
109:             )
110: 
111:     def test_small_n_graceful(self) -> None:
112:         """With N=5, rankings should have length N-1=4 without error."""
113:         answers, profiles = _make_synthetic_answers(5)
114:         builder = MCBuilder(K=4, strategy="tfidf_profile")
115:         categories: dict[str, str] = {}
116: 
117:         rankings = builder._compute_rankings(answers, profiles, categories)
118: 
119:         for answer, ranked in rankings.items():
120:             assert len(ranked) == 4, (
121:                 f"Answer '{answer}' has {len(ranked)} distractors, expected 4"
122:             )
123: 
124:     def test_category_random_unaffected(self) -> None:
125:         """category_random strategy should not use argpartition path."""
126:         answers, profiles = _make_synthetic_answers(10)
127:         categories = {a: "History" for a in answers}
128:         builder = MCBuilder(K=4, strategy="category_random")
129: 
130:         rankings = builder._compute_rankings(answers, profiles, categories)
131: 
132:         for answer, ranked in rankings.items():
133:             # All same-category peers (minus self) should be present
134:             assert set(ranked) == set(a for a in answers if a != answer), (
135:                 f"Answer '{answer}': category_random should include all peers"
136:             )
````

## File: tests/test_mc_builder_variable_k.py
````python
 1: """Tests for variable-K MC question construction."""
 2: 
 3: from __future__ import annotations
 4: 
 5: import pytest
 6: 
 7: from qb_data.answer_profiles import AnswerProfileBuilder
 8: from qb_data.data_loader import TossupQuestion
 9: from qb_data.mc_builder import MCBuilder
10: 
11: 
12: def _make_questions(n: int = 20, n_unique_answers: int | None = None) -> list[TossupQuestion]:
13:     n_ans = n_unique_answers if n_unique_answers is not None else n
14:     questions = []
15:     for i in range(n):
16:         tokens = [f"word{i}_{j}" for j in range(10)]
17:         questions.append(
18:             TossupQuestion(
19:                 qid=f"q{i:03d}",
20:                 question=" ".join(tokens),
21:                 tokens=tokens,
22:                 answer_primary=f"Answer_{i % n_ans}",
23:                 clean_answers=[f"Answer_{i % n_ans}"],
24:                 run_indices=[2, 5, 9],
25:                 human_buzz_positions=[],
26:                 category=["History", "Science"][i % 2],
27:                 cumulative_prefixes=[
28:                     " ".join(tokens[:3]),
29:                     " ".join(tokens[:6]),
30:                     " ".join(tokens),
31:                 ],
32:             )
33:         )
34:     return questions
35: 
36: 
37: class TestFixedKUnchanged:
38:     def test_fixed_k_default(self) -> None:
39:         qs = _make_questions(20)
40:         builder = MCBuilder(K=4, strategy="category_random", random_seed=42)
41:         profile = AnswerProfileBuilder()
42:         mc = builder.build(qs, profile)
43:         for q in mc:
44:             assert len(q.options) == 4
45: 
46:     def test_variable_k_false_is_fixed(self) -> None:
47:         qs = _make_questions(20)
48:         builder = MCBuilder(K=4, strategy="category_random", random_seed=42, variable_K=False)
49:         profile = AnswerProfileBuilder()
50:         mc = builder.build(qs, profile)
51:         for q in mc:
52:             assert len(q.options) == 4
53: 
54: 
55: class TestVariableK:
56:     def test_variable_k_yields_mixed(self) -> None:
57:         qs = _make_questions(40)
58:         builder = MCBuilder(
59:             K=6, strategy="category_random", random_seed=42,
60:             variable_K=True, min_K=2, max_K=6,
61:         )
62:         profile = AnswerProfileBuilder()
63:         mc = builder.build(qs, profile)
64:         option_counts = {len(q.options) for q in mc}
65:         assert len(option_counts) > 1, f"Expected mixed K, got only {option_counts}"
66:         for q in mc:
67:             assert 2 <= len(q.options) <= 6
68: 
69:     def test_gold_index_valid(self) -> None:
70:         qs = _make_questions(30)
71:         builder = MCBuilder(
72:             K=5, strategy="category_random", random_seed=42,
73:             variable_K=True, min_K=2, max_K=5,
74:         )
75:         profile = AnswerProfileBuilder()
76:         mc = builder.build(qs, profile)
77:         for q in mc:
78:             assert 0 <= q.gold_index < len(q.options)
79:             assert q.options[q.gold_index] in q.clean_answers or \
80:                 q.option_answer_primary[q.gold_index] == q.answer_primary
81: 
82:     def test_profiles_match_options(self) -> None:
83:         qs = _make_questions(20)
84:         builder = MCBuilder(
85:             K=5, strategy="category_random", random_seed=42,
86:             variable_K=True, min_K=3, max_K=5,
87:         )
88:         profile = AnswerProfileBuilder()
89:         mc = builder.build(qs, profile)
90:         for q in mc:
91:             assert len(q.option_profiles) == len(q.options)
92:             assert len(q.option_answer_primary) == len(q.options)
````

## File: tests/test_metrics.py
````python
  1: """Unit tests for evaluation metrics.
  2: 
  3: Tests edge cases for system_score (S_q), calibration metrics (ECE, Brier),
  4: and per-category accuracy grouping.
  5: """
  6: 
  7: import pytest
  8: 
  9: from evaluation.metrics import (
 10:     brier_score,
 11:     calibration_at_buzz,
 12:     expected_calibration_error,
 13:     expected_wins_score,
 14:     per_category_accuracy,
 15:     summarize_buzz_metrics,
 16:     system_score,
 17: )
 18: 
 19: 
 20: # ---------------------------------------------------------------------------
 21: # system_score (S_q) edge cases
 22: # ---------------------------------------------------------------------------
 23: 
 24: 
 25: def test_system_score_empty_trace():
 26:     """S_q should return 0.0 for empty traces."""
 27:     assert system_score([], []) == 0.0
 28: 
 29: 
 30: def test_system_score_all_zero_confidence():
 31:     """S_q should return 0.0 when agent never considers buzzing."""
 32:     c_trace = [0.0, 0.0, 0.0]
 33:     g_trace = [1.0, 1.0, 1.0]  # All correct but agent doesn't buzz
 34:     assert system_score(c_trace, g_trace) == 0.0
 35: 
 36: 
 37: def test_system_score_all_correct_immediate_buzz():
 38:     """S_q should equal first g_trace value when agent buzzes immediately."""
 39:     c_trace = [1.0, 0.0, 0.0]  # Buzz on step 0
 40:     g_trace = [1.0, 1.0, 1.0]
 41:     expected = 1.0 * 1.0  # b_0 = c_0 * 1.0 = 1.0, survival after = 0
 42:     assert abs(system_score(c_trace, g_trace) - expected) < 1e-9
 43: 
 44: 
 45: def test_system_score_gradual_confidence():
 46:     """S_q should accumulate survival-weighted correctness."""
 47:     c_trace = [0.3, 0.5, 1.0]
 48:     g_trace = [0.0, 0.0, 1.0]  # Only correct at final step
 49:     # b_0 = 0.3 * 1.0 = 0.3, survival = 0.7
 50:     # b_1 = 0.5 * 0.7 = 0.35, survival = 0.7 * 0.5 = 0.35
 51:     # b_2 = 1.0 * 0.35 = 0.35
 52:     # S_q = 0.3*0 + 0.35*0 + 0.35*1 = 0.35
 53:     expected = 0.35
 54:     assert abs(system_score(c_trace, g_trace) - expected) < 1e-9
 55: 
 56: 
 57: def test_system_score_single_step():
 58:     """S_q should work for single-step episodes."""
 59:     c_trace = [1.0]
 60:     g_trace = [1.0]
 61:     assert abs(system_score(c_trace, g_trace) - 1.0) < 1e-9
 62: 
 63:     c_trace = [0.5]
 64:     g_trace = [1.0]
 65:     assert abs(system_score(c_trace, g_trace) - 0.5) < 1e-9
 66: 
 67: 
 68: def test_system_score_never_correct():
 69:     """S_q should return 0.0 when g_trace is all zeros."""
 70:     c_trace = [0.5, 0.5, 0.5]
 71:     g_trace = [0.0, 0.0, 0.0]
 72:     assert system_score(c_trace, g_trace) == 0.0
 73: 
 74: 
 75: # ---------------------------------------------------------------------------
 76: # Expected Calibration Error (ECE)
 77: # ---------------------------------------------------------------------------
 78: 
 79: 
 80: def test_expected_calibration_error_perfect():
 81:     """ECE should be near 0.0 for perfectly calibrated predictions."""
 82:     # 70% confidence with 70% accuracy
 83:     confidences = [0.7] * 10
 84:     outcomes = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
 85:     ece = expected_calibration_error(confidences, outcomes, n_bins=10)
 86:     assert ece < 0.01  # Near zero for perfect calibration
 87: 
 88: 
 89: def test_expected_calibration_error_empty():
 90:     """ECE should return 0.0 for empty inputs."""
 91:     assert expected_calibration_error([], []) == 0.0
 92: 
 93: 
 94: # ---------------------------------------------------------------------------
 95: # Brier Score
 96: # ---------------------------------------------------------------------------
 97: 
 98: 
 99: def test_brier_score_perfect():
100:     """Brier score should be 0.0 for perfect predictions."""
101:     confidences = [1.0, 1.0, 0.0, 0.0]
102:     outcomes = [1, 1, 0, 0]
103:     bs = brier_score(confidences, outcomes)
104:     assert bs == 0.0
105: 
106: 
107: def test_brier_score_worst():
108:     """Brier score should be 1.0 for worst-case predictions."""
109:     confidences = [0.0, 0.0, 1.0, 1.0]
110:     outcomes = [1, 1, 0, 0]
111:     bs = brier_score(confidences, outcomes)
112:     assert abs(bs - 1.0) < 1e-9
113: 
114: 
115: def test_brier_score_empty():
116:     """Brier score should return 0.0 for empty inputs."""
117:     assert brier_score([], []) == 0.0
118: 
119: 
120: # ---------------------------------------------------------------------------
121: # summarize_buzz_metrics
122: # ---------------------------------------------------------------------------
123: 
124: 
125: def test_summarize_buzz_metrics_empty():
126:     """summarize_buzz_metrics should handle empty results."""
127:     result = summarize_buzz_metrics([])
128:     assert result["n"] == 0.0
129:     assert result["buzz_accuracy"] == 0.0
130: 
131: 
132: def test_summarize_buzz_metrics_basic():
133:     """summarize_buzz_metrics should compute correct aggregates."""
134:     results = [
135:         {
136:             "qid": "q1",
137:             "correct": True,
138:             "buzz_step": 2,
139:             "c_trace": [0.0, 0.0, 1.0],
140:             "g_trace": [0.0, 0.0, 1.0],
141:             "reward_like": 0.8,
142:         },
143:         {
144:             "qid": "q2",
145:             "correct": False,
146:             "buzz_step": 1,
147:             "c_trace": [0.0, 1.0],
148:             "g_trace": [0.0, 0.0],
149:             "reward_like": -0.1,
150:         },
151:     ]
152:     summary = summarize_buzz_metrics(results)
153:     assert summary["n"] == 2.0
154:     assert abs(summary["buzz_accuracy"] - 0.5) < 1e-9
155:     assert abs(summary["mean_buzz_step"] - 1.5) < 1e-9
156: 
157: 
158: # ---------------------------------------------------------------------------
159: # per_category_accuracy
160: # ---------------------------------------------------------------------------
161: 
162: 
163: def test_per_category_accuracy_basic():
164:     """per_category_accuracy should group results by question category."""
165:     results = [
166:         {
167:             "qid": "q1",
168:             "correct": True,
169:             "buzz_step": 2,
170:             "c_trace": [0.0, 0.0, 1.0],
171:             "g_trace": [0.0, 0.0, 1.0],
172:             "reward_like": 0.8,
173:         },
174:         {
175:             "qid": "q2",
176:             "correct": False,
177:             "buzz_step": 1,
178:             "c_trace": [0.0, 1.0],
179:             "g_trace": [0.0, 0.0],
180:             "reward_like": -0.1,
181:         },
182:         {
183:             "qid": "q3",
184:             "correct": True,
185:             "buzz_step": 3,
186:             "c_trace": [0.0, 0.0, 0.0, 1.0],
187:             "g_trace": [0.0, 0.0, 0.0, 1.0],
188:             "reward_like": 0.7,
189:         },
190:     ]
191:     questions = [
192:         {"qid": "q1", "category": "History"},
193:         {"qid": "q2", "category": "Science"},
194:         {"qid": "q3", "category": "History"},
195:     ]
196:     cat_metrics = per_category_accuracy(results, questions)
197:     assert "History" in cat_metrics
198:     assert "Science" in cat_metrics
199:     assert cat_metrics["History"]["n"] == 2.0
200:     assert cat_metrics["History"]["buzz_accuracy"] == 1.0
201:     assert cat_metrics["Science"]["n"] == 1.0
202:     assert cat_metrics["Science"]["buzz_accuracy"] == 0.0
203: 
204: 
205: def test_per_category_accuracy_missing_category():
206:     """per_category_accuracy should default missing categories to 'unknown'."""
207:     results = [
208:         {
209:             "qid": "q1",
210:             "correct": True,
211:             "buzz_step": 0,
212:             "c_trace": [1.0],
213:             "g_trace": [1.0],
214:             "reward_like": 1.0,
215:         },
216:     ]
217:     questions = [
218:         {"qid": "q1", "category": ""},
219:     ]
220:     cat_metrics = per_category_accuracy(results, questions)
221:     assert "unknown" in cat_metrics
222:     assert cat_metrics["unknown"]["n"] == 1.0
223: 
224: 
225: def test_per_category_accuracy_none_category():
226:     """per_category_accuracy should handle None category."""
227:     results = [
228:         {
229:             "qid": "q1",
230:             "correct": True,
231:             "buzz_step": 0,
232:             "c_trace": [1.0],
233:             "g_trace": [1.0],
234:             "reward_like": 1.0,
235:         },
236:     ]
237:     questions = [
238:         {"qid": "q1", "category": None},
239:     ]
240:     cat_metrics = per_category_accuracy(results, questions)
241:     assert "unknown" in cat_metrics
242: 
243: 
244: def test_per_category_accuracy_unmatched_qid():
245:     """Results with qids not in questions should group to 'unknown'."""
246:     results = [
247:         {
248:             "qid": "q_orphan",
249:             "correct": False,
250:             "buzz_step": 0,
251:             "c_trace": [1.0],
252:             "g_trace": [0.0],
253:             "reward_like": -0.1,
254:         },
255:     ]
256:     questions = [
257:         {"qid": "q1", "category": "History"},
258:     ]
259:     cat_metrics = per_category_accuracy(results, questions)
260:     assert "unknown" in cat_metrics
261:     assert cat_metrics["unknown"]["n"] == 1.0
262: 
263: 
264: # ---------------------------------------------------------------------------
265: # calibration_at_buzz — uses top_p_trace, not g_trace
266: # ---------------------------------------------------------------------------
267: 
268: 
269: def test_calibration_at_buzz_uses_top_p_trace():
270:     """calibration_at_buzz must use top_p_trace (belief prob), not g_trace (binary)."""
271:     results = [
272:         {
273:             "qid": "q1",
274:             "correct": True,
275:             "buzz_step": 2,
276:             "c_trace": [0.1, 0.3, 0.9],
277:             "g_trace": [0.0, 0.0, 1.0],
278:             "top_p_trace": [0.3, 0.5, 0.8],
279:         },
280:         {
281:             "qid": "q2",
282:             "correct": False,
283:             "buzz_step": 1,
284:             "c_trace": [0.2, 0.7],
285:             "g_trace": [0.0, 0.0],
286:             "top_p_trace": [0.4, 0.6],
287:         },
288:     ]
289:     cal = calibration_at_buzz(results)
290:     assert cal["n_calibration"] == 2.0
291:     # Confidence from top_p_trace at buzz_step:
292:     # q1: top_p_trace[2] = 0.8, q2: top_p_trace[1] = 0.6
293:     # Brier = ((0.8-1)^2 + (0.6-0)^2)/2 = (0.04+0.36)/2 = 0.2
294:     assert abs(cal["brier"] - 0.2) < 1e-9
295: 
296: 
297: def test_calibration_at_buzz_falls_back_to_c_trace():
298:     """When top_p_trace is absent, calibration should fall back to c_trace."""
299:     results = [
300:         {
301:             "qid": "q1",
302:             "correct": True,
303:             "buzz_step": 0,
304:             "c_trace": [0.7],
305:             "g_trace": [1.0],
306:         },
307:     ]
308:     cal = calibration_at_buzz(results)
309:     assert cal["n_calibration"] == 1.0
310:     assert abs(cal["brier"] - (0.7 - 1.0) ** 2) < 1e-9
311: 
312: 
313: def test_calibration_at_buzz_empty():
314:     """calibration_at_buzz should return zeros for empty input."""
315:     cal = calibration_at_buzz([])
316:     assert cal["ece"] == 0.0
317:     assert cal["brier"] == 0.0
318:     assert cal["n_calibration"] == 0.0
319: 
320: 
321: def test_calibration_at_buzz_binary_g_trace_not_used():
322:     """Regression: binary g_trace must NOT be used as confidence.
323: 
324:     If g_trace (binary 0/1) were used, Brier for a correct episode with
325:     g_trace=[1.0] would be 0.0 regardless of actual confidence.  With
326:     top_p_trace=[0.5] and correct=True, Brier = (0.5-1)^2 = 0.25.
327:     """
328:     results = [
329:         {
330:             "qid": "q1",
331:             "correct": True,
332:             "buzz_step": 0,
333:             "c_trace": [0.9],
334:             "g_trace": [1.0],
335:             "top_p_trace": [0.5],
336:         },
337:     ]
338:     cal = calibration_at_buzz(results)
339:     assert abs(cal["brier"] - 0.25) < 1e-9
340: 
341: 
342: # ---------------------------------------------------------------------------
343: # expected_wins_score
344: # ---------------------------------------------------------------------------
345: 
346: 
347: def test_expected_wins_score_binary_g_trace():
348:     """Hand-worked EW with baseline-style binary g_trace.
349: 
350:     Agent buzzes immediately (c=[1.0]), correct (g=[1.0]),
351:     opponent survival=0.8 → EW = 1.0 * [0.8*10 + 0.2*0] = 8.0
352:     """
353:     ew = expected_wins_score(
354:         c_trace=[1.0],
355:         g_trace=[1.0],
356:         opponent_survival_trace=[0.8],
357:         reward_correct=10.0,
358:         reward_incorrect=-5.0,
359:         opponent_expected_value=0.0,
360:     )
361:     assert abs(ew - 8.0) < 1e-9
362: 
363: 
364: def test_expected_wins_score_fractional_g_trace():
365:     """Hand-worked EW with PPO-style fractional g_trace.
366: 
367:     c=[1.0], g=[0.6], S=[0.8]
368:     V_self = 0.6*10 + 0.4*(-5) = 4.0
369:     V = 0.8*4.0 + 0.2*0 = 3.2
370:     EW = 1.0 * 3.2 = 3.2
371:     """
372:     ew = expected_wins_score(
373:         c_trace=[1.0],
374:         g_trace=[0.6],
375:         opponent_survival_trace=[0.8],
376:         reward_correct=10.0,
377:         reward_incorrect=-5.0,
378:         opponent_expected_value=0.0,
379:     )
380:     assert abs(ew - 3.2) < 1e-9
381: 
382: 
383: def test_expected_wins_score_empty():
384:     assert expected_wins_score([], [], []) == 0.0
385: 
386: 
387: def test_expected_wins_does_not_regress_system_score():
388:     """system_score must remain unchanged by EW addition."""
389:     c = [0.3, 0.5, 1.0]
390:     g = [0.0, 0.0, 1.0]
391:     expected = 0.35
392:     assert abs(system_score(c, g) - expected) < 1e-9
````

## File: tests/test_opponent_models.py
````python
  1: """Tests for qb_env/opponent_models.py."""
  2: 
  3: from __future__ import annotations
  4: 
  5: import pytest
  6: 
  7: from qb_data.mc_builder import MCQuestion
  8: from qb_env.opponent_models import (
  9:     EmpiricalHistogramOpponentModel,
 10:     LogisticOpponentModel,
 11:     build_opponent_model_from_config,
 12: )
 13: 
 14: 
 15: def _make_question(
 16:     human_buzz_positions=None,
 17:     num_steps: int = 6,
 18: ) -> MCQuestion:
 19:     tokens = [f"t{i}" for i in range(num_steps * 2)]
 20:     run_indices = list(range(0, num_steps * 2, 2))
 21:     prefixes = [" ".join(tokens[: ri + 1]) for ri in run_indices]
 22:     return MCQuestion(
 23:         qid="q_test",
 24:         question=" ".join(tokens),
 25:         tokens=tokens,
 26:         answer_primary="Answer A",
 27:         clean_answers=["Answer A"],
 28:         run_indices=run_indices,
 29:         human_buzz_positions=human_buzz_positions or [],
 30:         category="Test",
 31:         cumulative_prefixes=prefixes,
 32:         options=["Answer A", "Answer B", "Answer C", "Answer D"],
 33:         gold_index=0,
 34:         option_profiles=["prof_a", "prof_b", "prof_c", "prof_d"],
 35:         option_answer_primary=["Answer A", "Answer B", "Answer C", "Answer D"],
 36:         distractor_strategy="test",
 37:     )
 38: 
 39: 
 40: class TestLogisticOpponentModel:
 41:     def test_monotonicity(self) -> None:
 42:         model = LogisticOpponentModel(midpoint=0.5, steepness=6.0)
 43:         q = _make_question(num_steps=10)
 44:         probs = [model.prob_buzzed_before_step(q, t) for t in range(10)]
 45:         for i in range(1, len(probs)):
 46:             assert probs[i] >= probs[i - 1] - 1e-12
 47: 
 48:     def test_range_01(self) -> None:
 49:         model = LogisticOpponentModel()
 50:         q = _make_question(num_steps=20)
 51:         for t in range(20):
 52:             p = model.prob_buzzed_before_step(q, t)
 53:             assert 0.0 <= p <= 1.0
 54: 
 55:     def test_survive_complement(self) -> None:
 56:         model = LogisticOpponentModel()
 57:         q = _make_question(num_steps=10)
 58:         for t in range(10):
 59:             assert abs(
 60:                 model.prob_buzzed_before_step(q, t)
 61:                 + model.prob_survive_to_step(q, t)
 62:                 - 1.0
 63:             ) < 1e-12
 64: 
 65:     def test_step_zero_near_zero(self) -> None:
 66:         model = LogisticOpponentModel(midpoint=0.6, steepness=6.0)
 67:         q = _make_question(num_steps=10)
 68:         assert model.prob_buzzed_before_step(q, 0) < 0.1
 69: 
 70: 
 71: class TestEmpiricalHistogramOpponentModel:
 72:     def test_cumulative_from_positions(self) -> None:
 73:         q = _make_question(human_buzz_positions=[(2, 3), (4, 7)], num_steps=6)
 74:         model = EmpiricalHistogramOpponentModel()
 75:         p_at_3 = model.prob_buzzed_before_step(q, 1)
 76:         p_at_5 = model.prob_buzzed_before_step(q, 2)
 77:         assert p_at_5 >= p_at_3
 78: 
 79:     def test_fallback_when_no_data(self) -> None:
 80:         q = _make_question(human_buzz_positions=[], num_steps=10)
 81:         model = EmpiricalHistogramOpponentModel()
 82:         p = model.prob_buzzed_before_step(q, 5)
 83:         assert 0.0 <= p <= 1.0
 84: 
 85:     def test_global_fallback(self) -> None:
 86:         q = _make_question(human_buzz_positions=[], num_steps=6)
 87:         model = EmpiricalHistogramOpponentModel(
 88:             global_positions=[(2, 5), (4, 5)]
 89:         )
 90:         p = model.prob_buzzed_before_step(q, 2)
 91:         assert p > 0.0
 92: 
 93: 
 94: class TestBuildOpponentModelFromConfig:
 95:     def test_none_when_disabled(self) -> None:
 96:         cfg = {"environment": {"opponent_buzz_model": {"type": "none"}}}
 97:         assert build_opponent_model_from_config(config=cfg) is None
 98: 
 99:     def test_none_when_missing(self) -> None:
100:         assert build_opponent_model_from_config(config={}) is None
101:         assert build_opponent_model_from_config(config=None) is None
102: 
103:     def test_logistic(self) -> None:
104:         cfg = {"environment": {"opponent_buzz_model": {"type": "logistic", "midpoint": 0.4}}}
105:         model = build_opponent_model_from_config(config=cfg)
106:         assert isinstance(model, LogisticOpponentModel)
107:         assert model.midpoint == 0.4
108: 
109:     def test_empirical(self) -> None:
110:         q = _make_question(human_buzz_positions=[(2, 5)])
111:         cfg = {"environment": {"opponent_buzz_model": {"type": "empirical"}}}
112:         model = build_opponent_model_from_config(questions=[q], config=cfg)
113:         assert isinstance(model, EmpiricalHistogramOpponentModel)
````

## File: tests/test_t5_policy.py
````python
  1: """Unit tests for T5PolicyModel and PolicyHead.
  2: 
  3: Tests cover PolicyHead architecture, T5PolicyModel forward pass, action
  4: decomposition, tokenization, mean pooling, and checkpoint I/O.
  5: 
  6: Uses t5-small (60M params) for speed -- tests complete in <30 seconds.
  7: The model fixture is module-scoped to load t5-small only once.
  8: """
  9: 
 10: from __future__ import annotations
 11: 
 12: import os
 13: import tempfile
 14: 
 15: import pytest
 16: import torch
 17: 
 18: from models.t5_policy import PolicyHead, T5PolicyModel
 19: 
 20: 
 21: # ---------------------------------------------------------------------------
 22: # Fixtures
 23: # ---------------------------------------------------------------------------
 24: 
 25: 
 26: @pytest.fixture(scope="module")
 27: def t5_small_config() -> dict:
 28:     """Return a minimal config dict for T5PolicyModel with t5-small."""
 29:     return {
 30:         "model_name": "t5-small",
 31:         "device": "cpu",
 32:         "max_input_length": 128,
 33:         "num_choices": 4,
 34:     }
 35: 
 36: 
 37: @pytest.fixture(scope="module")
 38: def t5_small_model(t5_small_config):
 39:     """Load T5PolicyModel with t5-small once per test module."""
 40:     model = T5PolicyModel(t5_small_config)
 41:     model.eval()
 42:     return model
 43: 
 44: 
 45: @pytest.fixture
 46: def sample_texts() -> list[str]:
 47:     """Return sample text inputs in quiz bowl format."""
 48:     return [
 49:         "CLUES: Who was the first president | CHOICES: (1) Washington (2) Jefferson (3) Adams (4) Franklin",
 50:         "CLUES: This element has atomic number 1 | CHOICES: (1) Hydrogen (2) Helium (3) Lithium (4) Carbon",
 51:     ]
 52: 
 53: 
 54: # ---------------------------------------------------------------------------
 55: # PolicyHead Tests
 56: # ---------------------------------------------------------------------------
 57: 
 58: 
 59: class TestPolicyHead:
 60:     """Tests for PolicyHead class."""
 61: 
 62:     def test_policy_head_forward(self):
 63:         """PolicyHead returns 3 tensors with correct shapes [B,2], [B,K], [B,1]."""
 64:         batch_size = 4
 65:         hidden_size = 512
 66:         num_choices = 4
 67: 
 68:         head = PolicyHead(hidden_size=hidden_size, num_choices=num_choices)
 69:         x = torch.randn(batch_size, hidden_size)
 70: 
 71:         wait_logits, answer_logits, values = head(x)
 72: 
 73:         assert wait_logits.shape == (batch_size, 2)
 74:         assert answer_logits.shape == (batch_size, num_choices)
 75:         assert values.shape == (batch_size, 1)
 76: 
 77:     def test_policy_head_different_num_choices(self):
 78:         """PolicyHead handles non-default num_choices."""
 79:         head = PolicyHead(hidden_size=256, num_choices=6)
 80:         x = torch.randn(2, 256)
 81: 
 82:         wait_logits, answer_logits, values = head(x)
 83: 
 84:         assert wait_logits.shape == (2, 2)
 85:         assert answer_logits.shape == (2, 6)
 86:         assert values.shape == (2, 1)
 87: 
 88:     def test_policy_head_dropout(self):
 89:         """Dropout layers exist and affect output in training mode."""
 90:         head = PolicyHead(hidden_size=128, num_choices=4)
 91:         head.train()  # Enable dropout
 92: 
 93:         x = torch.randn(8, 128)
 94: 
 95:         # Run forward twice in training mode; outputs should differ with high probability
 96:         out1 = head(x)[0]
 97:         out2 = head(x)[0]
 98: 
 99:         # Not strictly guaranteed but extremely likely with 8 samples and dropout
100:         # Use eval mode comparison for determinism
101:         head.eval()
102:         out3 = head(x)[0]
103:         out4 = head(x)[0]
104:         assert torch.allclose(out3, out4), "Eval mode should be deterministic"
105: 
106:     def test_policy_head_single_sample(self):
107:         """PolicyHead works with batch_size=1."""
108:         head = PolicyHead(hidden_size=512, num_choices=4)
109:         x = torch.randn(1, 512)
110: 
111:         wait_logits, answer_logits, values = head(x)
112: 
113:         assert wait_logits.shape == (1, 2)
114:         assert answer_logits.shape == (1, 4)
115:         assert values.shape == (1, 1)
116: 
117: 
118: # ---------------------------------------------------------------------------
119: # T5PolicyModel Tests
120: # ---------------------------------------------------------------------------
121: 
122: 
123: class TestT5PolicyModel:
124:     """Tests for T5PolicyModel class."""
125: 
126:     def test_t5_policy_init(self, t5_small_model):
127:         """T5PolicyModel initializes without errors and has correct structure."""
128:         model = t5_small_model
129: 
130:         assert hasattr(model, "encoder")
131:         assert hasattr(model, "tokenizer")
132:         assert hasattr(model, "policy_head")
133:         assert isinstance(model.policy_head, PolicyHead)
134: 
135:     def test_t5_policy_forward(self, t5_small_model, sample_texts):
136:         """Forward pass returns correct shapes for text inputs."""
137:         model = t5_small_model
138:         wait_logits, answer_logits, values = model(sample_texts)
139: 
140:         batch_size = len(sample_texts)
141:         assert wait_logits.shape == (batch_size, 2)
142:         assert answer_logits.shape == (batch_size, 4)
143:         assert values.shape == (batch_size, 1)
144: 
145:     def test_t5_policy_forward_no_value(self, t5_small_model, sample_texts):
146:         """Forward pass with return_value=False returns None for values."""
147:         model = t5_small_model
148:         wait_logits, answer_logits, values = model(sample_texts, return_value=False)
149: 
150:         assert values is None
151:         assert wait_logits.shape[0] == len(sample_texts)
152: 
153:     def test_encode_input(self, t5_small_model, sample_texts):
154:         """Tokenization produces input_ids and attention_mask with correct device."""
155:         model = t5_small_model
156:         encoding = model.encode_input(sample_texts)
157: 
158:         assert "input_ids" in encoding
159:         assert "attention_mask" in encoding
160:         assert encoding["input_ids"].shape[0] == len(sample_texts)
161:         assert encoding["attention_mask"].shape == encoding["input_ids"].shape
162:         assert encoding["input_ids"].device == model.device
163: 
164:     def test_encode_input_padding(self, t5_small_model):
165:         """Tokenization handles inputs of different lengths with padding."""
166:         model = t5_small_model
167:         texts = ["short", "this is a much longer text input with more tokens"]
168:         encoding = model.encode_input(texts)
169: 
170:         # Both should have same seq_len after padding
171:         assert encoding["input_ids"].shape[0] == 2
172:         # Second text should have more non-padding tokens
173:         mask_sums = encoding["attention_mask"].sum(dim=1)
174:         assert mask_sums[1] > mask_sums[0]
175: 
176:     def test_mean_pooling(self, t5_small_model):
177:         """Mean pooling respects attention mask (padded tokens have zero contribution)."""
178:         model = t5_small_model
179: 
180:         # Create a simple case: two identical sentences, one with extra padding
181:         texts = ["hello world"]
182:         encoding = model.encode_input(texts)
183: 
184:         pooled = model.get_encoder_output(
185:             encoding["input_ids"], encoding["attention_mask"]
186:         )
187: 
188:         # Output should be [1, hidden_size]
189:         assert pooled.shape == (1, model.encoder.config.d_model)
190:         assert not torch.isnan(pooled).any()
191:         assert not torch.isinf(pooled).any()
192: 
193: 
194: # ---------------------------------------------------------------------------
195: # Action Decomposition Tests
196: # ---------------------------------------------------------------------------
197: 
198: 
199: class TestActionDecomposition:
200:     """Tests for action decomposition in select_action and get_action_log_probs."""
201: 
202:     def test_action_decomposition_wait(self, t5_small_model, sample_texts):
203:         """action=0 decomposes to wait=0 in get_action_log_probs."""
204:         model = t5_small_model
205:         encoding = model.encode_input(sample_texts)
206: 
207:         # WAIT action
208:         actions = torch.zeros(len(sample_texts), dtype=torch.long, device=model.device)
209:         log_probs, entropy, values = model.get_action_log_probs(
210:             encoding["input_ids"], encoding["attention_mask"], actions
211:         )
212: 
213:         assert log_probs.shape == (len(sample_texts),)
214:         assert entropy.shape == (len(sample_texts),)
215:         assert values.shape == (len(sample_texts),)
216:         # Log probs should be negative
217:         assert (log_probs <= 0).all()
218:         # Entropy should be non-negative
219:         assert (entropy >= 0).all()
220: 
221:     def test_action_decomposition_buzz(self, t5_small_model, sample_texts):
222:         """actions 1-4 decompose to wait=1, answer=0-3."""
223:         model = t5_small_model
224:         encoding = model.encode_input(sample_texts[:1])  # Single sample
225: 
226:         for action_val in [1, 2, 3, 4]:
227:             actions = torch.tensor([action_val], dtype=torch.long, device=model.device)
228:             log_probs, entropy, values = model.get_action_log_probs(
229:                 encoding["input_ids"], encoding["attention_mask"], actions
230:             )
231: 
232:             assert log_probs.shape == (1,)
233:             assert (log_probs <= 0).all()
234: 
235:     def test_joint_action_log_prob_wait_vs_buzz(self, t5_small_model):
236:         """WAIT uses only wait prob; BUZZ uses wait+buzzed-answer prob."""
237:         model = t5_small_model
238:         wait_logits = torch.tensor(
239:             [[2.0, 0.0], [0.0, 2.0]],
240:             dtype=torch.float32,
241:             device=model.device,
242:         )
243:         answer_logits = torch.tensor(
244:             [[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, -1.0, -2.0]],
245:             dtype=torch.float32,
246:             device=model.device,
247:         )
248:         actions = torch.tensor([0, 2], dtype=torch.long, device=model.device)
249: 
250:         log_probs = model._joint_action_log_prob(wait_logits, answer_logits, actions)
251: 
252:         wait_log_probs = torch.log_softmax(wait_logits, dim=-1)
253:         answer_log_probs = torch.log_softmax(answer_logits, dim=-1)
254:         expected = torch.stack(
255:             [
256:                 wait_log_probs[0, 0],
257:                 wait_log_probs[1, 1] + answer_log_probs[1, 1],
258:             ]
259:         )
260:         assert torch.allclose(log_probs, expected, atol=1e-6)
261: 
262:     def test_joint_entropy_matches_chain_rule(self, t5_small_model):
263:         """Entropy follows H(wait) + p_buzz * H(answer)."""
264:         model = t5_small_model
265:         wait_logits = torch.tensor(
266:             [[1.0, 0.0], [0.0, 1.0]],
267:             dtype=torch.float32,
268:             device=model.device,
269:         )
270:         answer_logits = torch.tensor(
271:             [[0.1, 0.2, 0.3, 0.4], [2.0, 1.0, 0.0, -1.0]],
272:             dtype=torch.float32,
273:             device=model.device,
274:         )
275: 
276:         entropy = model._joint_entropy(wait_logits, answer_logits)
277: 
278:         wait_probs = torch.softmax(wait_logits, dim=-1)
279:         wait_log_probs = torch.log_softmax(wait_logits, dim=-1)
280:         answer_probs = torch.softmax(answer_logits, dim=-1)
281:         answer_log_probs = torch.log_softmax(answer_logits, dim=-1)
282:         expected = (
283:             -(wait_probs * wait_log_probs).sum(dim=-1)
284:             + wait_probs[:, 1] * (-(answer_probs * answer_log_probs).sum(dim=-1))
285:         )
286:         assert torch.allclose(entropy, expected, atol=1e-6)
287: 
288:     def test_select_action_skips_answer_sampling_when_all_wait(
289:         self, t5_small_model, monkeypatch: pytest.MonkeyPatch
290:     ):
291:         """Answer sampling only runs for buzz examples, not all WAIT examples."""
292:         model = t5_small_model
293:         encoding = model.encode_input(["alpha", "beta"])
294: 
295:         hidden_size = model.encoder.config.d_model
296:         fake_pooled = torch.zeros((2, hidden_size), dtype=torch.float32, device=model.device)
297:         monkeypatch.setattr(model, "get_encoder_output", lambda *_args, **_kwargs: fake_pooled)
298: 
299:         def fake_head(_pooled):
300:             wait_logits = torch.tensor(
301:                 [[10.0, -10.0], [8.0, -8.0]],
302:                 dtype=torch.float32,
303:                 device=model.device,
304:             )
305:             answer_logits = torch.tensor(
306:                 [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
307:                 dtype=torch.float32,
308:                 device=model.device,
309:             )
310:             values = torch.zeros((2, 1), dtype=torch.float32, device=model.device)
311:             return wait_logits, answer_logits, values
312: 
313:         monkeypatch.setattr(model.policy_head, "forward", fake_head)
314: 
315:         sample_shapes = []
316:         original_sample = torch.distributions.Categorical.sample
317: 
318:         def fake_sample(self, sample_shape=torch.Size()):
319:             sample_shapes.append(tuple(self.probs.shape))
320:             return torch.zeros(self.probs.shape[:-1], dtype=torch.long, device=self.probs.device)
321: 
322:         monkeypatch.setattr(torch.distributions.Categorical, "sample", fake_sample)
323: 
324:         actions, _info = model.select_action(
325:             encoding["input_ids"],
326:             encoding["attention_mask"],
327:             deterministic=False,
328:         )
329: 
330:         assert torch.equal(actions.cpu(), torch.zeros(2, dtype=torch.long))
331:         assert sample_shapes == [(2, 2)]
332: 
333:     def test_select_action_deterministic(self, t5_small_model, sample_texts):
334:         """Deterministic mode produces consistent actions."""
335:         model = t5_small_model
336:         encoding = model.encode_input(sample_texts)
337: 
338:         actions1, info1 = model.select_action(
339:             encoding["input_ids"],
340:             encoding["attention_mask"],
341:             deterministic=True,
342:         )
343:         actions2, info2 = model.select_action(
344:             encoding["input_ids"],
345:             encoding["attention_mask"],
346:             deterministic=True,
347:         )
348: 
349:         assert torch.equal(actions1, actions2)
350: 
351:     def test_select_action_stochastic(self, t5_small_model, sample_texts):
352:         """Stochastic mode samples from distribution (info dict has correct keys)."""
353:         model = t5_small_model
354:         encoding = model.encode_input(sample_texts)
355: 
356:         actions, info = model.select_action(
357:             encoding["input_ids"],
358:             encoding["attention_mask"],
359:             deterministic=False,
360:         )
361: 
362:         assert actions.shape == (len(sample_texts),)
363:         assert "wait_logits" in info
364:         assert "answer_logits" in info
365:         assert "wait_probs" in info
366:         assert "answer_probs" in info
367:         assert "values" in info
368:         assert "log_probs" in info
369: 
370:         # All actions should be in valid range [0, K]
371:         assert (actions >= 0).all()
372:         assert (actions <= 4).all()
373: 
374:     def test_select_action_returns_valid_range(self, t5_small_model, sample_texts):
375:         """Combined actions are in range [0, num_choices]."""
376:         model = t5_small_model
377:         encoding = model.encode_input(sample_texts)
378: 
379:         # Run many times to cover both wait and buzz actions
380:         for _ in range(10):
381:             actions, info = model.select_action(
382:                 encoding["input_ids"],
383:                 encoding["attention_mask"],
384:                 deterministic=False,
385:                 temperature=2.0,  # Higher temp for more randomness
386:             )
387:             assert (actions >= 0).all()
388:             assert (actions <= 4).all()
389: 
390:     def test_get_action_log_probs_matches_select(self, t5_small_model, sample_texts):
391:         """Log probs from get_action_log_probs are consistent with select_action."""
392:         model = t5_small_model
393:         model.eval()
394:         encoding = model.encode_input(sample_texts[:1])
395: 
396:         # Get deterministic action
397:         actions, info = model.select_action(
398:             encoding["input_ids"],
399:             encoding["attention_mask"],
400:             deterministic=True,
401:         )
402: 
403:         # Compute log probs for the same action
404:         log_probs, entropy, values = model.get_action_log_probs(
405:             encoding["input_ids"],
406:             encoding["attention_mask"],
407:             actions,
408:         )
409: 
410:         # Log probs should be finite
411:         assert torch.isfinite(log_probs).all()
412:         assert torch.isfinite(entropy).all()
413:         assert torch.isfinite(values).all()
414: 
415: 
416: # ---------------------------------------------------------------------------
417: # Predict Answer Tests
418: # ---------------------------------------------------------------------------
419: 
420: 
421: class TestPredictAnswer:
422:     """Tests for supervised training interface."""
423: 
424:     def test_predict_answer(self, t5_small_model, sample_texts):
425:         """predict_answer returns logits and predictions with correct shapes."""
426:         model = t5_small_model
427:         encoding = model.encode_input(sample_texts)
428: 
429:         answer_logits, predictions = model.predict_answer(
430:             encoding["input_ids"],
431:             encoding["attention_mask"],
432:         )
433: 
434:         assert answer_logits.shape == (len(sample_texts), 4)
435:         assert predictions.shape == (len(sample_texts),)
436:         # Predictions should be in valid range
437:         assert (predictions >= 0).all()
438:         assert (predictions < 4).all()
439: 
440: 
441: # ---------------------------------------------------------------------------
442: # Checkpoint Tests
443: # ---------------------------------------------------------------------------
444: 
445: 
446: class TestCheckpoint:
447:     """Tests for save/load checkpoint functionality."""
448: 
449:     def test_save_load_checkpoint(self, t5_small_model, sample_texts):
450:         """Save then load produces identical model outputs."""
451:         model = t5_small_model
452:         model.eval()
453: 
454:         with tempfile.TemporaryDirectory() as tmpdir:
455:             save_path = os.path.join(tmpdir, "checkpoint")
456: 
457:             # Get output before save
458:             with torch.no_grad():
459:                 wait_before, answer_before, value_before = model(sample_texts)
460: 
461:             # Save
462:             model.save(save_path)
463: 
464:             # Verify files exist
465:             assert os.path.exists(os.path.join(save_path, "policy_head.pt"))
466:             assert os.path.exists(os.path.join(save_path, "config.json"))
467: 
468:             # Load into same model
469:             model.load(save_path)
470: 
471:             # Get output after load
472:             with torch.no_grad():
473:                 wait_after, answer_after, value_after = model(sample_texts)
474: 
475:             # Outputs should be identical
476:             assert torch.allclose(wait_before, wait_after, atol=1e-5)
477:             assert torch.allclose(answer_before, answer_after, atol=1e-5)
478:             assert torch.allclose(value_before, value_after, atol=1e-5)
````

## File: tests/test_text_wrapper.py
````python
  1: """Unit tests for TextObservationWrapper.
  2: 
  3: Tests verify that the wrapper correctly converts TossupMCEnv's numeric
  4: belief observations into text-formatted strings for T5PolicyModel input.
  5: 
  6: Uses TF-IDF likelihood for fast test execution (<1 second total).
  7: """
  8: 
  9: from __future__ import annotations
 10: 
 11: import pytest
 12: 
 13: from qb_data.mc_builder import MCQuestion
 14: from qb_env.text_wrapper import TextObservationWrapper
 15: from qb_env.tossup_env import TossupMCEnv
 16: from models.likelihoods import TfIdfLikelihood
 17: 
 18: 
 19: # ---------------------------------------------------------------------------
 20: # Fixtures
 21: # ---------------------------------------------------------------------------
 22: 
 23: 
 24: @pytest.fixture
 25: def sample_mc_question() -> MCQuestion:
 26:     """Return a minimal MCQuestion for wrapper testing."""
 27:     tokens = [
 28:         "Who", "was", "the", "first", "president",
 29:         "of", "the", "United", "States", "?",
 30:     ]
 31:     run_indices = [0, 2, 4, 6, 8, 9]
 32:     cumulative_prefixes = [
 33:         "Who",
 34:         "Who was the",
 35:         "Who was the first president",
 36:         "Who was the first president of the",
 37:         "Who was the first president of the United States",
 38:         "Who was the first president of the United States ?",
 39:     ]
 40:     return MCQuestion(
 41:         qid="test_q1",
 42:         question="Who was the first president of the United States?",
 43:         tokens=tokens,
 44:         answer_primary="George Washington",
 45:         clean_answers=["George Washington", "Washington"],
 46:         run_indices=run_indices,
 47:         human_buzz_positions=[],
 48:         category="History",
 49:         cumulative_prefixes=cumulative_prefixes,
 50:         options=[
 51:             "George Washington",
 52:             "Thomas Jefferson",
 53:             "John Adams",
 54:             "Benjamin Franklin",
 55:         ],
 56:         gold_index=0,
 57:         option_profiles=[
 58:             "George Washington first president commander revolutionary war",
 59:             "Thomas Jefferson third president declaration independence",
 60:             "John Adams second president Massachusetts diplomat",
 61:             "Benjamin Franklin inventor diplomat Philadelphia printing",
 62:         ],
 63:         option_answer_primary=[
 64:             "George Washington",
 65:             "Thomas Jefferson",
 66:             "John Adams",
 67:             "Benjamin Franklin",
 68:         ],
 69:         distractor_strategy="test",
 70:     )
 71: 
 72: 
 73: @pytest.fixture
 74: def wrapped_env(sample_mc_question: MCQuestion) -> TextObservationWrapper:
 75:     """Return a TextObservationWrapper around a TossupMCEnv."""
 76:     corpus = sample_mc_question.option_profiles[:]
 77:     model = TfIdfLikelihood(corpus_texts=corpus)
 78:     questions = [sample_mc_question] * 3
 79:     env = TossupMCEnv(
 80:         questions=questions,
 81:         likelihood_model=model,
 82:         K=4,
 83:         reward_mode="simple",
 84:         wait_penalty=0.0,
 85:         buzz_correct=1.0,
 86:         buzz_incorrect=-1.0,
 87:         belief_mode="from_scratch",
 88:         beta=5.0,
 89:     )
 90:     return TextObservationWrapper(env)
 91: 
 92: 
 93: # ---------------------------------------------------------------------------
 94: # Tests
 95: # ---------------------------------------------------------------------------
 96: 
 97: 
 98: class TestTextObservationWrapper:
 99:     """Tests for TextObservationWrapper class."""
100: 
101:     def test_wrapper_observation_format(self, wrapped_env: TextObservationWrapper):
102:         """Observation returns 'CLUES: ... | CHOICES: ...' format."""
103:         obs, info = wrapped_env.reset()
104: 
105:         assert isinstance(obs, str), f"Expected str, got {type(obs)}"
106:         assert "CLUES:" in obs, "Observation must contain 'CLUES:'"
107:         assert "CHOICES:" in obs, "Observation must contain 'CHOICES:'"
108:         assert "(1)" in obs, "Choices must be numbered starting at (1)"
109:         assert "(4)" in obs, "All 4 choices must be present"
110: 
111:     def test_wrapper_incremental_clues(self, wrapped_env: TextObservationWrapper):
112:         """Wrapper shows correct clues based on step_idx progression."""
113:         obs0, _ = wrapped_env.reset()
114: 
115:         # Initial: first token only
116:         clues_part = obs0.split(" | CHOICES:")[0].replace("CLUES: ", "")
117:         assert clues_part == "Who", f"Initial clues should be 'Who', got '{clues_part}'"
118: 
119:         # After first WAIT: cumulative_prefixes[0] = "Who"
120:         obs1, _, _, _, _ = wrapped_env.step(0)
121:         clues1 = obs1.split(" | CHOICES:")[0].replace("CLUES: ", "")
122:         assert clues1 == "Who", f"After 1st WAIT should be 'Who', got '{clues1}'"
123: 
124:         # After second WAIT: cumulative_prefixes[1] = "Who was the"
125:         obs2, _, _, _, _ = wrapped_env.step(0)
126:         clues2 = obs2.split(" | CHOICES:")[0].replace("CLUES: ", "")
127:         assert clues2 == "Who was the", f"After 2nd WAIT should be 'Who was the', got '{clues2}'"
128: 
129:     def test_wrapper_gymnasium_api(self, wrapped_env: TextObservationWrapper):
130:         """reset() and step() still work after wrapping."""
131:         # reset returns (obs, info) tuple
132:         result = wrapped_env.reset()
133:         assert isinstance(result, tuple)
134:         assert len(result) == 2
135:         obs, info = result
136:         assert isinstance(obs, str)
137:         assert isinstance(info, dict)
138:         assert "qid" in info
139: 
140:         # step returns (obs, reward, terminated, truncated, info)
141:         result = wrapped_env.step(0)  # WAIT
142:         assert isinstance(result, tuple)
143:         assert len(result) == 5
144:         obs, reward, terminated, truncated, info = result
145:         assert isinstance(obs, str)
146:         assert isinstance(reward, float)
147:         assert isinstance(terminated, bool)
148:         assert isinstance(truncated, bool)
149:         assert isinstance(info, dict)
150: 
151:     def test_wrapper_preserves_reward(self, sample_mc_question: MCQuestion):
152:         """Reward from wrapped env matches underlying env behavior."""
153:         corpus = sample_mc_question.option_profiles[:]
154:         model = TfIdfLikelihood(corpus_texts=corpus)
155: 
156:         # Create unwrapped env
157:         env = TossupMCEnv(
158:             questions=[sample_mc_question] * 3,
159:             likelihood_model=model,
160:             K=4,
161:             reward_mode="simple",
162:             buzz_correct=1.0,
163:             buzz_incorrect=-1.0,
164:             seed=42,
165:         )
166: 
167:         # Create wrapped env with same seed
168:         env2 = TossupMCEnv(
169:             questions=[sample_mc_question] * 3,
170:             likelihood_model=model,
171:             K=4,
172:             reward_mode="simple",
173:             buzz_correct=1.0,
174:             buzz_incorrect=-1.0,
175:             seed=42,
176:         )
177:         wrapped = TextObservationWrapper(env2)
178: 
179:         # Reset both
180:         _, info1 = env.reset(seed=42)
181:         _, info2 = wrapped.reset(seed=42)
182: 
183:         # Take same actions
184:         _, r1, d1, t1, _ = env.step(0)
185:         _, r2, d2, t2, _ = wrapped.step(0)
186:         assert r1 == r2, f"Rewards differ: {r1} vs {r2}"
187:         assert d1 == d2, f"Terminated differs"
188:         assert t1 == t2, f"Truncated differs"
189: 
190:         # BUZZ with answer 1 (correct for gold_index=0)
191:         _, r1, d1, t1, _ = env.step(1)
192:         _, r2, d2, t2, _ = wrapped.step(1)
193:         assert r1 == r2, f"Buzz rewards differ: {r1} vs {r2}"
194:         assert d1 == d2
195: 
196:     def test_wrapper_multiple_steps(self, wrapped_env: TextObservationWrapper):
197:         """Multi-step episode produces increasing clue text."""
198:         obs, _ = wrapped_env.reset()
199:         prev_clues = obs.split(" | CHOICES:")[0]
200: 
201:         # Take multiple WAIT steps and verify clues grow
202:         grew_at_least_once = False
203:         for step in range(4):
204:             obs, _, terminated, truncated, _ = wrapped_env.step(0)
205:             if terminated or truncated:
206:                 break
207:             current_clues = obs.split(" | CHOICES:")[0]
208:             if len(current_clues) > len(prev_clues):
209:                 grew_at_least_once = True
210:             # Clues should never shrink
211:             assert len(current_clues) >= len(prev_clues), (
212:                 f"Clues shrank at step {step}: '{prev_clues}' -> '{current_clues}'"
213:             )
214:             prev_clues = current_clues
215: 
216:         assert grew_at_least_once, "Clue text should grow with more WAITs"
217: 
218:     def test_wrapper_choices_include_all_options(
219:         self, wrapped_env: TextObservationWrapper
220:     ):
221:         """All 4 answer options appear in the choices section."""
222:         obs, _ = wrapped_env.reset()
223:         choices_part = obs.split("CHOICES: ")[1]
224: 
225:         assert "George Washington" in choices_part
226:         assert "Thomas Jefferson" in choices_part
227:         assert "John Adams" in choices_part
228:         assert "Benjamin Franklin" in choices_part
229: 
230:     def test_wrapper_buzz_ends_episode(self, wrapped_env: TextObservationWrapper):
231:         """Buzzing with an answer ends the episode."""
232:         wrapped_env.reset()
233:         _, _, terminated, truncated, info = wrapped_env.step(1)  # BUZZ answer 0
234:         assert terminated or truncated, "Episode should end after BUZZ"
235: 
236:     def test_wrapper_complete_episode(self, wrapped_env: TextObservationWrapper):
237:         """Full episode: WAIT until truncated or BUZZ."""
238:         wrapped_env.reset()
239: 
240:         for step in range(20):
241:             obs, reward, terminated, truncated, info = wrapped_env.step(0)
242:             if terminated or truncated:
243:                 break
244:             assert isinstance(obs, str)
245: 
246:         # Episode must have ended (6 clue steps)
247:         assert terminated or truncated, "Episode should end within 20 steps"
248: 
249:     def test_wrapper_k3_formats_three_choices(
250:         self, sample_mc_question: MCQuestion
251:     ) -> None:
252:         """Text wrapper dynamically formats K=3 questions correctly."""
253:         from dataclasses import replace
254: 
255:         q3 = replace(
256:             sample_mc_question,
257:             qid="q_k3",
258:             options=sample_mc_question.options[:3],
259:             option_profiles=sample_mc_question.option_profiles[:3],
260:             option_answer_primary=sample_mc_question.option_answer_primary[:3],
261:             gold_index=0,
262:         )
263:         corpus = sample_mc_question.option_profiles[:]
264:         model = TfIdfLikelihood(corpus_texts=corpus)
265:         env = TossupMCEnv(
266:             questions=[q3],
267:             likelihood_model=model,
268:             K=3,
269:             reward_mode="simple",
270:             belief_mode="from_scratch",
271:         )
272:         wrapped = TextObservationWrapper(env)
273:         obs, _ = wrapped.reset(seed=42)
274:         assert "(3)" in obs
275:         assert "(4)" not in obs
````

## File: tests/test_variable_k_integration.py
````python
  1: """Integration test exercising a mixed-K pipeline path."""
  2: 
  3: from __future__ import annotations
  4: 
  5: import numpy as np
  6: import pytest
  7: 
  8: from qb_data.data_loader import TossupQuestion
  9: from qb_data.answer_profiles import AnswerProfileBuilder
 10: from qb_data.mc_builder import MCBuilder
 11: from models.likelihoods import TfIdfLikelihood
 12: from qb_env.tossup_env import TossupMCEnv
 13: from qb_env.text_wrapper import TextObservationWrapper
 14: from agents.threshold_buzzer import ThresholdBuzzer
 15: 
 16: 
 17: def _make_questions(n: int = 30) -> list[TossupQuestion]:
 18:     questions = []
 19:     for i in range(n):
 20:         tokens = [f"word{i}_{j}" for j in range(8)]
 21:         questions.append(
 22:             TossupQuestion(
 23:                 qid=f"q{i:03d}",
 24:                 question=" ".join(tokens),
 25:                 tokens=tokens,
 26:                 answer_primary=f"Answer_{i}",
 27:                 clean_answers=[f"Answer_{i}"],
 28:                 run_indices=[1, 3, 7],
 29:                 human_buzz_positions=[],
 30:                 category=["History", "Science"][i % 2],
 31:                 cumulative_prefixes=[
 32:                     " ".join(tokens[:2]),
 33:                     " ".join(tokens[:4]),
 34:                     " ".join(tokens),
 35:                 ],
 36:             )
 37:         )
 38:     return questions
 39: 
 40: 
 41: def test_mixed_k_build_env_baseline() -> None:
 42:     """Build mixed-K dataset, construct env, run a baseline agent."""
 43:     questions = _make_questions(30)
 44:     builder = MCBuilder(
 45:         K=5, strategy="category_random", random_seed=42,
 46:         variable_K=True, min_K=2, max_K=5,
 47:     )
 48:     profile = AnswerProfileBuilder()
 49:     mc = builder.build(questions, profile)
 50:     assert len(mc) > 0
 51: 
 52:     option_counts = {len(q.options) for q in mc}
 53:     assert len(option_counts) > 1, f"Expected mixed K, got {option_counts}"
 54: 
 55:     corpus = [q.question for q in mc] + [p for q in mc for p in q.option_profiles]
 56:     lm = TfIdfLikelihood(corpus_texts=corpus)
 57: 
 58:     max_k = max(len(q.options) for q in mc)
 59:     env = TossupMCEnv(
 60:         questions=mc, likelihood_model=lm,
 61:         K=max_k, variable_K=True, max_K=max_k,
 62:         reward_mode="simple", belief_mode="from_scratch",
 63:     )
 64: 
 65:     obs, info = env.reset(seed=42, options={"question_idx": 0})
 66:     assert obs.shape == (max_k + 6,)
 67: 
 68:     mask = env.action_masks()
 69:     k_actual = len(mc[0].options)
 70:     assert mask[0]
 71:     assert all(mask[1: k_actual + 1])
 72: 
 73:     buzzer = ThresholdBuzzer(
 74:         likelihood_model=lm, threshold=0.5, beta=5.0, alpha=10.0,
 75:     )
 76:     result = buzzer.run_episode(mc[0])
 77:     assert 0 <= result.buzz_index < len(mc[0].options)
 78: 
 79: 
 80: def test_mixed_k_text_wrapper_formats_correctly() -> None:
 81:     """TextObservationWrapper formats per-question K dynamically."""
 82:     questions = _make_questions(30)
 83:     builder = MCBuilder(
 84:         K=4, strategy="category_random", random_seed=42,
 85:         variable_K=True, min_K=2, max_K=4,
 86:     )
 87:     profile = AnswerProfileBuilder()
 88:     mc = builder.build(questions, profile)
 89:     assert len(mc) > 0
 90: 
 91:     corpus = [q.question for q in mc] + [p for q in mc for p in q.option_profiles]
 92:     lm = TfIdfLikelihood(corpus_texts=corpus)
 93: 
 94:     max_k = max(len(q.options) for q in mc)
 95:     env = TossupMCEnv(
 96:         questions=mc, likelihood_model=lm,
 97:         K=max_k, variable_K=True, max_K=max_k,
 98:         reward_mode="simple", belief_mode="from_scratch",
 99:     )
100:     wrapped = TextObservationWrapper(env)
101: 
102:     for idx in range(min(5, len(mc))):
103:         obs, _ = wrapped.reset(seed=42, options={"question_idx": idx})
104:         n_opts = len(mc[idx].options)
105:         assert f"({n_opts})" in obs
106:         if n_opts < max_k:
107:             assert f"({n_opts + 1})" not in obs
````

## File: training/hazard_pretrain.py
````python
 1: """Hazard pretraining bridge utilities for stopping-aware warm starts."""
 2: 
 3: from __future__ import annotations
 4: 
 5: from dataclasses import dataclass
 6: 
 7: import torch
 8: 
 9: 
10: @dataclass
11: class HazardBatchOutput:
12:     """Container for hazard-bridge intermediate tensors."""
13: 
14:     stop_probs: torch.Tensor
15:     survival: torch.Tensor
16:     stop_mass: torch.Tensor
17:     nll_per_prefix: torch.Tensor
18:     loss: torch.Tensor
19: 
20: 
21: def compute_survival_terms(stop_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
22:     """Compute survival and stop-mass terms from per-prefix stop probabilities."""
23:     stay_probs = 1.0 - stop_probs
24:     batch_size, steps = stop_probs.shape
25:     survival = torch.ones(
26:         (batch_size, steps + 1), dtype=stop_probs.dtype, device=stop_probs.device
27:     )
28:     if steps > 0:
29:         survival[:, 1:] = torch.cumprod(stay_probs, dim=1)
30:     stop_mass = survival[:, :-1] * stop_probs
31:     return survival, stop_mass
32: 
33: 
34: def hazard_expected_nll_loss(
35:     stop_probs: torch.Tensor,
36:     nll_per_prefix: torch.Tensor,
37:     beta_terminal: float = 1.0,
38:     mask: torch.Tensor | None = None,
39: ) -> torch.Tensor:
40:     """Compute the hazard-bridge expected NLL loss prior to PPO."""
41:     survival, stop_mass = compute_survival_terms(stop_probs)
42:     weighted_nll = stop_mass * nll_per_prefix
43:     if mask is not None:
44:         weighted_nll = weighted_nll * mask
45:     seq_loss = weighted_nll.sum(dim=1) + beta_terminal * survival[:, -1]
46:     return seq_loss.mean()
````

## File: training/train_ppo_t5.py
````python
  1: """
  2: Custom PPO Training for T5 Policy Model
  3: 
  4: Implements PPOTrainer with RolloutBuffer for end-to-end PPO fine-tuning of
  5: T5PolicyModel on incremental quiz bowl episodes. Uses Generalized Advantage
  6: Estimation (GAE) for variance reduction and dynamic batch padding to minimize
  7: memory footprint.
  8: 
  9: Key design decisions:
 10:     - Rollout tensors (input_ids, attention_mask) are immediately detached and
 11:       moved to CPU after collection to prevent GPU memory accumulation.
 12:     - Dynamic padding: each mini-batch is padded to the max length within that
 13:       batch, not a global 512-token maximum, saving ~50%+ memory.
 14:     - Config-dict interface for compatibility with the unified codebase YAML
 15:       config pattern (see configs/t5_policy.yaml).
 16: 
 17: Ported from qanta-buzzer reference implementation (train_ppo.py) with:
 18:     - TextObservationWrapper for text-based rollout collection
 19:     - Memory-safe tensor management (detach + CPU storage)
 20:     - Dynamic padding per mini-batch
 21:     - Config dict interface replacing Config class
 22:     - NumPy-style docstrings
 23: 
 24: Usage
 25: -----
 26: From Python::
 27: 
 28:     from training.train_ppo_t5 import PPOTrainer, run_ppo_training
 29:     from models.t5_policy import T5PolicyModel
 30:     from qb_data.mc_builder import MCQuestion
 31: 
 32:     model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
 33:     trainer = PPOTrainer(model, train_qs, val_qs, config)
 34:     trainer.train()
 35: 
 36: From command line::
 37: 
 38:     python scripts/train_t5_policy.py --config configs/t5_policy.yaml
 39: """
 40: 
 41: from __future__ import annotations
 42: 
 43: import json
 44: import random
 45: from dataclasses import dataclass
 46: from pathlib import Path
 47: from typing import Any, Dict, List, Optional, Tuple
 48: 
 49: import numpy as np
 50: import torch
 51: import torch.nn as nn
 52: import torch.optim as optim
 53: 
 54: from models.t5_policy import T5PolicyModel
 55: from qb_data.mc_builder import MCQuestion
 56: 
 57: 
 58: @dataclass
 59: class RolloutStep:
 60:     """Single step in an episode rollout.
 61: 
 62:     Stores observation text, action, reward, value estimate, and log probability
 63:     for a single environment step. Tokenized tensors (input_ids, attention_mask)
 64:     are stored on CPU to prevent GPU memory accumulation during rollout collection.
 65: 
 66:     Attributes
 67:     ----------
 68:     observation_text : str
 69:         Text observation at this step (CLUES: ... | CHOICES: ...).
 70:     action : int
 71:         Combined action taken (0=WAIT, 1..K=SELECT).
 72:     reward : float
 73:         Scalar reward received.
 74:     done : bool
 75:         Whether this step ended the episode.
 76:     value : float
 77:         Value estimate from the critic at this step.
 78:     log_prob : float
 79:         Log probability of the action under the policy at collection time.
 80:     input_ids : torch.Tensor or None
 81:         Tokenized input IDs stored on CPU. Shape ``[1, seq_len]``.
 82:     attention_mask : torch.Tensor or None
 83:         Attention mask stored on CPU. Shape ``[1, seq_len]``.
 84:     return_ : float
 85:         Discounted return (filled by ``compute_returns_and_advantages``).
 86:     advantage : float
 87:         GAE advantage (filled by ``compute_returns_and_advantages``).
 88:     """
 89: 
 90:     observation_text: str
 91:     action: int
 92:     reward: float
 93:     done: bool
 94:     value: float
 95:     log_prob: float
 96:     input_ids: Optional[torch.Tensor] = None
 97:     attention_mask: Optional[torch.Tensor] = None
 98:     return_: float = 0.0
 99:     advantage: float = 0.0
100: 
101: 
102: class RolloutBuffer:
103:     """Buffer to store and process episode rollouts for PPO updates.
104: 
105:     Accumulates complete episode rollouts (lists of RolloutStep), then computes
106:     discounted returns and GAE advantages across all episodes. Provides a flat
107:     view of all steps for mini-batch iteration during PPO updates.
108: 
109:     Attributes
110:     ----------
111:     rollouts : list[list[RolloutStep]]
112:         List of episode rollouts, each a list of steps.
113:     """
114: 
115:     def __init__(self) -> None:
116:         self.rollouts: List[List[RolloutStep]] = []
117: 
118:     def reset(self) -> None:
119:         """Clear all stored rollouts."""
120:         self.rollouts = []
121: 
122:     def add_rollout(self, steps: List[RolloutStep]) -> None:
123:         """Add a complete episode rollout to the buffer.
124: 
125:         Parameters
126:         ----------
127:         steps : list[RolloutStep]
128:             Complete episode rollout (ordered list of steps from reset to done).
129:         """
130:         self.rollouts.append(steps)
131: 
132:     def get_all_steps(self) -> List[RolloutStep]:
133:         """Get a flat list of all steps from all rollouts.
134: 
135:         Returns
136:         -------
137:         list[RolloutStep]
138:             All steps concatenated in order (rollout 0 steps, then rollout 1, ...).
139:         """
140:         all_steps: List[RolloutStep] = []
141:         for rollout in self.rollouts:
142:             all_steps.extend(rollout)
143:         return all_steps
144: 
145:     def compute_returns_and_advantages(
146:         self, gamma: float, gae_lambda: float
147:     ) -> None:
148:         """Compute discounted returns and GAE advantages for all rollouts.
149: 
150:         Uses Generalized Advantage Estimation (GAE) to compute per-step
151:         advantages. For each rollout, iterates backward from the terminal
152:         step computing:
153: 
154:             delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
155:             A_t = delta_t + gamma * lambda * A_{t+1}
156:             G_t = A_t + V(s_t)
157: 
158:         Terminal states reset next_value and gae to 0.
159: 
160:         Parameters
161:         ----------
162:         gamma : float
163:             Discount factor in [0, 1]. Higher values weight future rewards more.
164:         gae_lambda : float
165:             GAE lambda in [0, 1]. Trades off bias (low) vs variance (high).
166:         """
167:         for rollout in self.rollouts:
168:             rewards = [step.reward for step in rollout]
169:             values = [step.value for step in rollout]
170:             dones = [step.done for step in rollout]
171: 
172:             # GAE computation (backward pass)
173:             gae = 0.0
174:             next_value = 0.0  # Terminal state value
175: 
176:             for t in reversed(range(len(rollout))):
177:                 if dones[t]:
178:                     next_value = 0.0
179:                     gae = 0.0
180: 
181:                 # TD error
182:                 delta = rewards[t] + gamma * next_value - values[t]
183: 
184:                 # GAE accumulation
185:                 gae = delta + gamma * gae_lambda * gae
186: 
187:                 # Store return and advantage
188:                 rollout[t].return_ = gae + values[t]
189:                 rollout[t].advantage = gae
190: 
191:                 next_value = values[t]
192: 
193:     def __len__(self) -> int:
194:         return len(self.rollouts)
195: 
196: 
197: class PPOTrainer:
198:     """Custom PPO trainer for T5PolicyModel on quiz bowl episodes.
199: 
200:     Collects rollouts by running T5PolicyModel in text-observation episodes
201:     (via TextObservationWrapper), then updates the policy using clipped
202:     surrogate PPO loss with value function and entropy regularization.
203: 
204:     The trainer handles the complete training loop:
205:     1. Collect rollouts (episodes) using the current policy
206:     2. Compute GAE advantages
207:     3. Update policy with mini-batch PPO for multiple epochs
208:     4. Periodically validate and save checkpoints
209: 
210:     Parameters
211:     ----------
212:     model : T5PolicyModel
213:         T5 policy model to train. Should be pre-trained via supervised
214:         warm-start for faster convergence.
215:     train_questions : list[MCQuestion]
216:         Training set questions for rollout collection.
217:     val_questions : list[MCQuestion]
218:         Validation set questions for periodic evaluation.
219:     config : dict[str, Any]
220:         Configuration dictionary with PPO hyperparameters:
221: 
222:         - ``ppo_lr`` (float): Learning rate. Default 1e-5.
223:         - ``ppo_iterations`` (int): Number of collect-update cycles. Default 100.
224:         - ``ppo_batch_size`` (int): Mini-batch size for PPO updates. Default 8.
225:         - ``ppo_epochs_per_iter`` (int): PPO epochs per iteration. Default 4.
226:         - ``ppo_gamma`` (float): Discount factor. Default 0.99.
227:         - ``ppo_gae_lambda`` (float): GAE lambda. Default 0.95.
228:         - ``ppo_clip_ratio`` (float): PPO clip ratio. Default 0.2.
229:         - ``ppo_value_coef`` (float): Value loss coefficient. Default 0.5.
230:         - ``ppo_entropy_coef`` (float): Entropy bonus coefficient. Default 0.01.
231:         - ``ppo_max_grad_norm`` (float): Gradient clip norm. Default 0.5.
232:         - ``ppo_episodes_per_iter`` (int): Episodes per rollout. Default 16.
233:         - ``eval_interval`` (int): Validate every N iterations. Default 10.
234:         - ``save_interval`` (int): Save checkpoint every N iterations. Default 20.
235:         - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
236:         - ``reward_time_penalty`` (float): Time penalty for env. Default 0.1.
237: 
238:     Attributes
239:     ----------
240:     model : T5PolicyModel
241:         The model being trained.
242:     optimizer : torch.optim.AdamW
243:         Optimizer with weight decay.
244:     best_val_reward : float
245:         Best validation reward seen so far.
246:     history : list[dict]
247:         Per-iteration training metrics.
248:     checkpoint_dir : Path
249:         Directory for saving PPO checkpoints.
250:     """
251: 
252:     def __init__(
253:         self,
254:         model: T5PolicyModel,
255:         train_questions: List[MCQuestion],
256:         val_questions: List[MCQuestion],
257:         config: Dict[str, Any],
258:     ) -> None:
259:         self.model = model
260:         self.train_questions = list(train_questions)
261:         self.val_questions = list(val_questions)
262:         self.config = config
263: 
264:         self.device = model.device
265: 
266:         # PPO hyperparameters
267:         self.lr = float(config.get("ppo_lr", 1e-5))
268:         self.iterations = int(config.get("ppo_iterations", 100))
269:         self.batch_size = int(config.get("ppo_batch_size", 8))
270:         self.epochs_per_iter = int(config.get("ppo_epochs_per_iter", 4))
271:         self.gamma = float(config.get("ppo_gamma", 0.99))
272:         self.gae_lambda = float(config.get("ppo_gae_lambda", 0.95))
273:         self.clip_ratio = float(config.get("ppo_clip_ratio", 0.2))
274:         self.value_coef = float(config.get("ppo_value_coef", 0.5))
275:         self.entropy_coef = float(config.get("ppo_entropy_coef", 0.01))
276:         self.max_grad_norm = float(config.get("ppo_max_grad_norm", 0.5))
277:         self.episodes_per_iter = int(config.get("ppo_episodes_per_iter", 16))
278:         self.eval_interval = int(config.get("eval_interval", 10))
279:         self.save_interval = int(config.get("save_interval", 20))
280:         self.reward_time_penalty = float(config.get("reward_time_penalty", 0.1))
281:         self.max_input_length = int(config.get("max_input_length", 512))
282: 
283:         # Optimizer
284:         self.optimizer = optim.AdamW(
285:             model.parameters(), lr=self.lr, weight_decay=0.01
286:         )
287: 
288:         # Training state
289:         self.current_iteration = 0
290:         self.best_val_reward = -float("inf")
291:         self.history: List[Dict[str, Any]] = []
292: 
293:         # Checkpoint directory
294:         self.checkpoint_dir = (
295:             Path(config.get("checkpoint_dir", "checkpoints")) / "ppo_t5"
296:         )
297:         self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
298: 
299:     def collect_rollouts(self, num_episodes: int) -> RolloutBuffer:
300:         """Collect rollouts by running episodes with the current policy.
301: 
302:         Creates a TossupMCEnv + TextObservationWrapper for each sampled
303:         question, runs the policy until episode termination, and stores
304:         all steps in a RolloutBuffer. Tokenized tensors are detached and
305:         moved to CPU immediately to prevent GPU memory accumulation.
306: 
307:         Parameters
308:         ----------
309:         num_episodes : int
310:             Number of episodes to collect.
311: 
312:         Returns
313:         -------
314:         RolloutBuffer
315:             Buffer containing all collected episode rollouts.
316:         """
317:         from qb_env.text_wrapper import TextObservationWrapper
318:         from qb_env.tossup_env import TossupMCEnv
319:         from models.likelihoods import TfIdfLikelihood
320: 
321:         self.model.eval()
322:         buffer = RolloutBuffer()
323: 
324:         # Sample questions for this iteration
325:         questions = random.choices(self.train_questions, k=num_episodes)
326: 
327:         # Build a simple TF-IDF likelihood for environment scoring
328:         # (The T5 policy reads text directly; likelihood is only used for
329:         # environment reward computation via belief updates)
330:         corpus = []
331:         for q in self.train_questions[:100]:  # Use subset for speed
332:             corpus.extend(q.option_profiles)
333:         likelihood_model = TfIdfLikelihood(corpus_texts=corpus)
334: 
335:         with torch.no_grad():
336:             for question in questions:
337:                 env = TossupMCEnv(
338:                     questions=[question],
339:                     likelihood_model=likelihood_model,
340:                     K=len(question.options),
341:                     reward_mode="time_penalty",
342:                     wait_penalty=self.reward_time_penalty,
343:                     belief_mode="from_scratch",
344:                 )
345:                 wrapped_env = TextObservationWrapper(env)
346: 
347:                 obs, info = wrapped_env.reset()
348:                 done = False
349:                 rollout: List[RolloutStep] = []
350: 
351:                 while not done:
352:                     # Tokenize text observation
353:                     inputs = self.model.tokenizer(
354:                         obs,
355:                         return_tensors="pt",
356:                         padding=True,
357:                         truncation=True,
358:                         max_length=self.max_input_length,
359:                     ).to(self.device)
360: 
361:                     # Get action from policy
362:                     actions, act_info = self.model.select_action(
363:                         inputs["input_ids"],
364:                         inputs["attention_mask"],
365:                         deterministic=False,
366:                     )
367: 
368:                     action = actions.item()
369:                     value = act_info["values"].squeeze().item()
370:                     log_prob = act_info["log_probs"].item()
371: 
372:                     # Take environment step
373:                     next_obs, reward, terminated, truncated, step_info = (
374:                         wrapped_env.step(action)
375:                     )
376:                     done = terminated or truncated
377: 
378:                     # CRITICAL: Detach and move tensors to CPU immediately
379:                     # to prevent GPU memory accumulation during rollout collection
380:                     step = RolloutStep(
381:                         observation_text=obs,
382:                         action=action,
383:                         reward=reward,
384:                         done=done,
385:                         value=value,
386:                         log_prob=log_prob,
387:                         input_ids=inputs["input_ids"].detach().cpu(),
388:                         attention_mask=inputs["attention_mask"].detach().cpu(),
389:                     )
390:                     rollout.append(step)
391: 
392:                     obs = next_obs
393: 
394:                 buffer.add_rollout(rollout)
395: 
396:         return buffer
397: 
398:     def _pad_batch(
399:         self, batch_steps: List[RolloutStep]
400:     ) -> Tuple[torch.Tensor, torch.Tensor]:
401:         """Dynamically pad a mini-batch of steps to the max length in the batch.
402: 
403:         Instead of padding all sequences to the global max (512 tokens), pads
404:         only to the longest sequence in the current mini-batch. This typically
405:         saves 50%+ memory since most quiz bowl observations are 100-200 tokens.
406: 
407:         Parameters
408:         ----------
409:         batch_steps : list[RolloutStep]
410:             Mini-batch of rollout steps with stored input_ids and attention_mask.
411: 
412:         Returns
413:         -------
414:         input_ids : torch.Tensor
415:             Padded input IDs of shape ``[batch_size, max_len]``, on device.
416:         attention_mask : torch.Tensor
417:             Padded attention mask of shape ``[batch_size, max_len]``, on device.
418:         """
419:         max_len = max(step.input_ids.shape[1] for step in batch_steps)
420:         pad_token_id = self.model.tokenizer.pad_token_id
421: 
422:         padded_input_ids = []
423:         padded_attention_mask = []
424: 
425:         for step in batch_steps:
426:             seq_len = step.input_ids.shape[1]
427:             if seq_len < max_len:
428:                 pad_len = max_len - seq_len
429:                 input_ids_padded = torch.cat(
430:                     [
431:                         step.input_ids,
432:                         torch.full(
433:                             (1, pad_len),
434:                             pad_token_id,
435:                             dtype=step.input_ids.dtype,
436:                         ),
437:                     ],
438:                     dim=1,
439:                 )
440:                 attention_mask_padded = torch.cat(
441:                     [
442:                         step.attention_mask,
443:                         torch.zeros(
444:                             (1, pad_len), dtype=step.attention_mask.dtype
445:                         ),
446:                     ],
447:                     dim=1,
448:                 )
449:             else:
450:                 input_ids_padded = step.input_ids
451:                 attention_mask_padded = step.attention_mask
452: 
453:             padded_input_ids.append(input_ids_padded)
454:             padded_attention_mask.append(attention_mask_padded)
455: 
456:         input_ids = torch.cat(padded_input_ids).to(self.device)
457:         attention_mask = torch.cat(padded_attention_mask).to(self.device)
458: 
459:         return input_ids, attention_mask
460: 
461:     def update_policy(self, buffer: RolloutBuffer) -> Dict[str, float]:
462:         """Update the policy using PPO with clipped surrogate loss.
463: 
464:         Computes GAE advantages, normalizes them, then runs multiple epochs
465:         of mini-batch PPO updates. Each update computes the clipped surrogate
466:         policy loss, value function MSE loss, and entropy bonus.
467: 
468:         Parameters
469:         ----------
470:         buffer : RolloutBuffer
471:             Buffer with collected rollouts (compute_returns_and_advantages
472:             will be called internally).
473: 
474:         Returns
475:         -------
476:         dict[str, float]
477:             Training metrics: policy_loss, value_loss, entropy, num_updates.
478:         """
479:         self.model.train()
480: 
481:         # Compute returns and advantages
482:         buffer.compute_returns_and_advantages(
483:             gamma=self.gamma, gae_lambda=self.gae_lambda
484:         )
485: 
486:         # Get all steps
487:         all_steps = buffer.get_all_steps()
488:         if not all_steps:
489:             return {
490:                 "policy_loss": 0.0,
491:                 "value_loss": 0.0,
492:                 "entropy": 0.0,
493:                 "num_updates": 0,
494:             }
495: 
496:         # Normalize advantages
497:         advantages = torch.tensor(
498:             [step.advantage for step in all_steps], dtype=torch.float32
499:         )
500:         advantages = (advantages - advantages.mean()) / (
501:             advantages.std() + 1e-8
502:         )
503: 
504:         # Training metrics
505:         total_policy_loss = 0.0
506:         total_value_loss = 0.0
507:         total_entropy = 0.0
508:         num_updates = 0
509: 
510:         # PPO epochs
511:         for epoch in range(self.epochs_per_iter):
512:             # Shuffle step indices
513:             indices = np.random.permutation(len(all_steps))
514: 
515:             # Mini-batch updates
516:             for start_idx in range(0, len(all_steps), self.batch_size):
517:                 end_idx = min(start_idx + self.batch_size, len(all_steps))
518:                 batch_indices = indices[start_idx:end_idx]
519: 
520:                 # Get batch steps
521:                 batch_steps = [all_steps[i] for i in batch_indices]
522: 
523:                 # Dynamic padding to max length in THIS batch
524:                 input_ids, attention_mask = self._pad_batch(batch_steps)
525: 
526:                 # Prepare batch tensors
527:                 actions = torch.tensor(
528:                     [step.action for step in batch_steps],
529:                     dtype=torch.long,
530:                 ).to(self.device)
531:                 old_log_probs = torch.tensor(
532:                     [step.log_prob for step in batch_steps],
533:                     dtype=torch.float32,
534:                 ).to(self.device)
535:                 returns = torch.tensor(
536:                     [step.return_ for step in batch_steps],
537:                     dtype=torch.float32,
538:                 ).to(self.device)
539:                 batch_advantages = advantages[batch_indices].to(self.device)
540: 
541:                 # Get new log probs, entropy, and values from current policy
542:                 new_log_probs, entropy, values = (
543:                     self.model.get_action_log_probs(
544:                         input_ids, attention_mask, actions
545:                     )
546:                 )
547: 
548:                 # PPO clipped surrogate policy loss
549:                 ratio = torch.exp(new_log_probs - old_log_probs)
550:                 surr1 = ratio * batch_advantages
551:                 surr2 = (
552:                     torch.clamp(
553:                         ratio,
554:                         1.0 - self.clip_ratio,
555:                         1.0 + self.clip_ratio,
556:                     )
557:                     * batch_advantages
558:                 )
559:                 policy_loss = -torch.min(surr1, surr2).mean()
560: 
561:                 # Value function loss (MSE)
562:                 value_loss = nn.MSELoss()(values, returns)
563: 
564:                 # Entropy bonus (negative because we maximize entropy)
565:                 entropy_loss = -entropy.mean()
566: 
567:                 # Total loss
568:                 loss = (
569:                     policy_loss
570:                     + self.value_coef * value_loss
571:                     + self.entropy_coef * entropy_loss
572:                 )
573: 
574:                 # Backward pass and optimizer step
575:                 self.optimizer.zero_grad()
576:                 loss.backward()
577:                 torch.nn.utils.clip_grad_norm_(
578:                     self.model.parameters(), self.max_grad_norm
579:                 )
580:                 self.optimizer.step()
581: 
582:                 # Track metrics
583:                 total_policy_loss += policy_loss.item()
584:                 total_value_loss += value_loss.item()
585:                 total_entropy += entropy.mean().item()
586:                 num_updates += 1
587: 
588:         return {
589:             "policy_loss": total_policy_loss / max(1, num_updates),
590:             "value_loss": total_value_loss / max(1, num_updates),
591:             "entropy": total_entropy / max(1, num_updates),
592:             "num_updates": num_updates,
593:         }
594: 
595:     def validate(self) -> Dict[str, float]:
596:         """Validate on validation set by running deterministic episodes.
597: 
598:         Runs one episode per validation question with deterministic action
599:         selection (argmax) and computes accuracy and average reward.
600: 
601:         Returns
602:         -------
603:         dict[str, float]
604:             Validation metrics: accuracy, average_reward, avg_episode_length.
605:         """
606:         from qb_env.text_wrapper import TextObservationWrapper
607:         from qb_env.tossup_env import TossupMCEnv
608:         from models.likelihoods import TfIdfLikelihood
609: 
610:         self.model.eval()
611: 
612:         corpus = []
613:         for q in self.train_questions[:100]:
614:             corpus.extend(q.option_profiles)
615:         likelihood_model = TfIdfLikelihood(corpus_texts=corpus)
616: 
617:         correct = 0
618:         total = 0
619:         total_reward = 0.0
620:         total_length = 0
621: 
622:         # Limit validation size for speed
623:         val_questions = self.val_questions[:50]
624: 
625:         with torch.no_grad():
626:             for question in val_questions:
627:                 env = TossupMCEnv(
628:                     questions=[question],
629:                     likelihood_model=likelihood_model,
630:                     K=len(question.options),
631:                     reward_mode="time_penalty",
632:                     wait_penalty=self.reward_time_penalty,
633:                     belief_mode="from_scratch",
634:                 )
635:                 wrapped_env = TextObservationWrapper(env)
636: 
637:                 obs, info = wrapped_env.reset()
638:                 done = False
639:                 episode_reward = 0.0
640:                 episode_length = 0
641: 
642:                 while not done:
643:                     inputs = self.model.tokenizer(
644:                         obs,
645:                         return_tensors="pt",
646:                         padding=True,
647:                         truncation=True,
648:                         max_length=self.max_input_length,
649:                     ).to(self.device)
650: 
651:                     actions, act_info = self.model.select_action(
652:                         inputs["input_ids"],
653:                         inputs["attention_mask"],
654:                         deterministic=True,
655:                     )
656: 
657:                     action = actions.item()
658:                     obs, reward, terminated, truncated, step_info = (
659:                         wrapped_env.step(action)
660:                     )
661:                     done = terminated or truncated
662:                     episode_reward += reward
663:                     episode_length += 1
664: 
665:                 total_reward += episode_reward
666:                 total_length += episode_length
667:                 total += 1
668: 
669:                 # Check if answer was correct
670:                 if step_info.get("correct", False) or step_info.get(
671:                     "forced_correct", False
672:                 ):
673:                     correct += 1
674: 
675:         return {
676:             "accuracy": correct / max(1, total),
677:             "average_reward": total_reward / max(1, total),
678:             "avg_episode_length": total_length / max(1, total),
679:         }
680: 
681:     def train(self) -> Dict[str, Any]:
682:         """Run the full PPO training loop.
683: 
684:         Alternates between rollout collection and policy updates for
685:         ``self.iterations`` cycles. Periodically validates and saves
686:         checkpoints.
687: 
688:         Returns
689:         -------
690:         dict[str, Any]
691:             Training summary: best_val_reward, total_iterations.
692:         """
693:         print(f"Starting PPO training for {self.iterations} iterations")
694:         print(f"  Training questions: {len(self.train_questions)}")
695:         print(f"  Validation questions: {len(self.val_questions)}")
696:         print(f"  Batch size: {self.batch_size}")
697:         print(f"  Episodes per iteration: {self.episodes_per_iter}")
698:         print(f"  Device: {self.device}")
699:         print()
700: 
701:         for iteration in range(self.iterations):
702:             self.current_iteration = iteration
703: 
704:             # Collect rollouts
705:             print(f"\nIteration {iteration + 1}/{self.iterations}")
706:             print("  Collecting rollouts...")
707:             buffer = self.collect_rollouts(self.episodes_per_iter)
708: 
709:             # Compute episode statistics
710:             episode_rewards = []
711:             episode_lengths = []
712:             for rollout in buffer.rollouts:
713:                 episode_reward = sum(step.reward for step in rollout)
714:                 episode_rewards.append(episode_reward)
715:                 episode_lengths.append(len(rollout))
716: 
717:             avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
718:             avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
719: 
720:             print(f"  Avg episode reward: {avg_reward:.4f}")
721:             print(f"  Avg episode length: {avg_length:.2f}")
722: 
723:             # Update policy
724:             print("  Updating policy...")
725:             update_metrics = self.update_policy(buffer)
726: 
727:             print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
728:             print(f"  Value loss: {update_metrics['value_loss']:.4f}")
729:             print(f"  Entropy: {update_metrics['entropy']:.4f}")
730: 
731:             # Validate periodically
732:             if (iteration + 1) % self.eval_interval == 0:
733:                 print("\n  Validating...")
734:                 val_summary = self.validate()
735:                 val_reward = val_summary.get("average_reward", 0.0)
736: 
737:                 print(f"  Val Accuracy: {val_summary['accuracy']:.4f}")
738:                 print(f"  Val Reward: {val_reward:.4f}")
739:                 print(
740:                     f"  Val Avg Length: {val_summary['avg_episode_length']:.2f}"
741:                 )
742: 
743:                 # Save history
744:                 self.history.append(
745:                     {
746:                         "iteration": iteration + 1,
747:                         "train_reward": float(avg_reward),
748:                         "train_length": float(avg_length),
749:                         **update_metrics,
750:                         "val": val_summary,
751:                     }
752:                 )
753: 
754:                 # Save best model
755:                 if val_reward > self.best_val_reward:
756:                     self.best_val_reward = val_reward
757:                     self.save_checkpoint(is_best=True)
758:                     print(
759:                         f"  -> New best validation reward: {val_reward:.4f}"
760:                     )
761: 
762:             # Save regular checkpoint
763:             if (iteration + 1) % self.save_interval == 0:
764:                 self.save_checkpoint(is_best=False)
765:                 self.save_history()
766: 
767:         print("\n" + "=" * 60)
768:         print("PPO training completed!")
769:         print(f"Best validation reward: {self.best_val_reward:.4f}")
770:         print("=" * 60)
771: 
772:         # Save final history
773:         self.save_history()
774: 
775:         return {
776:             "best_val_reward": self.best_val_reward,
777:             "total_iterations": self.iterations,
778:         }
779: 
780:     def save_checkpoint(self, is_best: bool = False) -> Path:
781:         """Save model checkpoint to disk.
782: 
783:         Parameters
784:         ----------
785:         is_best : bool
786:             If True, save to ``best_model/`` directory.
787: 
788:         Returns
789:         -------
790:         Path
791:             Path to the saved checkpoint directory.
792:         """
793:         if is_best:
794:             save_path = self.checkpoint_dir / "best_model"
795:         else:
796:             save_path = (
797:                 self.checkpoint_dir
798:                 / f"iter_{self.current_iteration + 1}"
799:             )
800: 
801:         # Use T5PolicyModel's save() method
802:         self.model.save(str(save_path))
803: 
804:         # Save training state
805:         state = {
806:             "iteration": self.current_iteration + 1,
807:             "best_val_reward": self.best_val_reward,
808:             "optimizer_state_dict": self.optimizer.state_dict(),
809:         }
810:         torch.save(state, save_path / "training_state.pt")
811: 
812:         print(f"  Checkpoint saved to {save_path}")
813:         return save_path
814: 
815:     def save_history(self) -> Path:
816:         """Save training history to JSON.
817: 
818:         Returns
819:         -------
820:         Path
821:             Path to the saved history file.
822:         """
823:         history_path = self.checkpoint_dir / "history.json"
824:         with open(history_path, "w") as f:
825:             json.dump(self.history, f, indent=2, default=float)
826:         return history_path
827: 
828: 
829: def run_ppo_training(
830:     config: Dict[str, Any],
831:     train_questions: List[MCQuestion],
832:     val_questions: List[MCQuestion],
833:     test_questions: Optional[List[MCQuestion]] = None,
834:     pretrained_model_path: Optional[str] = None,
835: ) -> Tuple[T5PolicyModel, PPOTrainer]:
836:     """Run the PPO training pipeline with optional pretrained model.
837: 
838:     Creates or loads a T5PolicyModel, trains it with PPO on quiz bowl
839:     episodes, and optionally evaluates on a test set.
840: 
841:     Parameters
842:     ----------
843:     config : dict[str, Any]
844:         Configuration dictionary with model and PPO hyperparameters.
845:     train_questions : list[MCQuestion]
846:         Training set questions.
847:     val_questions : list[MCQuestion]
848:         Validation set questions.
849:     test_questions : list[MCQuestion] or None
850:         Optional test set for final evaluation.
851:     pretrained_model_path : str or None
852:         Path to a supervised pretrained checkpoint. If provided, loads the
853:         model from this path. Otherwise creates a new model.
854: 
855:     Returns
856:     -------
857:     model : T5PolicyModel
858:         The trained model.
859:     trainer : PPOTrainer
860:         The trainer instance with training history.
861:     """
862:     print("=" * 60)
863:     print("PPO TRAINING PHASE (T5 Policy)")
864:     print("=" * 60)
865: 
866:     # Load or create model
867:     if pretrained_model_path:
868:         print(f"Loading pretrained model from {pretrained_model_path}")
869:         device = config.get("device", "cpu")
870:         if device == "auto":
871:             if torch.cuda.is_available():
872:                 device = "cuda"
873:             elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
874:                 device = "mps"
875:             else:
876:                 device = "cpu"
877:         model = T5PolicyModel.load_pretrained(
878:             pretrained_model_path, device=device
879:         )
880:     else:
881:         print("Initializing new model (no pretraining)")
882:         model_config = {
883:             "model_name": config.get("model_name", "t5-large"),
884:             "device": config.get("device", "cpu"),
885:             "max_input_length": config.get("max_input_length", 512),
886:             "num_choices": config.get("num_choices", 4),
887:         }
888:         model = T5PolicyModel(model_config)
889: 
890:     # Create trainer
891:     trainer = PPOTrainer(
892:         model=model,
893:         train_questions=train_questions,
894:         val_questions=val_questions,
895:         config=config,
896:     )
897: 
898:     # Train
899:     summary = trainer.train()
900: 
901:     # Evaluate on test set if provided
902:     if test_questions is not None:
903:         print("\n" + "=" * 60)
904:         print("FINAL EVALUATION ON TEST SET")
905:         print("=" * 60)
906: 
907:         # Load best model if it exists
908:         best_model_path = trainer.checkpoint_dir / "best_model"
909:         if best_model_path.exists():
910:             print(f"Loading best model from {best_model_path}")
911:             model.load(str(best_model_path))
912: 
913:         # Run validation on test set
914:         # Temporarily swap val questions with test questions
915:         original_val = trainer.val_questions
916:         trainer.val_questions = list(test_questions)
917:         test_metrics = trainer.validate()
918:         trainer.val_questions = original_val
919: 
920:         print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
921:         print(f"Test Avg Reward: {test_metrics['average_reward']:.4f}")
922: 
923:         # Save test results
924:         test_results = {
925:             "test_metrics": test_metrics,
926:             "training_summary": summary,
927:         }
928:         results_path = trainer.checkpoint_dir / "test_results.json"
929:         with open(results_path, "w") as f:
930:             json.dump(test_results, f, indent=2, default=float)
931:         print(f"Test results saved to {results_path}")
932: 
933:     return model, trainer
````

## File: pyproject.toml
````toml
 1: [build-system]
 2: requires = ["setuptools>=69.0"]
 3: build-backend = "setuptools.build_meta"
 4: 
 5: [project]
 6: name = "qanta-buzzer"
 7: version = "1.0.0"
 8: description = "Unified quiz bowl RL buzzer system for Stanford CS234"
 9: readme = "README.md"
10: requires-python = ">=3.11"
11: dependencies = [
12:   "datasets>=2.14.0",
13:   "gymnasium>=1.1.0",
14:   "jsonlines>=3.1.0",
15:   "matplotlib>=3.7.0",
16:   "numpy>=1.24.0",
17:   "pandas>=2.0.0",
18:   "PyYAML>=6.0.0",
19:   "scikit-learn>=1.3.0",
20:   "seaborn>=0.12.0",
21:   "sentence-transformers>=2.2.0",
22:   "stable-baselines3>=2.6.0",
23:   "torch>=2.0.0",
24:   "tqdm>=4.65.0",
25:   "transformers>=4.30.0",
26: ]
27: 
28: [project.optional-dependencies]
29: openai = ["openai>=1.0.0"]
30: maskable = ["sb3-contrib>=2.6.0"]
31: dspy = ["dspy>=2.5.0"]
32: 
33: [tool.setuptools.packages.find]
34: include = ["agents", "evaluation", "models", "qb_data", "qb_env", "training"]
35: 
36: [tool.pytest.ini_options]
37: testpaths = ["tests"]
````

## File: agents/bayesian_buzzer.py
````python
  1: from __future__ import annotations
  2: 
  3: from dataclasses import dataclass
  4: from typing import TYPE_CHECKING
  5: 
  6: import numpy as np
  7: 
  8: from agents._math import sigmoid
  9: from models.likelihoods import LikelihoodModel
 10: from qb_data.mc_builder import MCQuestion
 11: 
 12: if TYPE_CHECKING:
 13:     from agents.threshold_buzzer import _PrecomputedQuestion
 14: 
 15: 
 16: 
 17: @dataclass
 18: class SoftmaxEpisodeResult:
 19:     qid: str
 20:     buzz_step: int
 21:     buzz_index: int
 22:     gold_index: int
 23:     correct: bool
 24:     c_trace: list[float]
 25:     g_trace: list[float]
 26:     top_p_trace: list[float]
 27:     entropy_trace: list[float]
 28: 
 29: 
 30: class SoftmaxProfileBuzzer:
 31:     def __init__(
 32:         self,
 33:         likelihood_model: LikelihoodModel,
 34:         threshold: float = 0.8,
 35:         beta: float = 5.0,
 36:         alpha: float = 10.0,
 37:     ):
 38:         self.likelihood_model = likelihood_model
 39:         self.threshold = threshold
 40:         self.beta = beta
 41:         self.alpha = alpha
 42:         self.belief: np.ndarray | None = None
 43: 
 44:     def _belief_from_scratch(self, cumulative_prefix: str, option_profiles: list[str]) -> np.ndarray:
 45:         scores = self.likelihood_model.score(cumulative_prefix, option_profiles)
 46:         scores = scores - np.max(scores)
 47:         probs = np.exp(self.beta * scores)
 48:         probs = probs / max(1e-12, probs.sum())
 49:         return probs.astype(np.float32)
 50: 
 51:     def confidence_proxy(self, top_p: float) -> float:
 52:         return sigmoid(self.alpha * (top_p - self.threshold))
 53: 
 54:     def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
 55:         c_trace: list[float] = []
 56:         g_trace: list[float] = []
 57:         top_p_trace: list[float] = []
 58:         entropy_trace: list[float] = []
 59: 
 60:         chosen_idx = 0
 61:         chosen_step = len(question.cumulative_prefixes) - 1
 62: 
 63:         for step_idx, prefix in enumerate(question.cumulative_prefixes):
 64:             belief = self._belief_from_scratch(prefix, question.option_profiles)
 65:             self.belief = belief
 66:             top_idx = int(np.argmax(belief))
 67:             top_p = float(np.max(belief))
 68:             entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
 69:             c_t = self.confidence_proxy(top_p)
 70:             g_t = 1.0 if top_idx == question.gold_index else 0.0
 71: 
 72:             c_trace.append(c_t)
 73:             g_trace.append(g_t)
 74:             top_p_trace.append(top_p)
 75:             entropy_trace.append(entropy)
 76: 
 77:             is_last = step_idx == len(question.cumulative_prefixes) - 1
 78:             if top_p >= self.threshold or is_last:
 79:                 chosen_step = step_idx
 80:                 chosen_idx = top_idx
 81:                 break
 82: 
 83:         return SoftmaxEpisodeResult(
 84:             qid=question.qid,
 85:             buzz_step=chosen_step,
 86:             buzz_index=chosen_idx,
 87:             gold_index=question.gold_index,
 88:             correct=(chosen_idx == question.gold_index),
 89:             c_trace=c_trace,
 90:             g_trace=g_trace,
 91:             top_p_trace=top_p_trace,
 92:             entropy_trace=entropy_trace,
 93:         )
 94: 
 95: 
 96: class SequentialBayesBuzzer:
 97:     def __init__(
 98:         self,
 99:         likelihood_model: LikelihoodModel,
100:         threshold: float = 0.8,
101:         beta: float = 5.0,
102:         alpha: float = 10.0,
103:     ):
104:         self.likelihood_model = likelihood_model
105:         self.threshold = threshold
106:         self.beta = beta
107:         self.alpha = alpha
108: 
109:     def _step_update(self, prior: np.ndarray, fragment: str, option_profiles: list[str]) -> np.ndarray:
110:         scores = self.likelihood_model.score(fragment, option_profiles)
111:         scores = scores - np.max(scores)
112:         likelihood = np.exp(self.beta * scores)
113:         posterior = prior * likelihood
114:         denom = posterior.sum()
115:         if denom <= 0:
116:             return np.ones_like(prior) / len(prior)
117:         return (posterior / denom).astype(np.float32)
118: 
119:     def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
120:         c_trace: list[float] = []
121:         g_trace: list[float] = []
122:         top_p_trace: list[float] = []
123:         entropy_trace: list[float] = []
124: 
125:         K = len(question.options)
126:         belief = np.ones(K, dtype=np.float32) / K
127:         chosen_idx = 0
128:         chosen_step = len(question.cumulative_prefixes) - 1
129: 
130:         for step_idx, token_idx in enumerate(question.run_indices):
131:             prev_token_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
132:             fragment = " ".join(question.tokens[prev_token_idx + 1 : token_idx + 1])
133:             belief = self._step_update(belief, fragment, question.option_profiles)
134:             top_idx = int(np.argmax(belief))
135:             top_p = float(np.max(belief))
136:             entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
137:             c_t = sigmoid(self.alpha * (top_p - self.threshold))
138:             g_t = 1.0 if top_idx == question.gold_index else 0.0
139: 
140:             c_trace.append(c_t)
141:             g_trace.append(g_t)
142:             top_p_trace.append(top_p)
143:             entropy_trace.append(entropy)
144: 
145:             is_last = step_idx == len(question.cumulative_prefixes) - 1
146:             if top_p >= self.threshold or is_last:
147:                 chosen_step = step_idx
148:                 chosen_idx = top_idx
149:                 break
150: 
151:         return SoftmaxEpisodeResult(
152:             qid=question.qid,
153:             buzz_step=chosen_step,
154:             buzz_index=chosen_idx,
155:             gold_index=question.gold_index,
156:             correct=(chosen_idx == question.gold_index),
157:             c_trace=c_trace,
158:             g_trace=g_trace,
159:             top_p_trace=top_p_trace,
160:             entropy_trace=entropy_trace,
161:         )
162: 
163: 
164: def precompute_sequential_beliefs(
165:     questions: list[MCQuestion],
166:     likelihood_model: LikelihoodModel,
167:     beta: float,
168: ) -> list["_PrecomputedQuestion"]:
169:     """Compute Bayesian sequential beliefs at every step for every question.
170: 
171:     Starts with a uniform prior and applies Bayesian update
172:     ``posterior = prior * likelihood`` using token fragments derived from
173:     ``question.run_indices``.  Returns one ``_PrecomputedQuestion`` per
174:     question where ``beliefs`` are the Bayesian posteriors (NOT the
175:     from-scratch softmax beliefs).
176:     """
177:     from agents.threshold_buzzer import _PrecomputedQuestion
178: 
179:     out: list[_PrecomputedQuestion] = []
180:     for q in questions:
181:         K = len(q.options)
182:         belief = np.ones(K, dtype=np.float32) / K
183:         beliefs: list[np.ndarray] = []
184: 
185:         for step_idx, token_idx in enumerate(q.run_indices):
186:             prev_token_idx = q.run_indices[step_idx - 1] if step_idx > 0 else -1
187:             fragment = " ".join(q.tokens[prev_token_idx + 1 : token_idx + 1])
188:             scores = likelihood_model.score(fragment, q.option_profiles)
189:             scores = scores - np.max(scores)
190:             likelihood = np.exp(beta * scores)
191:             posterior = belief * likelihood
192:             denom = posterior.sum()
193:             if denom <= 0:
194:                 belief = np.ones_like(belief) / len(belief)
195:             else:
196:                 belief = (posterior / denom).astype(np.float32)
197:             beliefs.append(belief.copy())
198: 
199:         out.append(_PrecomputedQuestion(
200:             qid=q.qid,
201:             gold_index=q.gold_index,
202:             num_options=K,
203:             beliefs=beliefs,
204:         ))
205:     return out
206: 
207: 
208: def _sequential_episode_from_precomputed(
209:     pq: "_PrecomputedQuestion",
210:     threshold: float,
211:     alpha: float,
212: ) -> SoftmaxEpisodeResult:
213:     """Build a SoftmaxEpisodeResult from pre-computed sequential beliefs.
214: 
215:     Identical buzzing logic to ``SequentialBayesBuzzer.run_episode`` but
216:     reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
217:     likelihood model.
218:     """
219:     from agents.threshold_buzzer import _belief_stats
220: 
221:     c_trace: list[float] = []
222:     g_trace: list[float] = []
223:     top_p_trace: list[float] = []
224:     entropy_trace: list[float] = []
225: 
226:     chosen_step = len(pq.beliefs) - 1
227:     chosen_idx = 0
228: 
229:     for step_idx, belief in enumerate(pq.beliefs):
230:         top_idx, top_p, entropy = _belief_stats(belief)
231:         c_t = sigmoid(alpha * (top_p - threshold))
232:         g_t = 1.0 if top_idx == pq.gold_index else 0.0
233: 
234:         c_trace.append(c_t)
235:         g_trace.append(g_t)
236:         top_p_trace.append(top_p)
237:         entropy_trace.append(entropy)
238: 
239:         is_last = step_idx == len(pq.beliefs) - 1
240:         if top_p >= threshold or is_last:
241:             chosen_step = step_idx
242:             chosen_idx = top_idx
243:             break
244: 
245:     correct = chosen_idx == pq.gold_index
246:     return SoftmaxEpisodeResult(
247:         qid=pq.qid,
248:         buzz_step=chosen_step,
249:         buzz_index=chosen_idx,
250:         gold_index=pq.gold_index,
251:         correct=correct,
252:         c_trace=c_trace,
253:         g_trace=g_trace,
254:         top_p_trace=top_p_trace,
255:         entropy_trace=entropy_trace,
256:     )
257: 
258: 
259: def sweep_sequential_thresholds(
260:     questions: list[MCQuestion],
261:     likelihood_model: LikelihoodModel,
262:     thresholds: list[float],
263:     beta: float = 5.0,
264:     alpha: float = 10.0,
265:     precomputed: list["_PrecomputedQuestion"] | None = None,
266: ) -> dict[float, list[SoftmaxEpisodeResult]]:
267:     """Sweep multiple thresholds with a single sequential belief pass.
268: 
269:     If *precomputed* is provided the expensive model calls are skipped
270:     entirely and the sweep is pure numpy.  Otherwise beliefs are computed
271:     once internally and reused across thresholds.
272:     """
273:     if precomputed is None:
274:         precomputed = precompute_sequential_beliefs(questions, likelihood_model, beta)
275: 
276:     out: dict[float, list[SoftmaxEpisodeResult]] = {}
277:     for threshold in thresholds:
278:         out[float(threshold)] = [
279:             _sequential_episode_from_precomputed(pq, threshold, alpha)
280:             for pq in precomputed
281:         ]
282:     return out
````

## File: agents/ppo_buzzer.py
````python
  1: """PPO Buzzer agent wrapping Stable-Baselines3's PPO.
  2: 
  3: Provides the PPOBuzzer class for training an MLP policy on belief-feature
  4: observations from TossupMCEnv, and PPOEpisodeTrace for recording per-step
  5: action probabilities needed to compute the S_q scoring metric.
  6: 
  7: The key design rationale: SB3's ``learn()`` does not expose per-step action
  8: distributions, so ``run_episode()`` implements custom episode execution that
  9: records c_trace (buzz probability) and g_trace (correctness probability)
 10: at each step for downstream S_q computation.
 11: 
 12: Ported from qb-rl reference implementation (agents/ppo_buzzer.py) with
 13: import path adaptations for the unified qanta-buzzer codebase.
 14: """
 15: 
 16: from __future__ import annotations
 17: 
 18: from dataclasses import dataclass
 19: from pathlib import Path
 20: from typing import Any
 21: 
 22: import numpy as np
 23: import torch as th
 24: from stable_baselines3 import PPO
 25: 
 26: from qb_env.tossup_env import TossupMCEnv
 27: 
 28: 
 29: @dataclass
 30: class PPOEpisodeTrace:
 31:     """Record of a single episode with per-step action probability traces.
 32: 
 33:     Used to compute the S_q scoring metric: S_q = sum(c_t * g_t) over steps,
 34:     and calibration metrics (ECE, Brier) via ``top_p_trace``.
 35: 
 36:     Attributes
 37:     ----------
 38:     qid : str
 39:         Question identifier.
 40:     buzz_step : int
 41:         Step at which the agent buzzed (-1 if never buzzed voluntarily).
 42:     buzz_index : int
 43:         Index of the chosen answer option (0-based, -1 if forced).
 44:     gold_index : int
 45:         Index of the correct answer option (0-based).
 46:     correct : bool
 47:         Whether the agent selected the correct answer.
 48:     episode_reward : float
 49:         Total accumulated reward over the episode.
 50:     c_trace : list[float]
 51:         Per-step buzz probability: 1 - P(wait) at each timestep.
 52:     g_trace : list[float]
 53:         Per-step correctness probability: P(gold_option) / P(buzz).
 54:     top_p_trace : list[float]
 55:         Per-step max belief probability: max(env.belief). Used as the
 56:         confidence proxy for calibration metrics, consistent with
 57:         baseline agents.
 58:     entropy_trace : list[float]
 59:         Per-step policy entropy over the full action distribution.
 60:     """
 61: 
 62:     qid: str
 63:     buzz_step: int
 64:     buzz_index: int
 65:     gold_index: int
 66:     correct: bool
 67:     episode_reward: float
 68:     c_trace: list[float]
 69:     g_trace: list[float]
 70:     top_p_trace: list[float]
 71:     entropy_trace: list[float]
 72: 
 73: 
 74: class PPOBuzzer:
 75:     """PPO-trained buzzer agent wrapping Stable-Baselines3's PPO.
 76: 
 77:     Trains an MLP policy on belief-feature observations (Box(K+6,)) from
 78:     TossupMCEnv. The policy maps observation vectors to a Discrete(K+1)
 79:     action space: WAIT (0) or BUZZ with option i (1..K).
 80: 
 81:     Parameters
 82:     ----------
 83:     env : TossupMCEnv
 84:         Gymnasium environment with belief-feature observations.
 85:     learning_rate : float
 86:         Learning rate for the Adam optimizer.
 87:     n_steps : int
 88:         Number of steps per rollout buffer collection.
 89:     batch_size : int
 90:         Minibatch size for PPO updates.
 91:     n_epochs : int
 92:         Number of optimization epochs per rollout.
 93:     gamma : float
 94:         Discount factor for return computation.
 95:     policy_kwargs : dict or None
 96:         Additional keyword arguments for the MLP policy. Defaults to
 97:         ``{"net_arch": [64, 64]}`` (two hidden layers of 64 units).
 98:     verbose : int
 99:         SB3 verbosity level (0=silent, 1=info, 2=debug).
100:     """
101: 
102:     def __init__(
103:         self,
104:         env: TossupMCEnv,
105:         learning_rate: float = 3e-4,
106:         n_steps: int = 128,
107:         batch_size: int = 32,
108:         n_epochs: int = 10,
109:         gamma: float = 0.99,
110:         seed: int | None = None,
111:         policy_kwargs: dict[str, Any] | None = None,
112:         verbose: int = 0,
113:         use_maskable_ppo: bool = False,
114:     ):
115:         if policy_kwargs is None:
116:             policy_kwargs = {"net_arch": [64, 64]}
117: 
118:         self.env = env
119:         self._use_maskable = use_maskable_ppo
120: 
121:         if use_maskable_ppo:
122:             try:
123:                 from sb3_contrib import MaskablePPO
124:             except ImportError as exc:
125:                 raise ImportError(
126:                     "MaskablePPO requires sb3-contrib. "
127:                     "Install with: pip install -e '.[maskable]'"
128:                 ) from exc
129:             self.model = MaskablePPO(
130:                 "MlpPolicy",
131:                 env,
132:                 verbose=verbose,
133:                 seed=seed,
134:                 learning_rate=learning_rate,
135:                 n_steps=n_steps,
136:                 batch_size=batch_size,
137:                 n_epochs=n_epochs,
138:                 gamma=gamma,
139:                 policy_kwargs=policy_kwargs,
140:             )
141:         else:
142:             self.model = PPO(
143:                 "MlpPolicy",
144:                 env,
145:                 verbose=verbose,
146:                 seed=seed,
147:                 learning_rate=learning_rate,
148:                 n_steps=n_steps,
149:                 batch_size=batch_size,
150:                 n_epochs=n_epochs,
151:                 gamma=gamma,
152:                 policy_kwargs=policy_kwargs,
153:             )
154: 
155:     def train(self, total_timesteps: int = 100_000) -> None:
156:         """Train the PPO policy for the specified number of timesteps.
157: 
158:         Parameters
159:         ----------
160:         total_timesteps : int
161:             Total environment steps to collect during training.
162:         """
163:         self.model.learn(total_timesteps=total_timesteps)
164: 
165:     def save(self, path: str | Path) -> None:
166:         """Save the trained PPO model to disk.
167: 
168:         Parameters
169:         ----------
170:         path : str or Path
171:             File path for the saved model (SB3 appends .zip if needed).
172:         """
173:         self.model.save(str(path))
174: 
175:     @classmethod
176:     def load(cls, path: str | Path, env: TossupMCEnv) -> "PPOBuzzer":
177:         """Load a previously saved PPO model.
178: 
179:         Parameters
180:         ----------
181:         path : str or Path
182:             Path to the saved model file.
183:         env : TossupMCEnv
184:             Environment to attach to the loaded model.
185: 
186:         Returns
187:         -------
188:         PPOBuzzer
189:             A PPOBuzzer with the loaded model weights.
190:         """
191:         agent = cls(env=env)
192:         agent.model = PPO.load(str(path), env=env)
193:         return agent
194: 
195:     def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
196:         """Extract action probabilities from the policy for a given observation.
197: 
198:         Parameters
199:         ----------
200:         obs : np.ndarray
201:             Observation vector of shape (K + 6,).
202: 
203:         Returns
204:         -------
205:         np.ndarray
206:             Action probability vector of shape (K + 1,), dtype float32.
207:             Index 0 = P(wait), indices 1..K = P(buzz with option i).
208:         """
209:         obs_tensor = th.as_tensor(
210:             obs, dtype=th.float32, device=self.model.device
211:         ).unsqueeze(0)
212:         dist = self.model.policy.get_distribution(obs_tensor)
213:         probs = dist.distribution.probs[0].detach().cpu().numpy()
214:         return probs.astype(np.float32)
215: 
216:     def _base_env(self) -> TossupMCEnv:
217:         """Return the underlying TossupMCEnv, unwrapping if needed."""
218:         return getattr(self.env, "unwrapped", self.env)
219: 
220:     def c_t(self, obs: np.ndarray) -> float:
221:         """Compute buzz probability at the current step.
222: 
223:         Parameters
224:         ----------
225:         obs : np.ndarray
226:             Observation vector of shape (K + 6,).
227: 
228:         Returns
229:         -------
230:         float
231:             Probability of buzzing: 1 - P(wait). Range [0, 1].
232:         """
233:         probs = self.action_probabilities(obs)
234:         return float(1.0 - probs[0])
235: 
236:     def g_t(self, obs: np.ndarray, gold_index: int) -> float:
237:         """Compute correctness probability at the current step.
238: 
239:         Given that the agent buzzes, what is the probability it selects
240:         the correct answer? Formally: P(gold_action) / P(buzz).
241: 
242:         Parameters
243:         ----------
244:         obs : np.ndarray
245:             Observation vector of shape (K + 6,).
246:         gold_index : int
247:             Index of the correct answer option (0-based).
248: 
249:         Returns
250:         -------
251:         float
252:             Conditional correctness probability. Returns 0.0 if buzz
253:             probability is near zero (< 1e-12).
254:         """
255:         probs = self.action_probabilities(obs)
256:         base_env = self._base_env()
257:         c_t = float(1.0 - probs[0])
258:         if c_t <= 1e-12:
259:             return 0.0
260:         if len(probs) == 2:
261:             if gold_index < 0 or base_env.belief is None:
262:                 return 0.0
263:             return float(base_env.belief[gold_index])
264:         return float(probs[gold_index + 1] / c_t)
265: 
266:     def run_episode(
267:         self,
268:         deterministic: bool = False,
269:         seed: int | None = None,
270:         question_idx: int | None = None,
271:     ) -> PPOEpisodeTrace:
272:         """Run a full episode and record per-step action probability traces.
273: 
274:         Executes the policy in the environment, computing c_trace (buzz
275:         probability), g_trace (correctness probability), and entropy_trace
276:         at each step. These traces are needed to compute the S_q metric.
277: 
278:         Parameters
279:         ----------
280:         deterministic : bool
281:             If True, select actions by argmax instead of sampling.
282:         seed : int or None
283:             If provided, seeds the environment reset for reproducibility.
284: 
285:         Returns
286:         -------
287:         PPOEpisodeTrace
288:             Complete episode record with action traces and outcome.
289:         """
290:         reset_options = None
291:         if question_idx is not None:
292:             reset_options = {"question_idx": int(question_idx)}
293: 
294:         obs, info = self.env.reset(seed=seed, options=reset_options)
295:         terminated = False
296:         truncated = False
297:         total_reward = 0.0
298:         c_trace: list[float] = []
299:         g_trace: list[float] = []
300:         top_p_trace: list[float] = []
301:         entropy_trace: list[float] = []
302: 
303:         buzz_step = -1
304:         buzz_index = -1
305:         gold_index = (
306:             self.env.question.gold_index if self.env.question is not None else -1
307:         )
308:         base_env = self._base_env()
309: 
310:         while not (terminated or truncated):
311:             probs = self.action_probabilities(obs)
312:             c_val = float(probs[1] if len(probs) == 2 else 1.0 - probs[0])
313:             if len(probs) == 2:
314:                 g_val = (
315:                     float(base_env.belief[gold_index])
316:                     if gold_index >= 0 and base_env.belief is not None
317:                     else 0.0
318:                 )
319:             else:
320:                 g_val = (
321:                     float(probs[gold_index + 1] / c_val) if c_val > 1e-12 else 0.0
322:                 )
323:             entropy = float(
324:                 -(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))).sum()
325:             )
326: 
327:             top_p_val = float(np.max(base_env.belief)) if base_env.belief is not None else c_val
328:             c_trace.append(c_val)
329:             g_trace.append(g_val)
330:             top_p_trace.append(top_p_val)
331:             entropy_trace.append(entropy)
332: 
333:             if deterministic:
334:                 action = int(np.argmax(probs))
335:             else:
336:                 action = int(np.random.choice(len(probs), p=probs))
337: 
338:             obs, reward, terminated, truncated, step_info = self.env.step(action)
339:             total_reward += reward
340: 
341:             if action != 0 and buzz_step < 0:
342:                 buzz_step = int(step_info.get("step_idx", 0))
343:                 if len(probs) == 2:
344:                     buzz_index = int(
345:                         step_info.get(
346:                             "chosen_idx",
347:                             step_info.get("forced_choice", np.argmax(base_env.belief)),
348:                         )
349:                     )
350:                 else:
351:                     buzz_index = action - 1
352:             if truncated and buzz_step < 0 and not step_info.get("no_buzz", False):
353:                 buzz_step = int(
354:                     step_info.get("step_idx", len(c_trace) - 1)
355:                 )
356:                 buzz_index = int(
357:                     step_info.get("forced_choice", np.argmax(base_env.belief))
358:                 )
359: 
360:         correct = buzz_index == gold_index
361:         return PPOEpisodeTrace(
362:             qid=info.get("qid", ""),
363:             buzz_step=buzz_step,
364:             buzz_index=buzz_index,
365:             gold_index=gold_index,
366:             correct=correct,
367:             episode_reward=total_reward,
368:             c_trace=c_trace,
369:             g_trace=g_trace,
370:             top_p_trace=top_p_trace,
371:             entropy_trace=entropy_trace,
372:         )
````

## File: agents/threshold_buzzer.py
````python
  1: from __future__ import annotations
  2: 
  3: from dataclasses import dataclass
  4: from typing import TYPE_CHECKING, Any
  5: 
  6: import numpy as np
  7: 
  8: from agents._math import sigmoid
  9: from models.likelihoods import LikelihoodModel
 10: from qb_data.mc_builder import MCQuestion
 11: 
 12: if TYPE_CHECKING:
 13:     from agents.bayesian_buzzer import SoftmaxEpisodeResult
 14: 
 15: 
 16: @dataclass
 17: class EpisodeResult:
 18:     qid: str
 19:     buzz_step: int
 20:     buzz_index: int
 21:     gold_index: int
 22:     correct: bool
 23:     reward_like: float
 24:     c_trace: list[float]
 25:     g_trace: list[float]
 26:     top_p_trace: list[float]
 27:     entropy_trace: list[float]
 28: 
 29: 
 30: def _scores_to_belief(scores: np.ndarray, beta: float) -> np.ndarray:
 31:     """Convert raw similarity scores to a belief distribution via softmax."""
 32:     shifted = scores - np.max(scores)
 33:     probs = np.exp(beta * shifted)
 34:     probs = probs / max(1e-12, probs.sum())
 35:     return probs.astype(np.float32)
 36: 
 37: 
 38: def _belief_stats(belief: np.ndarray) -> tuple[int, float, float]:
 39:     """Return (top_idx, top_p, entropy) from a belief distribution."""
 40:     top_idx = int(np.argmax(belief))
 41:     top_p = float(belief[top_idx])
 42:     clipped = np.clip(belief, 1e-12, 1.0)
 43:     entropy = float(-(clipped * np.log(clipped)).sum())
 44:     return top_idx, top_p, entropy
 45: 
 46: 
 47: @dataclass
 48: class _PrecomputedQuestion:
 49:     """Pre-computed belief distributions for every clue step of one question."""
 50:     qid: str
 51:     gold_index: int
 52:     num_options: int
 53:     beliefs: list[np.ndarray]
 54: 
 55: 
 56: def precompute_beliefs(
 57:     questions: list[MCQuestion],
 58:     likelihood_model: LikelihoodModel,
 59:     beta: float,
 60: ) -> list[_PrecomputedQuestion]:
 61:     """Compute beliefs at every step for every question (single model pass).
 62: 
 63:     After calling ``likelihood_model.precompute_embeddings()`` this is
 64:     pure cache lookups + numpy math, so it runs in seconds rather than
 65:     hours.
 66:     """
 67:     from tqdm import tqdm
 68: 
 69:     out: list[_PrecomputedQuestion] = []
 70:     for q in tqdm(questions, desc="Computing beliefs"):
 71:         beliefs = [
 72:             _scores_to_belief(
 73:                 likelihood_model.score(prefix, q.option_profiles), beta
 74:             )
 75:             for prefix in q.cumulative_prefixes
 76:         ]
 77:         out.append(_PrecomputedQuestion(
 78:             qid=q.qid,
 79:             gold_index=q.gold_index,
 80:             num_options=len(q.options),
 81:             beliefs=beliefs,
 82:         ))
 83:     return out
 84: 
 85: 
 86: class ThresholdBuzzer:
 87:     def __init__(
 88:         self,
 89:         likelihood_model: LikelihoodModel,
 90:         threshold: float = 0.8,
 91:         beta: float = 5.0,
 92:         alpha: float = 10.0,
 93:     ):
 94:         self.likelihood_model = likelihood_model
 95:         self.threshold = threshold
 96:         self.beta = beta
 97:         self.alpha = alpha
 98:         self.belief: np.ndarray | None = None
 99: 
100:     def _belief_from_prefix(self, prefix: str, option_profiles: list[str]) -> np.ndarray:
101:         scores = self.likelihood_model.score(prefix, option_profiles)
102:         return _scores_to_belief(scores, self.beta)
103: 
104:     def _confidence_proxy(self, top_p: float) -> float:
105:         return sigmoid(self.alpha * (top_p - self.threshold))
106: 
107:     def run_episode(self, question: MCQuestion) -> EpisodeResult:
108:         c_trace: list[float] = []
109:         g_trace: list[float] = []
110:         top_p_trace: list[float] = []
111:         entropy_trace: list[float] = []
112: 
113:         chosen_step = len(question.cumulative_prefixes) - 1
114:         chosen_idx = 0
115: 
116:         for step_idx, prefix in enumerate(question.cumulative_prefixes):
117:             belief = self._belief_from_prefix(prefix, question.option_profiles)
118:             self.belief = belief
119:             top_idx, top_p, entropy = _belief_stats(belief)
120:             c_t = self._confidence_proxy(top_p)
121:             g_t = 1.0 if top_idx == question.gold_index else 0.0
122: 
123:             c_trace.append(c_t)
124:             g_trace.append(g_t)
125:             top_p_trace.append(top_p)
126:             entropy_trace.append(entropy)
127: 
128:             is_last = step_idx == len(question.cumulative_prefixes) - 1
129:             if top_p >= self.threshold or is_last:
130:                 chosen_step = step_idx
131:                 chosen_idx = top_idx
132:                 break
133: 
134:         correct = chosen_idx == question.gold_index
135:         reward_like = 1.0 if correct else -0.5
136:         return EpisodeResult(
137:             qid=question.qid,
138:             buzz_step=chosen_step,
139:             buzz_index=chosen_idx,
140:             gold_index=question.gold_index,
141:             correct=correct,
142:             reward_like=reward_like,
143:             c_trace=c_trace,
144:             g_trace=g_trace,
145:             top_p_trace=top_p_trace,
146:             entropy_trace=entropy_trace,
147:         )
148: 
149: 
150: class AlwaysBuzzFinalBuzzer:
151:     def __init__(self, likelihood_model: LikelihoodModel, beta: float = 5.0):
152:         self.likelihood_model = likelihood_model
153:         self.beta = beta
154: 
155:     def run_episode(self, question: MCQuestion) -> EpisodeResult:
156:         c_trace: list[float] = []
157:         g_trace: list[float] = []
158:         top_p_trace: list[float] = []
159:         entropy_trace: list[float] = []
160: 
161:         final_step = len(question.cumulative_prefixes) - 1
162:         final_belief = np.ones(len(question.options), dtype=np.float32) / len(question.options)
163:         for prefix in question.cumulative_prefixes:
164:             scores = self.likelihood_model.score(prefix, question.option_profiles)
165:             probs = _scores_to_belief(scores, self.beta)
166:             final_belief = probs
167:             top_idx, top_p, entropy = _belief_stats(probs)
168:             c_trace.append(0.0)
169:             g_trace.append(1.0 if top_idx == question.gold_index else 0.0)
170:             top_p_trace.append(top_p)
171:             entropy_trace.append(entropy)
172: 
173:         c_trace[-1] = 1.0
174:         buzz_idx = int(np.argmax(final_belief))
175:         correct = buzz_idx == question.gold_index
176:         reward_like = 1.0 if correct else -0.5
177:         return EpisodeResult(
178:             qid=question.qid,
179:             buzz_step=final_step,
180:             buzz_index=buzz_idx,
181:             gold_index=question.gold_index,
182:             correct=correct,
183:             reward_like=reward_like,
184:             c_trace=c_trace,
185:             g_trace=g_trace,
186:             top_p_trace=top_p_trace,
187:             entropy_trace=entropy_trace,
188:         )
189: 
190: 
191: def _softmax_episode_from_precomputed(
192:     pq: _PrecomputedQuestion,
193:     threshold: float,
194:     alpha: float,
195: ) -> "SoftmaxEpisodeResult":
196:     """Build a SoftmaxEpisodeResult from pre-computed beliefs (pure numpy).
197: 
198:     Identical buzzing logic to ``SoftmaxProfileBuzzer.run_episode`` but
199:     reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
200:     likelihood model.
201:     """
202:     from agents.bayesian_buzzer import SoftmaxEpisodeResult
203: 
204:     c_trace: list[float] = []
205:     g_trace: list[float] = []
206:     top_p_trace: list[float] = []
207:     entropy_trace: list[float] = []
208: 
209:     chosen_step = len(pq.beliefs) - 1
210:     chosen_idx = 0
211: 
212:     for step_idx, belief in enumerate(pq.beliefs):
213:         top_idx, top_p, entropy = _belief_stats(belief)
214:         c_t = sigmoid(alpha * (top_p - threshold))
215:         g_t = 1.0 if top_idx == pq.gold_index else 0.0
216: 
217:         c_trace.append(c_t)
218:         g_trace.append(g_t)
219:         top_p_trace.append(top_p)
220:         entropy_trace.append(entropy)
221: 
222:         is_last = step_idx == len(pq.beliefs) - 1
223:         if top_p >= threshold or is_last:
224:             chosen_step = step_idx
225:             chosen_idx = top_idx
226:             break
227: 
228:     correct = chosen_idx == pq.gold_index
229:     return SoftmaxEpisodeResult(
230:         qid=pq.qid,
231:         buzz_step=chosen_step,
232:         buzz_index=chosen_idx,
233:         gold_index=pq.gold_index,
234:         correct=correct,
235:         c_trace=c_trace,
236:         g_trace=g_trace,
237:         top_p_trace=top_p_trace,
238:         entropy_trace=entropy_trace,
239:     )
240: 
241: 
242: def _always_final_from_precomputed(pq: _PrecomputedQuestion) -> EpisodeResult:
243:     """Build an EpisodeResult for AlwaysBuzzFinal from pre-computed beliefs.
244: 
245:     Iterates all beliefs (no early stopping), buzzes at the last step
246:     with argmax of the final belief.
247:     """
248:     c_trace: list[float] = []
249:     g_trace: list[float] = []
250:     top_p_trace: list[float] = []
251:     entropy_trace: list[float] = []
252: 
253:     for belief in pq.beliefs:
254:         top_idx, top_p, entropy = _belief_stats(belief)
255:         g_t = 1.0 if top_idx == pq.gold_index else 0.0
256:         c_trace.append(0.0)
257:         g_trace.append(g_t)
258:         top_p_trace.append(top_p)
259:         entropy_trace.append(entropy)
260: 
261:     c_trace[-1] = 1.0
262:     buzz_idx = int(np.argmax(pq.beliefs[-1]))
263:     correct = buzz_idx == pq.gold_index
264:     return EpisodeResult(
265:         qid=pq.qid,
266:         buzz_step=len(pq.beliefs) - 1,
267:         buzz_index=buzz_idx,
268:         gold_index=pq.gold_index,
269:         correct=correct,
270:         reward_like=1.0 if correct else -0.5,
271:         c_trace=c_trace,
272:         g_trace=g_trace,
273:         top_p_trace=top_p_trace,
274:         entropy_trace=entropy_trace,
275:     )
276: 
277: 
278: def _episode_from_precomputed(
279:     pq: _PrecomputedQuestion,
280:     threshold: float,
281:     alpha: float,
282: ) -> EpisodeResult:
283:     """Build an EpisodeResult from pre-computed beliefs (pure numpy)."""
284:     c_trace: list[float] = []
285:     g_trace: list[float] = []
286:     top_p_trace: list[float] = []
287:     entropy_trace: list[float] = []
288: 
289:     chosen_step = len(pq.beliefs) - 1
290:     chosen_idx = 0
291: 
292:     for step_idx, belief in enumerate(pq.beliefs):
293:         top_idx, top_p, entropy = _belief_stats(belief)
294:         c_t = sigmoid(alpha * (top_p - threshold))
295:         g_t = 1.0 if top_idx == pq.gold_index else 0.0
296: 
297:         c_trace.append(c_t)
298:         g_trace.append(g_t)
299:         top_p_trace.append(top_p)
300:         entropy_trace.append(entropy)
301: 
302:         is_last = step_idx == len(pq.beliefs) - 1
303:         if top_p >= threshold or is_last:
304:             chosen_step = step_idx
305:             chosen_idx = top_idx
306:             break
307: 
308:     correct = chosen_idx == pq.gold_index
309:     return EpisodeResult(
310:         qid=pq.qid,
311:         buzz_step=chosen_step,
312:         buzz_index=chosen_idx,
313:         gold_index=pq.gold_index,
314:         correct=correct,
315:         reward_like=1.0 if correct else -0.5,
316:         c_trace=c_trace,
317:         g_trace=g_trace,
318:         top_p_trace=top_p_trace,
319:         entropy_trace=entropy_trace,
320:     )
321: 
322: 
323: def sweep_thresholds(
324:     questions: list[MCQuestion],
325:     likelihood_model: LikelihoodModel,
326:     thresholds: list[float],
327:     beta: float = 5.0,
328:     alpha: float = 10.0,
329:     precomputed: list[_PrecomputedQuestion] | None = None,
330: ) -> dict[float, list[EpisodeResult]]:
331:     """Sweep multiple thresholds with a single belief-computation pass.
332: 
333:     If *precomputed* is provided the expensive model calls are skipped
334:     entirely and the sweep is pure numpy.  Otherwise beliefs are computed
335:     once internally and reused across thresholds.
336:     """
337:     if precomputed is None:
338:         precomputed = precompute_beliefs(questions, likelihood_model, beta)
339: 
340:     out: dict[float, list[EpisodeResult]] = {}
341:     for threshold in thresholds:
342:         out[float(threshold)] = [
343:             _episode_from_precomputed(pq, threshold, alpha)
344:             for pq in precomputed
345:         ]
346:     return out
347: 
348: 
349: def result_to_dict(result: EpisodeResult) -> dict[str, Any]:
350:     return {
351:         "qid": result.qid,
352:         "buzz_step": result.buzz_step,
353:         "buzz_index": result.buzz_index,
354:         "gold_index": result.gold_index,
355:         "correct": result.correct,
356:         "reward_like": result.reward_like,
357:         "c_trace": result.c_trace,
358:         "g_trace": result.g_trace,
359:         "top_p_trace": result.top_p_trace,
360:         "entropy_trace": result.entropy_trace,
361:     }
````

## File: evaluation/controls.py
````python
  1: """
  2: Control Experiments for Quiz Bowl Buzzer Evaluation
  3: 
  4: Implements three control experiments to validate that the buzzer agent
  5: genuinely uses question clues rather than exploiting surface-form artifacts:
  6: 
  7: 1. **Choices-only control**: Strips all clues, trains a logistic regression
  8:    on option surface features (char n-grams, length, capitalization). Expected
  9:    accuracy ~25% (1/K) if options have no exploitable artifacts.
 10: 
 11: 2. **Shuffle control**: Randomizes option ordering to verify the agent has
 12:    no position bias. Performance should be unchanged.
 13: 
 14: 3. **Alias substitution control**: Swaps answer text with aliases to verify
 15:    robustness to surface-form changes.
 16: 
 17: Ported from qb-rl reference implementation (evaluation/controls.py) with
 18: import path adaptations for the unified qanta-buzzer codebase.
 19: """
 20: 
 21: from __future__ import annotations
 22: 
 23: import random
 24: from dataclasses import replace
 25: from typing import TYPE_CHECKING, Any, Callable
 26: 
 27: import numpy as np
 28: from sklearn.feature_extraction.text import TfidfVectorizer
 29: from sklearn.linear_model import LogisticRegression
 30: 
 31: from qb_data.mc_builder import MCQuestion
 32: 
 33: if TYPE_CHECKING:
 34:     from agents.threshold_buzzer import _PrecomputedQuestion
 35: 
 36: 
 37: def _option_scalar_features(option: str) -> list[float]:
 38:     """Extract scalar surface features from a single option string.
 39: 
 40:     Parameters
 41:     ----------
 42:     option : str
 43:         Answer option text.
 44: 
 45:     Returns
 46:     -------
 47:     list[float]
 48:         Six scalar features: char length, token count, has_parens,
 49:         has_comma, is_title, is_lower.
 50:     """
 51:     tokens = option.split()
 52:     has_parens = 1.0 if "(" in option or ")" in option else 0.0
 53:     has_comma = 1.0 if "," in option else 0.0
 54:     is_title = 1.0 if option.istitle() else 0.0
 55:     is_lower = 1.0 if option.islower() else 0.0
 56:     return [
 57:         float(len(option)),
 58:         float(len(tokens)),
 59:         has_parens,
 60:         has_comma,
 61:         is_title,
 62:         is_lower,
 63:     ]
 64: 
 65: 
 66: def _cross_option_features(options: list[str]) -> list[float]:
 67:     """Extract cross-option comparative features.
 68: 
 69:     Parameters
 70:     ----------
 71:     options : list[str]
 72:         All answer options for a question.
 73: 
 74:     Returns
 75:     -------
 76:     list[float]
 77:         Three features: max/min length ratio, length std, number of
 78:         distinct capitalization patterns.
 79:     """
 80:     lengths = np.array(
 81:         [max(1, len(o.split())) for o in options], dtype=np.float32
 82:     )
 83:     cap_patterns = len(
 84:         set(
 85:             ("title" if o.istitle() else "lower" if o.islower() else "mixed")
 86:             for o in options
 87:         )
 88:     )
 89:     return [
 90:         float(lengths.max() / lengths.min()),
 91:         float(lengths.std()),
 92:         float(cap_patterns),
 93:     ]
 94: 
 95: 
 96: def run_choices_only_control(
 97:     questions: list[MCQuestion],
 98:     random_seed: int = 13,
 99:     test_fraction: float = 0.25,
100: ) -> dict[str, float]:
101:     """Run choices-only control: predict answer from surface features only.
102: 
103:     Strips all question clues and trains a logistic regression on option
104:     surface features (char n-grams, length, capitalization patterns).
105:     Expected accuracy ~25% (1/K) if options are well-constructed.
106: 
107:     Parameters
108:     ----------
109:     questions : list[MCQuestion]
110:         Full MC question dataset.
111:     random_seed : int
112:         Seed for reproducible train/test split.
113:     test_fraction : float
114:         Fraction of questions held out for testing.
115: 
116:     Returns
117:     -------
118:     dict[str, float]
119:         Control results: accuracy, chance baseline, and test set size.
120:     """
121:     if not questions:
122:         return {"accuracy": 0.0, "chance": 0.0, "n_test": 0.0}
123: 
124:     rng = random.Random(random_seed)
125:     shuffled = questions[:]
126:     rng.shuffle(shuffled)
127:     split_idx = max(1, int(len(shuffled) * (1.0 - test_fraction)))
128:     train_q = shuffled[:split_idx]
129:     test_q = shuffled[split_idx:]
130:     if not test_q:
131:         test_q = train_q
132: 
133:     vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
134:     vec.fit([opt for q in train_q for opt in q.options])
135: 
136:     def build_matrix(
137:         rows: list[MCQuestion],
138:     ) -> tuple[np.ndarray, np.ndarray, list[int]]:
139:         X = []
140:         y = []
141:         group_sizes: list[int] = []
142:         for q in rows:
143:             cross = _cross_option_features(q.options)
144:             group_sizes.append(len(q.options))
145:             tfidf = vec.transform(q.options).toarray()
146:             for i, option in enumerate(q.options):
147:                 feat = np.array(
148:                     _option_scalar_features(option) + cross, dtype=np.float32
149:                 )
150:                 row = np.concatenate([feat, tfidf[i]], axis=0)
151:                 X.append(row)
152:                 y.append(1 if i == q.gold_index else 0)
153:         return np.array(X), np.array(y), group_sizes
154: 
155:     X_train, y_train, _ = build_matrix(train_q)
156:     X_test, y_test, test_group_sizes = build_matrix(test_q)
157:     clf = LogisticRegression(max_iter=1000)
158:     clf.fit(X_train, y_train)
159:     probs = clf.predict_proba(X_test)[:, 1]
160: 
161:     offset = 0
162:     correct = 0
163:     total = 0
164:     for q, group_size in zip(test_q, test_group_sizes):
165:         group_probs = probs[offset : offset + group_size]
166:         pred_idx = int(np.argmax(group_probs))
167:         if pred_idx == q.gold_index:
168:             correct += 1
169:         total += 1
170:         offset += group_size
171: 
172:     accuracy = correct / max(1, total)
173:     chance = 1.0 / max(1, len(questions[0].options))
174:     return {
175:         "accuracy": float(accuracy),
176:         "chance": float(chance),
177:         "n_test": float(total),
178:     }
179: 
180: 
181: def shuffled_option_copy(
182:     question: MCQuestion, rng: random.Random
183: ) -> MCQuestion:
184:     """Create a copy of an MCQuestion with shuffled option ordering.
185: 
186:     Parameters
187:     ----------
188:     question : MCQuestion
189:         Original question.
190:     rng : random.Random
191:         Random number generator for shuffling.
192: 
193:     Returns
194:     -------
195:     MCQuestion
196:         Copy with permuted options, profiles, answer_primary, and
197:         updated gold_index.
198:     """
199:     perm = list(range(len(question.options)))
200:     rng.shuffle(perm)
201:     new_options = [question.options[i] for i in perm]
202:     new_profiles = [question.option_profiles[i] for i in perm]
203:     new_answer_primary = [question.option_answer_primary[i] for i in perm]
204:     new_gold = perm.index(question.gold_index)
205:     return replace(
206:         question,
207:         options=new_options,
208:         option_profiles=new_profiles,
209:         option_answer_primary=new_answer_primary,
210:         gold_index=new_gold,
211:     )
212: 
213: 
214: def run_shuffle_control(
215:     questions: list[MCQuestion],
216:     evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
217:     random_seed: int = 13,
218: ) -> dict[str, Any]:
219:     """Run shuffle control: randomize option ordering and evaluate.
220: 
221:     Permutes the answer options for each question and runs the evaluator.
222:     If the agent has no position bias, performance should be unchanged.
223: 
224:     Parameters
225:     ----------
226:     questions : list[MCQuestion]
227:         Full MC question dataset.
228:     evaluator : callable
229:         Function that takes a list of MCQuestion and returns a metrics dict.
230:     random_seed : int
231:         Seed for reproducible shuffling.
232: 
233:     Returns
234:     -------
235:     dict[str, Any]
236:         Evaluation metrics on shuffled questions.
237:     """
238:     rng = random.Random(random_seed)
239:     shuffled = [shuffled_option_copy(q, rng) for q in questions]
240:     return evaluator(shuffled)
241: 
242: 
243: def alias_substitution_copy(
244:     question: MCQuestion,
245:     alias_lookup: dict[str, list[str]],
246:     rng: random.Random,
247: ) -> MCQuestion:
248:     """Create a copy of an MCQuestion with alias-substituted options.
249: 
250:     Parameters
251:     ----------
252:     question : MCQuestion
253:         Original question.
254:     alias_lookup : dict[str, list[str]]
255:         Mapping from canonical answer to list of known aliases.
256:     rng : random.Random
257:         Random number generator for alias selection.
258: 
259:     Returns
260:     -------
261:     MCQuestion
262:         Copy with alias-substituted option text and profiles.
263:     """
264:     new_options = []
265:     new_profiles = list(question.option_profiles)
266:     for i, (option_text, answer_primary) in enumerate(
267:         zip(question.options, question.option_answer_primary)
268:     ):
269:         aliases = [
270:             a
271:             for a in alias_lookup.get(answer_primary, [])
272:             if a and a != option_text
273:         ]
274:         if aliases:
275:             alias = rng.choice(aliases)
276:             new_options.append(alias)
277:             if new_profiles[i].strip() == answer_primary.strip():
278:                 new_profiles[i] = alias
279:         else:
280:             new_options.append(option_text)
281:     return replace(question, options=new_options, option_profiles=new_profiles)
282: 
283: 
284: def run_alias_substitution_control(
285:     questions: list[MCQuestion],
286:     alias_lookup: dict[str, list[str]],
287:     evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
288:     random_seed: int = 13,
289: ) -> dict[str, Any]:
290:     """Run alias substitution control: swap answer text with aliases.
291: 
292:     Replaces option text with known aliases to verify the agent is robust
293:     to surface-form changes. Performance should be similar to full eval.
294: 
295:     Parameters
296:     ----------
297:     questions : list[MCQuestion]
298:         Full MC question dataset.
299:     alias_lookup : dict[str, list[str]]
300:         Mapping from canonical answer to list of known aliases.
301:     evaluator : callable
302:         Function that takes a list of MCQuestion and returns a metrics dict.
303:     random_seed : int
304:         Seed for reproducible alias selection.
305: 
306:     Returns
307:     -------
308:     dict[str, Any]
309:         Evaluation metrics on alias-substituted questions.
310:     """
311:     rng = random.Random(random_seed)
312:     swapped = [
313:         alias_substitution_copy(q, alias_lookup=alias_lookup, rng=rng)
314:         for q in questions
315:     ]
316:     return evaluator(swapped)
317: 
318: 
319: def run_shuffle_control_precomputed(
320:     precomputed: list["_PrecomputedQuestion"],
321:     threshold: float,
322:     alpha: float,
323:     random_seed: int = 13,
324: ) -> dict[str, Any]:
325:     """Run shuffle control by permuting precomputed belief vectors.
326: 
327:     Produces numerically identical results to ``run_shuffle_control`` with
328:     a live ``SoftmaxProfileBuzzer`` evaluator, but makes zero
329:     ``likelihood_model.score()`` calls.  Instead, the belief vectors
330:     stored in each ``_PrecomputedQuestion`` are reordered according to
331:     the same random permutation that ``shuffled_option_copy`` would apply.
332: 
333:     Parameters
334:     ----------
335:     precomputed : list[_PrecomputedQuestion]
336:         Pre-computed belief distributions (one per question).
337:     threshold : float
338:         Buzz threshold for the softmax profile buzzer.
339:     alpha : float
340:         Sigmoid steepness for the confidence proxy.
341:     random_seed : int
342:         Seed for reproducible shuffling (must match the seed used in
343:         ``run_shuffle_control`` for equivalence).
344: 
345:     Returns
346:     -------
347:     dict[str, Any]
348:         Summary metrics with ``"runs"`` key containing per-question dicts.
349:     """
350:     from dataclasses import asdict
351: 
352:     from agents.threshold_buzzer import (
353:         _PrecomputedQuestion,
354:         _softmax_episode_from_precomputed,
355:     )
356:     from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
357: 
358:     rng = random.Random(random_seed)
359:     runs: list[dict[str, Any]] = []
360:     for pq in precomputed:
361:         perm = list(range(pq.num_options))
362:         rng.shuffle(perm)
363:         new_gold = perm.index(pq.gold_index)
364:         shuffled_beliefs = [b[perm] for b in pq.beliefs]
365:         shuffled_pq = _PrecomputedQuestion(
366:             qid=pq.qid,
367:             gold_index=new_gold,
368:             num_options=pq.num_options,
369:             beliefs=shuffled_beliefs,
370:         )
371:         result = _softmax_episode_from_precomputed(shuffled_pq, threshold, alpha)
372:         runs.append(asdict(result))
373:     summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
374:     summary["runs"] = runs
375:     return summary
376: 
377: 
378: def bootstrap_ci(
379:     values: list[float],
380:     n_samples: int = 1000,
381:     alpha: float = 0.05,
382:     seed: int = 13,
383: ) -> tuple[float, float]:
384:     """Compute bootstrap confidence interval for the mean.
385: 
386:     Parameters
387:     ----------
388:     values : list[float]
389:         Observed values.
390:     n_samples : int
391:         Number of bootstrap resamples.
392:     alpha : float
393:         Significance level (0.05 = 95% CI).
394:     seed : int
395:         Random seed for reproducibility.
396: 
397:     Returns
398:     -------
399:     tuple[float, float]
400:         Lower and upper bounds of the confidence interval.
401:     """
402:     if not values:
403:         return 0.0, 0.0
404:     rng = np.random.default_rng(seed)
405:     arr = np.array(values, dtype=np.float64)
406:     samples = []
407:     for _ in range(n_samples):
408:         idx = rng.integers(0, len(arr), size=len(arr))
409:         samples.append(float(arr[idx].mean()))
410:     lo = np.quantile(samples, alpha / 2.0)
411:     hi = np.quantile(samples, 1.0 - alpha / 2.0)
412:     return float(lo), float(hi)
````

## File: models/dspy_likelihood.py
````python
  1: """DSPy-based likelihood model with score caching.
  2: 
  3: Wraps a DSPy listwise scorer behind the ``LikelihoodModel.score()``
  4: interface.  Unlike embedding-based models, the DSPy scorer calls an LM
  5: to rank options — so caching is at the *score* level (keyed by clue +
  6: options + program fingerprint), not at the embedding level.
  7: 
  8: This module is importable without the ``dspy`` extra installed.
  9: The ``dspy`` package is only required at runtime when a DSPy-backed
 10: scorer is actually invoked (e.g. via ``scripts/optimize_dspy.py``).
 11: """
 12: 
 13: from __future__ import annotations
 14: 
 15: import hashlib
 16: import json
 17: from pathlib import Path
 18: from typing import Any
 19: 
 20: import numpy as np
 21: 
 22: from models.likelihoods import LikelihoodModel
 23: 
 24: 
 25: def _score_cache_key(
 26:     clue_prefix: str,
 27:     option_profiles: list[str],
 28:     program_fingerprint: str,
 29: ) -> str:
 30:     """Build a deterministic cache key for a score() call."""
 31:     payload = json.dumps(
 32:         {"clue": clue_prefix, "options": option_profiles, "fp": program_fingerprint},
 33:         sort_keys=True,
 34:     )
 35:     return hashlib.sha256(payload.encode("utf-8")).hexdigest()
 36: 
 37: 
 38: class DSPyLikelihood(LikelihoodModel):
 39:     """LikelihoodModel subclass backed by a DSPy program.
 40: 
 41:     Inherits from ``LikelihoodModel`` so it satisfies the factory
 42:     return type and isinstance checks.  Overrides ``score()`` with
 43:     LM-based scoring and a score-level cache.  ``_embed_batch()`` raises
 44:     ``NotImplementedError`` because DSPy scoring is not embedding-based.
 45: 
 46:     Unlike TF-IDF/SBERT/T5, this model does NOT produce embeddings.
 47:     ``_embed_batch`` is explicitly unsupported — calling it raises
 48:     ``NotImplementedError``.  Instead, scores are cached directly,
 49:     keyed by ``(clue, options, program_fingerprint)``.
 50: 
 51:     Parameters
 52:     ----------
 53:     scorer : callable
 54:         A DSPy module or function that accepts ``(clue_prefix, options)``
 55:         and returns a list/array of K scores.
 56:     program_fingerprint : str
 57:         Opaque identifier for the current compiled program state.
 58:         Cache entries are invalidated when this changes.
 59:     cache_dir : str or Path or None
 60:         Directory for persistent score cache.  When None, caching is
 61:         in-memory only.
 62:     """
 63: 
 64:     def __init__(
 65:         self,
 66:         scorer: Any,
 67:         program_fingerprint: str = "default",
 68:         cache_dir: str | Path | None = None,
 69:     ) -> None:
 70:         super().__init__()
 71:         self.scorer = scorer
 72:         self.program_fingerprint = program_fingerprint
 73:         self._score_cache: dict[str, np.ndarray] = {}
 74:         self._cache_dir = Path(cache_dir) if cache_dir else None
 75:         if self._cache_dir:
 76:             self._load_persistent_cache()
 77: 
 78:     def _load_persistent_cache(self) -> None:
 79:         if self._cache_dir is None:
 80:             return
 81:         cache_file = self._cache_dir / f"dspy_scores_{self.program_fingerprint}.npz"
 82:         if cache_file.exists():
 83:             with np.load(cache_file, allow_pickle=False) as data:
 84:                 for key in data.files:
 85:                     self._score_cache[key] = data[key].astype(np.float32)
 86: 
 87:     def _save_persistent_cache(self) -> None:
 88:         if self._cache_dir is None or not self._score_cache:
 89:             return
 90:         self._cache_dir.mkdir(parents=True, exist_ok=True)
 91:         cache_file = self._cache_dir / f"dspy_scores_{self.program_fingerprint}.npz"
 92:         np.savez_compressed(cache_file, **self._score_cache)
 93: 
 94:     def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
 95:         """Score answer options using the DSPy scorer.
 96: 
 97:         Results are cached by ``(clue, options, program_fingerprint)``.
 98:         Validates that the returned array has shape ``(K,)`` where
 99:         ``K = len(option_profiles)``.
100:         """
101:         key = _score_cache_key(clue_prefix, option_profiles, self.program_fingerprint)
102:         if key in self._score_cache:
103:             return self._score_cache[key].copy()
104: 
105:         raw = self.scorer(clue_prefix, option_profiles)
106:         scores = np.array(raw, dtype=np.float32)
107:         expected_k = len(option_profiles)
108:         if scores.ndim != 1 or len(scores) != expected_k:
109:             raise ValueError(
110:                 f"DSPy scorer returned shape {scores.shape}, "
111:                 f"expected ({expected_k},)"
112:             )
113:         self._score_cache[key] = scores
114:         return scores.copy()
115: 
116:     def save_cache(self, path: str | Path | None = None) -> int:
117:         """Persist score cache to disk."""
118:         if path:
119:             p = Path(path)
120:             p.parent.mkdir(parents=True, exist_ok=True)
121:             np.savez_compressed(p, **self._score_cache)
122:         else:
123:             self._save_persistent_cache()
124:         return len(self._score_cache)
125: 
126:     def load_cache(self, path: str | Path) -> int:
127:         """Load score cache from disk, merging without overwriting."""
128:         p = Path(path)
129:         if not p.exists():
130:             return 0
131:         loaded = 0
132:         with np.load(p, allow_pickle=False) as data:
133:             for key in data.files:
134:                 if key not in self._score_cache:
135:                     self._score_cache[key] = data[key].astype(np.float32)
136:                     loaded += 1
137:         return loaded
138: 
139:     @property
140:     def cache_memory_bytes(self) -> int:
141:         return sum(v.nbytes for v in self._score_cache.values())
142: 
143:     def _embed_batch(self, texts: list[str]) -> np.ndarray:
144:         """Not supported — DSPy scoring is not embedding-based."""
145:         raise NotImplementedError(
146:             "DSPyLikelihood does not produce embeddings. "
147:             "Use score() directly."
148:         )
149: 
150:     def embed_and_cache(self, texts: list[str]) -> np.ndarray:
151:         """Not supported — DSPy scoring is not embedding-based."""
152:         raise NotImplementedError(
153:             "DSPyLikelihood does not produce embeddings. "
154:             "Use score() directly."
155:         )
````

## File: models/t5_policy.py
````python
  1: """
  2: T5-based Policy Model for Quiz Bowl RL Agent
  3: 
  4: Implements T5PolicyModel with a custom PolicyHead containing three independent
  5: heads (wait/answer/value) for end-to-end text-based policy learning. This
  6: provides an alternative to the MLP policy trained on belief features
  7: (Phase 4 approach).
  8: 
  9: Architecture overview:
 10: 
 11:     Text input  -->  T5 Encoder  -->  Mean Pooling  -->  PolicyHead
 12:                                                           |-- Wait head (2)
 13:                                                           |-- Answer head (K)
 14:                                                           |-- Value head (1)
 15: 
 16: The T5 encoder produces contextual embeddings from tokenized text. Mean pooling
 17: (attention-masked) reduces the variable-length sequence to a fixed-size vector.
 18: The PolicyHead then produces three independent outputs:
 19: 
 20: - **Wait logits** [B, 2]: probability of waiting vs answering now
 21: - **Answer logits** [B, K]: probability of selecting each answer option
 22: - **Value estimate** [B, 1]: state value for PPO advantage computation
 23: 
 24: Action space maps to the TossupMCEnv convention:
 25:     0 = WAIT (wait head selects "wait")
 26:     1..K = SELECT answer i-1 (wait head selects "answer now", answer head picks i-1)
 27: 
 28: Ported from qanta-buzzer reference implementation (model.py) with these changes:
 29:     - T5EncoderModel replaces T5ForConditionalGeneration (2x faster, 50% less memory)
 30:     - T5TokenizerFast replaces T5Tokenizer (3-5x faster tokenization via Rust backend)
 31:     - Config dict replaces qanta-buzzer's Config class for unified codebase compatibility
 32:     - NumPy-style docstrings added throughout
 33: """
 34: 
 35: from __future__ import annotations
 36: 
 37: import os
 38: from typing import Any, Dict, List, Optional, Tuple
 39: 
 40: import torch
 41: import torch.nn as nn
 42: import torch.nn.functional as F
 43: 
 44: 
 45: class PolicyHead(nn.Module):
 46:     """Custom policy head with three independent output heads.
 47: 
 48:     Attached to a T5 encoder's pooled output, this module produces the three
 49:     outputs needed for actor-critic RL in the quiz bowl POMDP: a binary
 50:     wait/answer-now decision, a K-way answer selection, and a scalar value
 51:     estimate.
 52: 
 53:     All three heads are fully independent (no shared hidden layers beyond the
 54:     encoder), using the same pattern: Linear -> ReLU -> Dropout -> Linear.
 55: 
 56:     Parameters
 57:     ----------
 58:     hidden_size : int
 59:         Dimensionality of the input from the T5 encoder's pooled output.
 60:         Default 1024 matches T5-large (``d_model``). Use 512 for t5-small,
 61:         768 for t5-base.
 62:     num_choices : int
 63:         Number of answer options (K). Default 4 for quiz bowl MC questions.
 64: 
 65:     Attributes
 66:     ----------
 67:     wait_head : nn.Sequential
 68:         Binary head producing [wait, answer_now] logits.
 69:     answer_head : nn.Sequential
 70:         Multi-class head producing logits over K answer choices.
 71:     value_head : nn.Sequential
 72:         Scalar head producing state value estimate.
 73:     """
 74: 
 75:     def __init__(self, hidden_size: int = 1024, num_choices: int = 4) -> None:
 76:         super().__init__()
 77: 
 78:         self.hidden_size = hidden_size
 79:         self.num_choices = num_choices
 80: 
 81:         # Wait/continue decision head (binary: wait vs answer_now)
 82:         self.wait_head = nn.Sequential(
 83:             nn.Linear(hidden_size, 256),
 84:             nn.ReLU(),
 85:             nn.Dropout(0.1),
 86:             nn.Linear(256, 2),  # [wait, answer_now]
 87:         )
 88: 
 89:         # Answer selection head (over K choices)
 90:         self.answer_head = nn.Sequential(
 91:             nn.Linear(hidden_size, 512),
 92:             nn.ReLU(),
 93:             nn.Dropout(0.1),
 94:             nn.Linear(512, num_choices),
 95:         )
 96: 
 97:         # Value head (state value estimate for PPO)
 98:         self.value_head = nn.Sequential(
 99:             nn.Linear(hidden_size, 256),
100:             nn.ReLU(),
101:             nn.Dropout(0.1),
102:             nn.Linear(256, 1),
103:         )
104: 
105:     def forward(
106:         self, encoder_hidden_state: torch.Tensor
107:     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
108:         """Forward pass through all three heads.
109: 
110:         Parameters
111:         ----------
112:         encoder_hidden_state : torch.Tensor
113:             Pooled encoder output of shape ``[batch_size, hidden_size]``.
114: 
115:         Returns
116:         -------
117:         wait_logits : torch.Tensor
118:             Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
119:         answer_logits : torch.Tensor
120:             Shape ``[batch_size, num_choices]`` -- logits over answer options.
121:         values : torch.Tensor
122:             Shape ``[batch_size, 1]`` -- state value estimates.
123:         """
124:         wait_logits = self.wait_head(encoder_hidden_state)
125:         answer_logits = self.answer_head(encoder_hidden_state)
126:         values = self.value_head(encoder_hidden_state)
127: 
128:         return wait_logits, answer_logits, values
129: 
130: 
131: class T5PolicyModel(nn.Module):
132:     """T5 encoder with custom policy head for end-to-end RL.
133: 
134:     Combines a pre-trained T5 encoder with a ``PolicyHead`` to produce policy
135:     outputs directly from text observations. This is the alternative approach
136:     to Phase 4's MLP policy, which operates on numeric belief features.
137: 
138:     The model processes text in three stages:
139: 
140:     1. **Tokenization**: Text is tokenized with ``T5TokenizerFast`` (Rust-backed
141:        for speed) with padding and truncation.
142:     2. **Encoding**: ``T5EncoderModel`` produces contextual hidden states
143:        ``[B, seq_len, d_model]``.
144:     3. **Pooling + Heads**: Attention-masked mean pooling reduces to
145:        ``[B, d_model]``, then PolicyHead produces wait/answer/value outputs.
146: 
147:     Action space follows TossupMCEnv convention:
148:         - 0 = WAIT
149:         - 1..K = SELECT answer (i-1)
150: 
151:     Combined actions are treated as a factorized policy:
152:         - ``P(WAIT) = p_wait``
153:         - ``P(BUZZ_i) = p_buzz * p_ans(i)``
154: 
155:     The answer distribution only contributes when the policy chooses to buzz.
156: 
157:     Parameters
158:     ----------
159:     config : dict[str, Any]
160:         Configuration dictionary with the following keys:
161: 
162:         - ``model_name`` (str): HuggingFace T5 model identifier.
163:           Default ``"t5-large"``. Options: ``"t5-small"``, ``"t5-base"``,
164:           ``"t5-large"``.
165:         - ``device`` (str): Torch device. Default auto-detects
166:           (cuda > mps > cpu).
167:         - ``max_input_length`` (int): Maximum token sequence length.
168:           Default 512.
169:         - ``num_choices`` (int): Number of answer options (K). Default 4.
170: 
171:     Attributes
172:     ----------
173:     config : dict[str, Any]
174:         Configuration dictionary.
175:     device : torch.device
176:         Computation device.
177:     encoder : T5EncoderModel
178:         Pre-trained T5 encoder.
179:     tokenizer : T5TokenizerFast
180:         Fast T5 tokenizer.
181:     policy_head : PolicyHead
182:         Custom three-head policy module.
183:     max_input_length : int
184:         Maximum token sequence length for tokenization.
185: 
186:     Examples
187:     --------
188:     >>> config = {"model_name": "t5-small", "device": "cpu", "num_choices": 4}
189:     >>> model = T5PolicyModel(config)
190:     >>> texts = ["CLUES: first president | CHOICES: (1) Washington (2) Jefferson"]
191:     >>> wait_logits, answer_logits, values = model(texts)
192:     >>> wait_logits.shape
193:     torch.Size([1, 2])
194:     """
195: 
196:     def __init__(self, config: Dict[str, Any]) -> None:
197:         super().__init__()
198:         from transformers import T5EncoderModel, T5TokenizerFast
199: 
200:         self.config = config
201:         model_name = config.get("model_name", "t5-large")
202:         self.max_input_length = config.get("max_input_length", 512)
203:         num_choices = config.get("num_choices", 4)
204: 
205:         # Auto-detect device
206:         default_device = "cpu"
207:         if torch.cuda.is_available():
208:             default_device = "cuda"
209:         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
210:             default_device = "mps"
211:         self.device = torch.device(config.get("device", default_device))
212: 
213:         # Load T5 encoder only (not full T5ForConditionalGeneration)
214:         # This is 2x faster and uses 50% less memory since the decoder is unused
215:         print(f"Loading T5 encoder: {model_name}")
216:         self.encoder = T5EncoderModel.from_pretrained(model_name)
217:         self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
218: 
219:         # Get hidden size from T5 config (512 for small, 768 for base, 1024 for large)
220:         hidden_size = self.encoder.config.d_model
221: 
222:         # Custom policy head
223:         self.policy_head = PolicyHead(
224:             hidden_size=hidden_size,
225:             num_choices=num_choices,
226:         )
227: 
228:         # Move to device
229:         self.to(self.device)
230: 
231:         # Print model info
232:         self._print_model_info()
233: 
234:     def _print_model_info(self) -> None:
235:         """Print model architecture summary and parameter counts."""
236:         encoder_params = sum(p.numel() for p in self.encoder.parameters())
237:         policy_params = sum(p.numel() for p in self.policy_head.parameters())
238:         total_params = encoder_params + policy_params
239: 
240:         print("Model Architecture:")
241:         print(f"  T5 encoder parameters: {encoder_params:,}")
242:         print(f"  Policy head parameters: {policy_params:,}")
243:         print(f"  Total parameters: {total_params:,}")
244:         print(f"  Device: {self.device}")
245: 
246:     def encode_input(
247:         self,
248:         text_inputs: List[str],
249:         max_length: Optional[int] = None,
250:     ) -> Dict[str, torch.Tensor]:
251:         """Tokenize text inputs using T5TokenizerFast.
252: 
253:         Parameters
254:         ----------
255:         text_inputs : list[str]
256:             List of input text strings to tokenize.
257:         max_length : int or None
258:             Maximum sequence length. If None, uses ``self.max_input_length``.
259: 
260:         Returns
261:         -------
262:         dict[str, torch.Tensor]
263:             Dictionary with ``"input_ids"`` and ``"attention_mask"`` tensors,
264:             both of shape ``[batch_size, seq_len]``, moved to ``self.device``.
265:         """
266:         if max_length is None:
267:             max_length = self.max_input_length
268: 
269:         encoding = self.tokenizer(
270:             text_inputs,
271:             padding=True,
272:             truncation=True,
273:             max_length=max_length,
274:             return_tensors="pt",
275:         )
276: 
277:         return {k: v.to(self.device) for k, v in encoding.items()}
278: 
279:     def get_encoder_output(
280:         self,
281:         input_ids: torch.Tensor,
282:         attention_mask: torch.Tensor,
283:     ) -> torch.Tensor:
284:         """Compute T5 encoder output and pool to a fixed-size vector.
285: 
286:         Uses attention-masked mean pooling: sum hidden states where attention
287:         mask is 1, divide by number of non-padding tokens. This ensures
288:         padding tokens contribute zero to the pooled representation.
289: 
290:         Parameters
291:         ----------
292:         input_ids : torch.Tensor
293:             Token IDs of shape ``[batch_size, seq_len]``.
294:         attention_mask : torch.Tensor
295:             Attention mask of shape ``[batch_size, seq_len]`` (1 for real
296:             tokens, 0 for padding).
297: 
298:         Returns
299:         -------
300:         torch.Tensor
301:             Pooled encoder output of shape ``[batch_size, hidden_size]``.
302:         """
303:         # Get encoder outputs
304:         encoder_outputs = self.encoder(
305:             input_ids=input_ids,
306:             attention_mask=attention_mask,
307:             return_dict=True,
308:         )
309: 
310:         # encoder_outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
311:         hidden_states = encoder_outputs.last_hidden_state
312: 
313:         # Attention-masked mean pooling over sequence dimension
314:         mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
315:         sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
316:         sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
317:         pooled_output = sum_hidden / sum_mask
318: 
319:         return pooled_output
320: 
321:     def forward(
322:         self,
323:         text_inputs: List[str],
324:         return_value: bool = True,
325:     ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
326:         """Forward pass: tokenize, encode, pool, then apply policy head.
327: 
328:         Parameters
329:         ----------
330:         text_inputs : list[str]
331:             List of text observations (e.g.,
332:             ``"CLUES: clue1 clue2 | CHOICES: (1) ans1 (2) ans2"``).
333:         return_value : bool
334:             If True, return value estimates. If False, values is None.
335: 
336:         Returns
337:         -------
338:         wait_logits : torch.Tensor
339:             Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
340:         answer_logits : torch.Tensor
341:             Shape ``[batch_size, num_choices]`` -- logits over answer options.
342:         values : torch.Tensor or None
343:             Shape ``[batch_size, 1]`` if return_value is True, else None.
344:         """
345:         # Encode inputs
346:         encoding = self.encode_input(text_inputs)
347: 
348:         # Get pooled encoder output
349:         pooled_output = self.get_encoder_output(
350:             encoding["input_ids"],
351:             encoding["attention_mask"],
352:         )
353: 
354:         # Pass through policy head
355:         wait_logits, answer_logits, values = self.policy_head(pooled_output)
356: 
357:         if not return_value:
358:             values = None
359: 
360:         return wait_logits, answer_logits, values
361: 
362:     def predict_answer(
363:         self,
364:         input_ids: torch.Tensor,
365:         attention_mask: torch.Tensor,
366:     ) -> Tuple[torch.Tensor, torch.Tensor]:
367:         """Predict answer choice for supervised training.
368: 
369:         Only uses the answer head (wait and value heads are ignored). This is
370:         the interface for supervised warm-start training where the model learns
371:         to select the correct answer from complete questions.
372: 
373:         Parameters
374:         ----------
375:         input_ids : torch.Tensor
376:             Token IDs of shape ``[batch_size, seq_len]``.
377:         attention_mask : torch.Tensor
378:             Attention mask of shape ``[batch_size, seq_len]``.
379: 
380:         Returns
381:         -------
382:         answer_logits : torch.Tensor
383:             Shape ``[batch_size, num_choices]`` -- logits over answer choices.
384:         predictions : torch.Tensor
385:             Shape ``[batch_size]`` -- predicted answer indices (argmax).
386:         """
387:         # Get encoder output
388:         pooled_output = self.get_encoder_output(input_ids, attention_mask)
389: 
390:         # Get answer logits from policy head
391:         _, answer_logits, _ = self.policy_head(pooled_output)
392: 
393:         # Get predictions
394:         predictions = torch.argmax(answer_logits, dim=-1)
395: 
396:         return answer_logits, predictions
397: 
398:     def _joint_action_log_prob(
399:         self,
400:         wait_logits: torch.Tensor,
401:         answer_logits: torch.Tensor,
402:         actions: torch.Tensor,
403:     ) -> torch.Tensor:
404:         """Compute factorized log-probabilities for flat WAIT/BUZZ actions.
405: 
406:         Parameters
407:         ----------
408:         wait_logits : torch.Tensor
409:             Binary logits of shape ``[batch_size, 2]`` for [WAIT, BUZZ].
410:         answer_logits : torch.Tensor
411:             Answer logits of shape ``[batch_size, K]``.
412:         actions : torch.Tensor
413:             Flat actions of shape ``[batch_size]`` where 0 = WAIT and
414:             1..K = BUZZ with answer index action-1.
415: 
416:         Returns
417:         -------
418:         torch.Tensor
419:             Log-probabilities of shape ``[batch_size]``.
420:         """
421:         wait_log_probs = F.log_softmax(wait_logits, dim=-1)
422:         answer_log_probs = F.log_softmax(answer_logits, dim=-1)
423: 
424:         wait_actions = (actions > 0).long()
425:         answer_actions = torch.clamp(actions - 1, min=0)
426: 
427:         selected_wait = wait_log_probs.gather(1, wait_actions.unsqueeze(-1)).squeeze(-1)
428:         selected_answer = answer_log_probs.gather(
429:             1, answer_actions.unsqueeze(-1)
430:         ).squeeze(-1)
431: 
432:         return torch.where(actions == 0, selected_wait, selected_wait + selected_answer)
433: 
434:     def _joint_entropy(
435:         self,
436:         wait_logits: torch.Tensor,
437:         answer_logits: torch.Tensor,
438:     ) -> torch.Tensor:
439:         """Compute chain-rule entropy for the factorized wait/answer policy.
440: 
441:         Returns ``H(wait) + p_buzz * H(answer)`` for each example.
442:         """
443:         wait_probs = F.softmax(wait_logits, dim=-1)
444:         wait_log_probs = F.log_softmax(wait_logits, dim=-1)
445:         answer_probs = F.softmax(answer_logits, dim=-1)
446:         answer_log_probs = F.log_softmax(answer_logits, dim=-1)
447: 
448:         wait_entropy = -(wait_probs * wait_log_probs).sum(dim=-1)
449:         answer_entropy = -(answer_probs * answer_log_probs).sum(dim=-1)
450:         return wait_entropy + wait_probs[:, 1] * answer_entropy
451: 
452:     def select_action(
453:         self,
454:         input_ids: torch.Tensor,
455:         attention_mask: torch.Tensor,
456:         deterministic: bool = False,
457:         temperature: float = 1.0,
458:     ) -> Tuple[torch.Tensor, Dict[str, Any]]:
459:         """Select flat WAIT/BUZZ actions from the factorized policy.
460: 
461:         Produces combined actions following TossupMCEnv convention:
462:         0 = WAIT, 1..K = SELECT answer 0..K-1. Under the factorized policy:
463: 
464:         - ``P(WAIT) = p_wait``
465:         - ``P(BUZZ_i) = p_buzz * p_ans(i)``
466: 
467:         Answer sampling only occurs for examples that actually buzz.
468: 
469:         Parameters
470:         ----------
471:         input_ids : torch.Tensor
472:             Token IDs of shape ``[batch_size, seq_len]``.
473:         attention_mask : torch.Tensor
474:             Attention mask of shape ``[batch_size, seq_len]``.
475:         deterministic : bool
476:             If True, use argmax instead of sampling.
477:         temperature : float
478:             Temperature for softmax. Higher values increase randomness.
479:             Default 1.0 (no scaling).
480: 
481:         Returns
482:         -------
483:         combined_actions : torch.Tensor
484:             Shape ``[batch_size]`` -- combined actions (0 = WAIT, 1..K = SELECT).
485:         info : dict[str, Any]
486:             Dictionary with keys:
487: 
488:             - ``wait_logits``: raw wait head output
489:             - ``answer_logits``: raw answer head output
490:             - ``wait_probs``: softmax of wait logits
491:             - ``answer_probs``: softmax of answer logits
492:             - ``wait_actions``: sampled wait decisions (0 or 1)
493:             - ``answer_actions``: sampled answer indices (0..K-1)
494:             - ``values``: value estimates
495:             - ``log_probs``: total log probability of the combined action
496:         """
497:         with torch.no_grad():
498:             pooled_output = self.get_encoder_output(input_ids, attention_mask)
499:             wait_logits, answer_logits, values = self.policy_head(pooled_output)
500: 
501:             wait_logits_scaled = wait_logits / temperature
502:             answer_logits_scaled = answer_logits / temperature
503: 
504:             wait_probs = F.softmax(wait_logits_scaled, dim=-1)
505:             answer_probs = F.softmax(answer_logits_scaled, dim=-1)
506:             flat_action_probs = torch.cat(
507:                 [wait_probs[:, :1], wait_probs[:, 1:2] * answer_probs],
508:                 dim=-1,
509:             )
510: 
511:             if deterministic:
512:                 combined_actions = torch.argmax(flat_action_probs, dim=-1)
513:                 wait_actions = (combined_actions > 0).long()
514:                 answer_actions = torch.clamp(combined_actions - 1, min=0)
515:             else:
516:                 wait_actions = torch.distributions.Categorical(wait_probs).sample()
517:                 answer_actions = torch.argmax(answer_probs, dim=-1)
518:                 buzz_mask = wait_actions == 1
519:                 if buzz_mask.any():
520:                     buzz_answers = torch.distributions.Categorical(
521:                         answer_probs[buzz_mask]
522:                     ).sample()
523:                     answer_actions = answer_actions.clone()
524:                     answer_actions[buzz_mask] = buzz_answers
525:                 combined_actions = torch.where(
526:                     wait_actions == 0,
527:                     torch.zeros_like(wait_actions),
528:                     1 + answer_actions,
529:                 )
530: 
531:             log_probs = self._joint_action_log_prob(
532:                 wait_logits_scaled, answer_logits_scaled, combined_actions
533:             )
534: 
535:             combined_actions = torch.where(
536:                 wait_actions == 0,
537:                 torch.zeros_like(wait_actions),
538:                 1 + answer_actions,
539:             )
540: 
541:             info = {
542:                 "wait_logits": wait_logits,
543:                 "answer_logits": answer_logits,
544:                 "wait_probs": wait_probs,
545:                 "answer_probs": answer_probs,
546:                 "wait_actions": wait_actions,
547:                 "answer_actions": answer_actions,
548:                 "values": values,
549:                 "log_probs": log_probs,
550:             }
551: 
552:             return combined_actions, info
553: 
554:     def get_action_log_probs(
555:         self,
556:         input_ids: torch.Tensor,
557:         attention_mask: torch.Tensor,
558:         actions: torch.Tensor,
559:     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
560:         """Compute log probabilities and entropy for given actions.
561: 
562:         Used during PPO training to evaluate old actions under the current
563:         policy. Combined actions follow the factorized semantics:
564: 
565:         - ``P(WAIT) = p_wait``
566:         - ``P(BUZZ_i) = p_buzz * p_ans(i)``
567: 
568:         Parameters
569:         ----------
570:         input_ids : torch.Tensor
571:             Token IDs of shape ``[batch_size, seq_len]``.
572:         attention_mask : torch.Tensor
573:             Attention mask of shape ``[batch_size, seq_len]``.
574:         actions : torch.Tensor
575:             Combined actions of shape ``[batch_size]``. Values in {0, 1, ..., K}.
576: 
577:         Returns
578:         -------
579:         log_probs : torch.Tensor
580:             Shape ``[batch_size]`` -- total log probability of each action.
581:         entropy : torch.Tensor
582:             Shape ``[batch_size]`` -- chain-rule entropy for the factorized policy.
583:         values : torch.Tensor
584:             Shape ``[batch_size]`` -- value estimates (squeezed).
585:         """
586:         pooled_output = self.get_encoder_output(input_ids, attention_mask)
587:         wait_logits, answer_logits, values = self.policy_head(pooled_output)
588: 
589:         log_probs = self._joint_action_log_prob(wait_logits, answer_logits, actions)
590:         entropy = self._joint_entropy(wait_logits, answer_logits)
591: 
592:         return log_probs, entropy, values.squeeze(-1)
593: 
594:     def save(self, save_dir: str) -> None:
595:         """Save model checkpoint to disk.
596: 
597:         Saves three components:
598:         1. T5 encoder weights and config (HuggingFace format)
599:         2. Tokenizer files (HuggingFace format)
600:         3. Policy head state dict (PyTorch format as ``policy_head.pt``)
601: 
602:         Parameters
603:         ----------
604:         save_dir : str
605:             Directory path to save the checkpoint. Created if it doesn't exist.
606:         """
607:         os.makedirs(save_dir, exist_ok=True)
608: 
609:         # Save T5 encoder
610:         self.encoder.save_pretrained(save_dir)
611:         self.tokenizer.save_pretrained(save_dir)
612: 
613:         # Save policy head
614:         policy_head_path = os.path.join(save_dir, "policy_head.pt")
615:         torch.save(self.policy_head.state_dict(), policy_head_path)
616: 
617:         print(f"Model saved to {save_dir}")
618: 
619:     def load(self, load_dir: str) -> None:
620:         """Load model checkpoint from disk.
621: 
622:         Loads T5 encoder weights, tokenizer, and policy head state dict from
623:         the specified directory. The model is moved to ``self.device`` after
624:         loading.
625: 
626:         Parameters
627:         ----------
628:         load_dir : str
629:             Directory containing a previously saved checkpoint.
630: 
631:         Raises
632:         ------
633:         FileNotFoundError
634:             If ``policy_head.pt`` is not found in ``load_dir``.
635:         """
636:         from transformers import T5EncoderModel, T5TokenizerFast
637: 
638:         # Load T5 encoder
639:         self.encoder = T5EncoderModel.from_pretrained(load_dir)
640:         self.tokenizer = T5TokenizerFast.from_pretrained(load_dir)
641: 
642:         # Load policy head
643:         policy_head_path = os.path.join(load_dir, "policy_head.pt")
644:         self.policy_head.load_state_dict(
645:             torch.load(policy_head_path, map_location=self.device, weights_only=True)
646:         )
647: 
648:         self.to(self.device)
649:         print(f"Model loaded from {load_dir}")
650: 
651:     @classmethod
652:     def load_pretrained(
653:         cls,
654:         load_dir: str,
655:         device: Optional[str] = None,
656:     ) -> "T5PolicyModel":
657:         """Load a pretrained model from a directory.
658: 
659:         Class method that creates a new T5PolicyModel instance and loads
660:         weights from a saved checkpoint.
661: 
662:         Parameters
663:         ----------
664:         load_dir : str
665:             Directory containing a previously saved checkpoint.
666:         device : str or None
667:             Device to load model on (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).
668:             If None, auto-detects.
669: 
670:         Returns
671:         -------
672:         T5PolicyModel
673:             A loaded model instance ready for inference.
674:         """
675:         from transformers import T5Config
676: 
677:         # Validate checkpoint integrity (lightweight — config JSON only)
678:         T5Config.from_pretrained(load_dir, local_files_only=True)
679: 
680:         # Infer num_choices from policy head state dict
681:         policy_head_path = os.path.join(load_dir, "policy_head.pt")
682:         policy_head_state = torch.load(
683:             policy_head_path, map_location="cpu", weights_only=True
684:         )
685:         # answer_head final linear layer weight shape is [num_choices, hidden_dim]
686:         num_choices = policy_head_state["answer_head.3.weight"].shape[0]
687: 
688:         config = {
689:             "model_name": load_dir,
690:             "num_choices": num_choices,
691:         }
692:         if device is not None:
693:             config["device"] = device
694: 
695:         model = cls(config)
696:         model.load(load_dir)
697:         return model
````

## File: qb_data/answer_profiles.py
````python
  1: """Answer profile builder with leave-one-out exclusion for quiz bowl questions."""
  2: 
  3: from __future__ import annotations
  4: 
  5: from collections import defaultdict
  6: from typing import Dict, List, Optional, Tuple
  7: 
  8: from qb_data.data_loader import TossupQuestion
  9: 
 10: 
 11: class AnswerProfileBuilder:
 12:     """Builds profiles for answers by aggregating question texts.
 13: 
 14:     The profile for an answer is created by concatenating all question texts
 15:     that have that answer. When building profiles for distractors, we use
 16:     all questions. For the gold answer, we exclude the current question to
 17:     prevent information leakage (leave-one-out).
 18: 
 19:     Attributes:
 20:         max_tokens_per_profile: Maximum number of tokens to keep in each profile.
 21:         min_questions_per_answer: Minimum questions needed to build a profile.
 22:         _grouped: Dictionary mapping answer_primary to list of (qid, question_text) tuples.
 23:     """
 24: 
 25:     def __init__(
 26:         self,
 27:         max_tokens_per_profile: int = 2000,
 28:         min_questions_per_answer: int = 1
 29:     ):
 30:         """Initialize the answer profile builder.
 31: 
 32:         Args:
 33:             max_tokens_per_profile: Maximum tokens to keep in each profile.
 34:             min_questions_per_answer: Minimum questions needed to build a profile.
 35:         """
 36:         self.max_tokens_per_profile = max_tokens_per_profile
 37:         self.min_questions_per_answer = min_questions_per_answer
 38:         self._grouped: Dict[str, List[Tuple[str, str]]] = {}
 39:         self._cache: Dict[Tuple[str, Optional[str]], str] = {}
 40: 
 41:     def fit(self, questions: List[TossupQuestion]) -> "AnswerProfileBuilder":
 42:         """Fit the builder on a set of questions.
 43: 
 44:         Groups questions by their primary answer for efficient profile building.
 45: 
 46:         Args:
 47:             questions: List of tossup questions to group by answer.
 48: 
 49:         Returns:
 50:             Self for method chaining.
 51:         """
 52:         grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
 53:         for q in questions:
 54:             # Store qid and full question text for each answer
 55:             grouped[q.answer_primary].append((q.qid, q.question))
 56:         self._grouped = dict(grouped)
 57:         self._cache = {}
 58:         return self
 59: 
 60:     def _profile_text(
 61:         self,
 62:         answer_primary: str,
 63:         exclude_qid: Optional[str] = None
 64:     ) -> str:
 65:         """Build profile text for an answer with optional exclusion.
 66: 
 67:         Args:
 68:             answer_primary: The answer to build a profile for.
 69:             exclude_qid: Optional question ID to exclude (leave-one-out).
 70: 
 71:         Returns:
 72:             Profile text truncated to max_tokens_per_profile.
 73:         """
 74:         key = (answer_primary, exclude_qid)
 75:         if key in self._cache:
 76:             return self._cache[key]
 77: 
 78:         items = self._grouped.get(answer_primary, [])
 79:         texts: List[str] = []
 80: 
 81:         # Collect all question texts except the excluded one
 82:         for qid, qtext in items:
 83:             if exclude_qid is not None and qid == exclude_qid:
 84:                 continue
 85:             texts.append(qtext)
 86: 
 87:         # If not enough questions after exclusion, fall back to answer text
 88:         if len(texts) < self.min_questions_per_answer:
 89:             self._cache[key] = answer_primary
 90:             return answer_primary
 91: 
 92:         # Merge all texts and split into tokens
 93:         merged = " ".join(texts).split()
 94: 
 95:         # Truncate to max tokens if specified
 96:         if self.max_tokens_per_profile > 0:
 97:             merged = merged[:self.max_tokens_per_profile]
 98: 
 99:         result = " ".join(merged) if merged else answer_primary
100:         self._cache[key] = result
101:         return result
102: 
103:     def profile_for_answer(
104:         self,
105:         answer_primary: str,
106:         exclude_qid: Optional[str] = None
107:     ) -> str:
108:         """Get the profile for a specific answer.
109: 
110:         Args:
111:             answer_primary: The answer to get a profile for.
112:             exclude_qid: Optional question ID to exclude (for gold answer).
113: 
114:         Returns:
115:             Profile text for the answer.
116:         """
117:         return self._profile_text(
118:             answer_primary=answer_primary,
119:             exclude_qid=exclude_qid
120:         )
121: 
122:     def build_profiles(
123:         self,
124:         questions: List[TossupQuestion],
125:         exclude_qid: Optional[str] = None,
126:     ) -> Dict[str, str]:
127:         """Build profiles for all answers in the dataset.
128: 
129:         Args:
130:             questions: List of questions (used to fit if not already fitted).
131:             exclude_qid: Optional question ID to exclude from all profiles.
132: 
133:         Returns:
134:             Dictionary mapping answer_primary to profile text.
135:         """
136:         if not self._grouped:
137:             self.fit(questions)
138: 
139:         return {
140:             answer: self._profile_text(answer, exclude_qid=exclude_qid)
141:             for answer in self._grouped.keys()
142:         }
````

## File: qb_data/dataset_splits.py
````python
  1: """
  2: Stratified dataset splitting utilities for quiz bowl data.
  3: 
  4: This module provides functions to create train/val/test splits that maintain
  5: category distribution across all splits.
  6: """
  7: 
  8: import hashlib
  9: import json
 10: import random
 11: from collections import defaultdict
 12: from pathlib import Path
 13: from typing import List, Tuple, Dict, Any
 14: 
 15: from qb_data.data_loader import TossupQuestion
 16: 
 17: 
 18: def create_stratified_splits(
 19:     questions: List[TossupQuestion],
 20:     ratios: List[float] = [0.7, 0.15, 0.15],
 21:     seed: int = 42
 22: ) -> Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]:
 23:     """
 24:     Create stratified train/val/test splits maintaining category distribution.
 25: 
 26:     Parameters
 27:     ----------
 28:     questions : List[TossupQuestion]
 29:         List of questions to split
 30:     ratios : List[float]
 31:         Train/val/test split ratios (must sum to 1.0)
 32:     seed : int
 33:         Random seed for reproducibility
 34: 
 35:     Returns
 36:     -------
 37:     Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]
 38:         Train, validation, and test splits
 39: 
 40:     Raises
 41:     ------
 42:     ValueError
 43:         If ratios don't sum to 1.0 or questions list is empty
 44:     """
 45:     # Validate inputs
 46:     if not questions:
 47:         raise ValueError("Cannot split empty question list")
 48: 
 49:     if abs(sum(ratios) - 1.0) > 1e-6:
 50:         raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
 51: 
 52:     # Initialize random generator for reproducibility
 53:     rng = random.Random(seed)
 54: 
 55:     # Group questions by category
 56:     category_groups = defaultdict(list)
 57:     for q in questions:
 58:         category_groups[q.category].append(q)
 59: 
 60:     # Initialize output lists
 61:     train_questions = []
 62:     val_questions = []
 63:     test_questions = []
 64: 
 65:     # Split each category maintaining ratios
 66:     for category, category_questions in category_groups.items():
 67:         # Sort for deterministic splits
 68:         sorted_questions = sorted(category_questions, key=lambda q: q.qid)
 69: 
 70:         # Deterministic per-category seed via MD5 (immune to PYTHONHASHSEED)
 71:         cat_hash = int(hashlib.md5(category.encode("utf-8")).hexdigest(), 16)
 72:         category_seed = seed + cat_hash % 1_000_000
 73:         category_rng = random.Random(category_seed)
 74:         shuffled = sorted_questions.copy()
 75:         category_rng.shuffle(shuffled)
 76: 
 77:         n = len(shuffled)
 78: 
 79:         # Calculate split indices
 80:         train_end = int(n * ratios[0])
 81:         val_end = train_end + int(n * ratios[1])
 82: 
 83:         # Handle small categories - ensure at least 1 in train if possible
 84:         if n == 1:
 85:             train_questions.extend(shuffled)
 86:         elif n == 2:
 87:             train_questions.extend(shuffled[:1])
 88:             val_questions.extend(shuffled[1:])
 89:         else:
 90:             # Standard split
 91:             train_questions.extend(shuffled[:train_end])
 92:             val_questions.extend(shuffled[train_end:val_end])
 93:             test_questions.extend(shuffled[val_end:])
 94: 
 95:     # Verify all questions assigned exactly once
 96:     total_original = len(questions)
 97:     total_split = len(train_questions) + len(val_questions) + len(test_questions)
 98: 
 99:     if total_original != total_split:
100:         raise RuntimeError(f"Split mismatch: {total_original} original vs {total_split} split")
101: 
102:     # Log category distribution statistics
103:     print(f"Dataset split complete:")
104:     print(f"  Train: {len(train_questions)} questions ({len(train_questions)/total_original:.1%})")
105:     print(f"  Val:   {len(val_questions)} questions ({len(val_questions)/total_original:.1%})")
106:     print(f"  Test:  {len(test_questions)} questions ({len(test_questions)/total_original:.1%})")
107: 
108:     # Category distribution analysis
109:     train_categories = defaultdict(int)
110:     val_categories = defaultdict(int)
111:     test_categories = defaultdict(int)
112: 
113:     for q in train_questions:
114:         train_categories[q.category] += 1
115:     for q in val_questions:
116:         val_categories[q.category] += 1
117:     for q in test_questions:
118:         test_categories[q.category] += 1
119: 
120:     all_categories = set(train_categories.keys()) | set(val_categories.keys()) | set(test_categories.keys())
121:     print(f"\nCategory distribution ({len(all_categories)} categories):")
122: 
123:     for category in sorted(all_categories)[:5]:  # Show first 5 categories
124:         orig_count = len(category_groups[category])
125:         train_count = train_categories.get(category, 0)
126:         val_count = val_categories.get(category, 0)
127:         test_count = test_categories.get(category, 0)
128:         print(f"  {category}: {train_count}/{val_count}/{test_count} (orig: {orig_count})")
129: 
130:     if len(all_categories) > 5:
131:         print(f"  ... and {len(all_categories) - 5} more categories")
132: 
133:     return train_questions, val_questions, test_questions
134: 
135: 
136: def save_splits(
137:     train: List[TossupQuestion],
138:     val: List[TossupQuestion],
139:     test: List[TossupQuestion],
140:     output_dir: str = "data"
141: ) -> None:
142:     """
143:     Save dataset splits to JSON files with metadata.
144: 
145:     Parameters
146:     ----------
147:     train : List[TossupQuestion]
148:         Training split
149:     val : List[TossupQuestion]
150:         Validation split
151:     test : List[TossupQuestion]
152:         Test split
153:     output_dir : str
154:         Directory to save split files
155:     """
156:     output_path = Path(output_dir)
157:     output_path.mkdir(parents=True, exist_ok=True)
158: 
159:     # Helper to convert TossupQuestion to dict
160:     def questions_to_dict(questions: List[TossupQuestion]) -> List[Dict[str, Any]]:
161:         return [
162:             {
163:                 "qid": q.qid,
164:                 "question": q.question,
165:                 "tokens": q.tokens,
166:                 "answer_primary": q.answer_primary,
167:                 "clean_answers": q.clean_answers,
168:                 "run_indices": q.run_indices,
169:                 "human_buzz_positions": q.human_buzz_positions,
170:                 "category": q.category,
171:                 "cumulative_prefixes": q.cumulative_prefixes
172:             }
173:             for q in questions
174:         ]
175: 
176:     # Calculate category distributions for metadata
177:     def get_category_distribution(questions: List[TossupQuestion]) -> Dict[str, int]:
178:         dist = defaultdict(int)
179:         for q in questions:
180:             dist[q.category] += 1
181:         return dict(dist)
182: 
183:     # Save each split with metadata
184:     splits = [
185:         ("train_dataset.json", train),
186:         ("val_dataset.json", val),
187:         ("test_dataset.json", test)
188:     ]
189: 
190:     for filename, questions in splits:
191:         filepath = output_path / filename
192: 
193:         data = {
194:             "metadata": {
195:                 "total_questions": len(questions),
196:                 "categories": len(set(q.category for q in questions)),
197:                 "category_distribution": get_category_distribution(questions),
198:                 "split_type": filename.replace("_dataset.json", "")
199:             },
200:             "questions": questions_to_dict(questions)
201:         }
202: 
203:         with open(filepath, 'w', encoding='utf-8') as f:
204:             json.dump(data, f, indent=2, ensure_ascii=False)
205: 
206:         print(f"Saved {len(questions)} questions to {filepath}")
207: 
208:     # Save combined metadata file
209:     metadata_path = output_path / "split_metadata.json"
210:     metadata = {
211:         "train": {
212:             "count": len(train),
213:             "categories": get_category_distribution(train)
214:         },
215:         "val": {
216:             "count": len(val),
217:             "categories": get_category_distribution(val)
218:         },
219:         "test": {
220:             "count": len(test),
221:             "categories": get_category_distribution(test)
222:         },
223:         "total_questions": len(train) + len(val) + len(test),
224:         "split_ratios": [
225:             len(train) / (len(train) + len(val) + len(test)),
226:             len(val) / (len(train) + len(val) + len(test)),
227:             len(test) / (len(train) + len(val) + len(test))
228:         ]
229:     }
230: 
231:     with open(metadata_path, 'w', encoding='utf-8') as f:
232:         json.dump(metadata, f, indent=2)
233: 
234:     print(f"\nSaved split metadata to {metadata_path}")
````

## File: qb_data/dspy_answer_profiles.py
````python
 1: """Optional DSPy-based answer profile augmentation.
 2: 
 3: Generates richer answer profiles using an LM when the ``dspy`` extra is
 4: installed and enabled.  The extractive ``AnswerProfileBuilder`` remains
 5: the default and fallback — this module only augments, never replaces.
 6: 
 7: This module requires the ``dspy`` optional extra.
 8: """
 9: 
10: from __future__ import annotations
11: 
12: from typing import Any
13: 
14: 
15: def build_dspy_profiles(
16:     answers: list[str],
17:     existing_profiles: dict[str, str],
18:     dspy_config: dict[str, Any],
19:     max_answers: int = 100,
20: ) -> dict[str, str]:
21:     """Generate LM-augmented answer profiles via DSPy.
22: 
23:     Leave-one-out discipline depends on the *caller* providing
24:     ``existing_profiles`` that already exclude the current question
25:     (as ``AnswerProfileBuilder.profile_for_answer(answer, exclude_qid)``
26:     does).  This function itself does not receive per-question exclusion
27:     context — it augments whatever profiles it is given.
28: 
29:     Parameters
30:     ----------
31:     answers : list[str]
32:         Answer strings to generate profiles for.
33:     existing_profiles : dict[str, str]
34:         Extractive profiles from ``AnswerProfileBuilder``.
35:     dspy_config : dict
36:         DSPy configuration section from YAML.
37:     max_answers : int
38:         Cap on number of answers to augment.
39: 
40:     Returns
41:     -------
42:     dict[str, str]
43:         Mapping from answer to augmented profile text.  Falls back to
44:         the extractive profile when augmentation fails.
45:     """
46:     try:
47:         import dspy
48:     except ImportError as exc:
49:         raise ImportError(
50:             "DSPy answer profile augmentation requires the dspy package. "
51:             "Install with: pip install -e '.[dspy]'"
52:         ) from exc
53: 
54:     lm_name = dspy_config.get("model", "openai/gpt-4o-mini")
55:     lm = dspy.LM(lm_name)
56:     dspy.configure(lm=lm)
57: 
58:     class AnswerProfileSignature(dspy.Signature):
59:         """Generate a rich factual profile for a quiz bowl answer."""
60:         answer: str = dspy.InputField(desc="the answer entity")
61:         existing_profile: str = dspy.InputField(desc="extractive profile from question corpus")
62:         augmented_profile: str = dspy.OutputField(desc="enriched factual profile suitable for quiz bowl scoring")
63: 
64:     generator = dspy.Predict(AnswerProfileSignature)
65: 
66:     import logging
67: 
68:     logger = logging.getLogger(__name__)
69: 
70:     result: dict[str, str] = {}
71:     n_augmented = 0
72:     n_fallback = 0
73:     for answer in answers[:max_answers]:
74:         existing = existing_profiles.get(answer, "")
75:         try:
76:             pred = generator(answer=answer, existing_profile=existing)
77:             result[answer] = pred.augmented_profile
78:             n_augmented += 1
79:         except Exception as exc:
80:             logger.warning("DSPy augmentation failed for %r: %s", answer, exc)
81:             result[answer] = existing
82:             n_fallback += 1
83: 
84:     for answer in answers[max_answers:]:
85:         result[answer] = existing_profiles.get(answer, "")
86: 
87:     if n_fallback:
88:         logger.info(
89:             "DSPy profiles: %d augmented, %d fell back to extractive",
90:             n_augmented, n_fallback,
91:         )
92: 
93:     return result
````

## File: qb_data/mc_builder.py
````python
  1: """Multiple-choice question builder with anti-artifact guards."""
  2: 
  3: from __future__ import annotations
  4: 
  5: import random
  6: from dataclasses import dataclass
  7: from difflib import SequenceMatcher
  8: from typing import Any, Dict, List, Optional, Set, Tuple
  9: 
 10: import numpy as np
 11: from sklearn.feature_extraction.text import TfidfVectorizer
 12: from sklearn.metrics.pairwise import cosine_similarity
 13: 
 14: from qb_data.answer_profiles import AnswerProfileBuilder
 15: from qb_data.data_loader import TossupQuestion
 16: from qb_data.text_utils import normalize_answer
 17: 
 18: 
 19: @dataclass
 20: class MCQuestion(TossupQuestion):
 21:     """A tossup question with multiple-choice options.
 22: 
 23:     Extends TossupQuestion with fields for multiple-choice presentation
 24:     and tracking of distractor generation strategy.
 25:     """
 26:     options: List[str]
 27:     gold_index: int
 28:     option_profiles: List[str]
 29:     option_answer_primary: List[str]
 30:     distractor_strategy: str
 31: 
 32: 
 33: def _normalized_edit_distance(a: str, b: str) -> float:
 34:     """Compute normalized edit distance between two strings.
 35: 
 36:     Args:
 37:         a: First string.
 38:         b: Second string.
 39: 
 40:     Returns:
 41:         Distance between 0 (identical) and 1 (completely different).
 42:     """
 43:     return 1.0 - SequenceMatcher(None, a, b).ratio()
 44: 
 45: 
 46: def _token_overlap(a: str, b: str) -> float:
 47:     """Compute token overlap between two strings.
 48: 
 49:     Args:
 50:         a: First string.
 51:         b: Second string.
 52: 
 53:     Returns:
 54:         Fraction of overlapping tokens (0 to 1).
 55:     """
 56:     a_tokens = set(a.lower().split())
 57:     b_tokens = set(b.lower().split())
 58:     if not a_tokens or not b_tokens:
 59:         return 0.0
 60:     return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
 61: 
 62: 
 63: class MCBuilder:
 64:     """Builder for multiple-choice questions with anti-artifact guards.
 65: 
 66:     This class implements four layers of guards to prevent spurious patterns
 67:     that agents could exploit:
 68:     1. Alias collision guard: Prevents distractors that are aliases of the gold answer
 69:     2. Duplicate guard: Prevents distractors with high token overlap
 70:     3. Length ratio guard: Prevents distractors much longer/shorter than others
 71:     4. Question overlap guard: Prevents answers that appear in the question text
 72:     """
 73: 
 74:     def __init__(
 75:         self,
 76:         K: int = 4,
 77:         strategy: str = "sbert_profile",
 78:         alias_edit_distance_threshold: float = 0.2,
 79:         duplicate_token_overlap_threshold: float = 0.8,
 80:         max_length_ratio: float = 3.0,
 81:         random_seed: int = 13,
 82:         embedding_model: str = "all-MiniLM-L6-v2",
 83:         openai_model: str = "text-embedding-3-small",
 84:         variable_K: bool = False,
 85:         min_K: int = 2,
 86:         max_K: int | None = None,
 87:     ):
 88:         """Initialize the MC builder.
 89: 
 90:         Args:
 91:             K: Default number of answer choices (must be >= 2).
 92:             strategy: Distractor selection strategy.
 93:             alias_edit_distance_threshold: Max edit distance for alias detection.
 94:             duplicate_token_overlap_threshold: Max token overlap between options.
 95:             max_length_ratio: Max ratio between longest and shortest option.
 96:             random_seed: Random seed for reproducibility.
 97:             embedding_model: SentenceTransformer model name for ``sbert_profile``.
 98:             openai_model: OpenAI embedding model for ``openai_profile``.
 99:             variable_K: If True, sample target K per question from
100:                 ``[min_K, max_K or K]``.
101:             min_K: Minimum K when ``variable_K`` is True.
102:             max_K: Maximum K when ``variable_K`` is True.  Defaults to ``K``.
103:         """
104:         if K < 2:
105:             raise ValueError("K must be >= 2")
106:         self.K = K
107:         self.variable_K = variable_K
108:         self.min_K = max(2, min_K)
109:         self.max_K = max_K if max_K is not None else K
110:         self.strategy = strategy
111:         self.alias_edit_distance_threshold = alias_edit_distance_threshold
112:         self.duplicate_token_overlap_threshold = duplicate_token_overlap_threshold
113:         self.max_length_ratio = max_length_ratio
114:         self.rng = random.Random(random_seed)
115:         self.embedding_model = embedding_model
116:         self.openai_model = openai_model
117: 
118:     def _prepare_lookup(
119:         self, questions: List[TossupQuestion]
120:     ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], List[str]]:
121:         """Prepare lookup structures for answer processing.
122: 
123:         Args:
124:             questions: List of tossup questions.
125: 
126:         Returns:
127:             Tuple of (answer_to_aliases, answer_to_category, answer_to_norm, answers).
128:         """
129:         answer_to_aliases: Dict[str, Set[str]] = {}
130:         answer_to_category: Dict[str, str] = {}
131: 
132:         for q in questions:
133:             # Collect all aliases for each answer
134:             aliases = answer_to_aliases.setdefault(q.answer_primary, set())
135:             aliases.update(str(alias) for alias in q.clean_answers)
136:             aliases.add(q.answer_primary)
137: 
138:             # Track category for category-based distractor selection
139:             if q.category and q.answer_primary not in answer_to_category:
140:                 answer_to_category[q.answer_primary] = q.category
141: 
142:         # Convert to sorted lists for consistency
143:         answer_to_aliases_list = {k: sorted(v) for k, v in answer_to_aliases.items()}
144:         answers = sorted(answer_to_aliases_list.keys())
145:         answer_to_norm = {a: str(normalize_answer(a)) for a in answers}
146: 
147:         return answer_to_aliases_list, answer_to_category, answer_to_norm, answers
148: 
149:     def _rank_by_similarity(
150:         self,
151:         sim: np.ndarray,
152:         answers: List[str],
153:         answer_idx: Dict[str, int],
154:         M: int,
155:     ) -> Dict[str, List[str]]:
156:         """Rank distractors for each answer using a similarity matrix.
157: 
158:         Uses ``np.argpartition`` for top-M retrieval when M < N-1,
159:         reducing per-answer work from O(N log N) to O(N + M log M).
160: 
161:         Parameters
162:         ----------
163:         sim : np.ndarray
164:             Pairwise similarity matrix of shape (N, N).
165:         answers : list[str]
166:             Ordered answer strings corresponding to matrix rows/cols.
167:         answer_idx : dict[str, int]
168:             Mapping from answer string to its index in *sim*.
169:         M : int
170:             Number of top candidates to retain per answer.
171: 
172:         Returns
173:         -------
174:         dict[str, list[str]]
175:             Each answer mapped to its ranked distractor list (length <= M).
176:         """
177:         N = len(answers)
178:         rankings: Dict[str, List[str]] = {}
179:         for answer in answers:
180:             idx = answer_idx[answer]
181:             row = sim[idx]
182:             if M >= N - 1:
183:                 # Small N: full sort (no benefit from partition)
184:                 order = np.argsort(-row).tolist()
185:             else:
186:                 # Top-M retrieval: O(N) partition + O(M log M) sort
187:                 top_m_idx = np.argpartition(-row, M)[:M]
188:                 top_m_idx = top_m_idx[np.argsort(-row[top_m_idx])]
189:                 order = top_m_idx.tolist()
190:             rankings[answer] = [answers[i] for i in order if answers[i] != answer]
191:         return rankings
192: 
193:     def _compute_rankings(
194:         self,
195:         answers: List[str],
196:         answer_profiles: Dict[str, str],
197:         answer_to_category: Dict[str, str],
198:     ) -> Dict[str, List[str]]:
199:         """Compute distractor rankings for each answer.
200: 
201:         For profile-based strategies, uses top-M retrieval via
202:         ``np.argpartition`` instead of full ``np.argsort`` to reduce
203:         per-answer complexity from O(N log N) to O(N + M log M) and
204:         total memory from O(N^2) to O(N*M), where M = max(5*K, 30).
205: 
206:         Args:
207:             answers: List of all unique answers.
208:             answer_profiles: Dictionary mapping answers to their profiles.
209:             answer_to_category: Dictionary mapping answers to categories.
210: 
211:         Returns:
212:             Dictionary mapping each answer to a ranked list of distractors.
213:         """
214:         if self.strategy == "category_random":
215:             # Random selection within the same category
216:             rankings: Dict[str, List[str]] = {}
217:             for answer in answers:
218:                 category = answer_to_category.get(answer, "")
219:                 # First try same category, then fall back to all answers
220:                 candidates = [
221:                     a for a in answers
222:                     if a != answer and answer_to_category.get(a, "") == category
223:                 ]
224:                 if len(candidates) < self.K - 1:
225:                     candidates = [a for a in answers if a != answer]
226:                 self.rng.shuffle(candidates)
227:                 rankings[answer] = candidates
228:             return rankings
229: 
230:         # Profile-based ranking strategies
231:         docs = [answer_profiles[a] for a in answers]
232:         answer_idx = {a: i for i, a in enumerate(answers)}
233:         M = min(max(5 * self.K, 30), len(answers) - 1)
234: 
235:         if self.strategy == "tfidf_profile":
236:             # TF-IDF based similarity
237:             vectorizer = TfidfVectorizer(stop_words="english")
238:             matrix = vectorizer.fit_transform(docs)
239:             sim = cosine_similarity(matrix, matrix)
240:             return self._rank_by_similarity(sim, answers, answer_idx, M)
241: 
242:         if self.strategy in {"sbert_profile", "openai_profile"}:
243:             if self.strategy == "sbert_profile":
244:                 # One-shot SBERT encoding for distractor ranking.
245:                 # This is separate from the SBERTLikelihood runtime cache
246:                 # because it runs only during MC dataset construction.
247:                 from sentence_transformers import SentenceTransformer
248:                 encoder = SentenceTransformer(self.embedding_model)
249:                 embeddings = encoder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
250:                 sim = embeddings @ embeddings.T
251:             else:
252:                 from models.likelihoods import OpenAILikelihood
253: 
254:                 likelihood = OpenAILikelihood(model=self.openai_model)
255:                 embeddings = likelihood.embed_and_cache(docs)
256:                 sim = embeddings @ embeddings.T
257: 
258:             return self._rank_by_similarity(sim, answers, answer_idx, M)
259: 
260:         raise ValueError(f"Unknown distractor strategy: {self.strategy}")
261: 
262:     def _aliases_collide(self, candidate: str, gold_aliases: List[str]) -> bool:
263:         """Check if a candidate is too similar to any gold answer alias.
264: 
265:         Args:
266:             candidate: Candidate distractor.
267:             gold_aliases: List of aliases for the gold answer.
268: 
269:         Returns:
270:             True if the candidate collides with a gold alias.
271:         """
272:         candidate_norm = str(normalize_answer(candidate))
273:         gold_norms = [str(normalize_answer(alias)) for alias in gold_aliases]
274: 
275:         # Check exact match
276:         if candidate_norm in set(gold_norms):
277:             return True
278: 
279:         # Check edit distance
280:         for gold_norm in gold_norms:
281:             if _normalized_edit_distance(candidate_norm, gold_norm) < self.alias_edit_distance_threshold:
282:                 return True
283: 
284:         return False
285: 
286:     def _violates_duplicate_guard(self, candidate: str, selected: List[str]) -> bool:
287:         """Check if candidate has too much token overlap with already selected options.
288: 
289:         Args:
290:             candidate: Candidate distractor.
291:             selected: List of already selected distractors.
292: 
293:         Returns:
294:             True if the candidate has too much overlap.
295:         """
296:         for chosen in selected:
297:             if _token_overlap(candidate, chosen) > self.duplicate_token_overlap_threshold:
298:                 return True
299:         return False
300: 
301:     def _violates_length_ratio_guard(self, options: List[str]) -> bool:
302:         """Check if options have too different lengths.
303: 
304:         Args:
305:             options: List of all options.
306: 
307:         Returns:
308:             True if the length ratio is too high.
309:         """
310:         lengths = [max(1, len(o.split())) for o in options]
311:         return (max(lengths) / min(lengths)) > self.max_length_ratio
312: 
313:     def _violates_question_overlap_guard(self, question: str, options: List[str]) -> bool:
314:         """Check if any option appears in the question text.
315: 
316:         Args:
317:             question: Question text.
318:             options: List of answer options.
319: 
320:         Returns:
321:             True if any option appears in the question.
322:         """
323:         q_norm = str(normalize_answer(question))
324:         for option in options:
325:             o_norm = str(normalize_answer(option))
326:             if o_norm and o_norm in q_norm:
327:                 return True
328:         return False
329: 
330:     def _target_k(self) -> int:
331:         """Return the target K for the next question.
332: 
333:         When ``variable_K`` is False, always returns ``self.K``.
334:         When True, samples uniformly from ``[min_K, max_K]``.
335:         """
336:         if not self.variable_K:
337:             return self.K
338:         return self.rng.randint(self.min_K, self.max_K)
339: 
340:     def build(
341:         self,
342:         questions: List[TossupQuestion],
343:         profile_builder: AnswerProfileBuilder,
344:     ) -> List[MCQuestion]:
345:         """Build multiple-choice questions with anti-artifact guards.
346: 
347:         Args:
348:             questions: List of tossup questions.
349:             profile_builder: Profile builder for answer representations.
350: 
351:         Returns:
352:             List of MCQuestion objects that passed all guards.
353:         """
354:         if not questions:
355:             return []
356: 
357:         # Build answer profiles
358:         profile_builder.fit(questions)
359:         answer_profiles = profile_builder.build_profiles(questions)
360: 
361:         # Prepare lookup structures
362:         answer_to_aliases, answer_to_category, _answer_to_norm, answers = self._prepare_lookup(questions)
363: 
364:         # Compute distractor rankings
365:         rankings = self._compute_rankings(answers, answer_profiles, answer_to_category)
366: 
367:         mc_questions: List[MCQuestion] = []
368: 
369:         for q in questions:
370:             target_k = self._target_k()
371:             gold = q.answer_primary
372:             gold_aliases = answer_to_aliases.get(gold, [gold])
373:             ranked = rankings.get(gold, [a for a in answers if a != gold])
374:             selected: List[str] = []
375: 
376:             # Select distractors from ranked list
377:             for candidate in ranked:
378:                 if candidate == gold:
379:                     continue
380:                 if self._aliases_collide(candidate, gold_aliases):
381:                     continue
382:                 if self._violates_duplicate_guard(candidate, selected):
383:                     continue
384:                 selected.append(candidate)
385:                 if len(selected) >= target_k - 1:
386:                     break
387: 
388:             # If not enough distractors from ranking, try random fallback
389:             if len(selected) < target_k - 1:
390:                 fallback = [a for a in answers if a not in selected and a != gold]
391:                 self.rng.shuffle(fallback)
392:                 for candidate in fallback:
393:                     if self._aliases_collide(candidate, gold_aliases):
394:                         continue
395:                     if self._violates_duplicate_guard(candidate, selected):
396:                         continue
397:                     selected.append(candidate)
398:                     if len(selected) >= target_k - 1:
399:                         break
400: 
401:             # Skip question if we can't find enough valid distractors
402:             if len(selected) < target_k - 1:
403:                 continue
404: 
405:             # Create options and shuffle
406:             option_answer_primary = [gold] + selected[:target_k - 1]
407:             self.rng.shuffle(option_answer_primary)
408:             gold_index = option_answer_primary.index(gold)
409:             options = option_answer_primary[:]
410: 
411:             # Apply guard 3: Check length ratio
412:             if self._violates_length_ratio_guard(options):
413:                 continue
414: 
415:             # Apply guard 4: Check question overlap
416:             if self._violates_question_overlap_guard(q.question, options):
417:                 continue
418: 
419:             # Build option profiles with leave-one-out for gold
420:             option_profiles: List[str] = []
421:             for answer in option_answer_primary:
422:                 exclude_qid = q.qid if answer == gold else None
423:                 option_profiles.append(
424:                     profile_builder.profile_for_answer(answer, exclude_qid=exclude_qid)
425:                 )
426: 
427:             # Create MCQuestion
428:             mc_questions.append(
429:                 MCQuestion(
430:                     qid=q.qid,
431:                     question=q.question,
432:                     tokens=q.tokens,
433:                     answer_primary=q.answer_primary,
434:                     clean_answers=q.clean_answers,
435:                     run_indices=q.run_indices,
436:                     human_buzz_positions=q.human_buzz_positions,
437:                     category=q.category,
438:                     cumulative_prefixes=q.cumulative_prefixes,
439:                     options=options,
440:                     gold_index=gold_index,
441:                     option_profiles=option_profiles,
442:                     option_answer_primary=option_answer_primary,
443:                     distractor_strategy=self.strategy,
444:                 )
445:             )
446: 
447:         return mc_questions
448: 
449: 
450: def build_mc_questions(
451:     questions: List[TossupQuestion],
452:     K: int,
453:     strategy: str,
454:     profile_builder: AnswerProfileBuilder,
455:     guards: Optional[Dict[str, Any]] = None,
456:     random_seed: int = 13,
457: ) -> List[MCQuestion]:
458:     """Factory function to build multiple-choice questions.
459: 
460:     Args:
461:         questions: List of tossup questions.
462:         K: Number of answer choices.
463:         strategy: Distractor selection strategy.
464:         profile_builder: Profile builder for answer representations.
465:         guards: Optional dictionary of guard thresholds.
466:         random_seed: Random seed for reproducibility.
467: 
468:     Returns:
469:         List of MCQuestion objects that passed all guards.
470:     """
471:     guards = guards or {}
472:     builder = MCBuilder(
473:         K=K,
474:         strategy=strategy,
475:         alias_edit_distance_threshold=float(guards.get("alias_edit_distance_threshold", 0.2)),
476:         duplicate_token_overlap_threshold=float(guards.get("duplicate_token_overlap_threshold", 0.8)),
477:         max_length_ratio=float(guards.get("max_length_ratio", 3.0)),
478:         random_seed=random_seed,
479:     )
480:     return builder.build(questions=questions, profile_builder=profile_builder)
````

## File: scripts/_common.py
````python
  1: """Shared utilities for pipeline scripts.
  2: 
  3: Provides config loading, JSON serialization, MC question deserialization,
  4: and path constants used across all pipeline scripts (build, baseline, train,
  5: evaluate).
  6: 
  7: Ported from qb-rl reference implementation with import path adaptations
  8: for the unified qanta-buzzer codebase.
  9: """
 10: 
 11: from __future__ import annotations
 12: 
 13: import argparse
 14: import json
 15: from dataclasses import asdict, is_dataclass
 16: from pathlib import Path
 17: from typing import Any
 18: 
 19: from models.likelihoods import LikelihoodModel, build_likelihood_from_config
 20: from qb_data.config import load_config as load_yaml_config
 21: from qb_data.mc_builder import MCQuestion
 22: 
 23: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 24: 
 25: 
 26: def _parse_value(value: str) -> Any:
 27:     """Parse a CLI override value string into a typed Python value.
 28: 
 29:     Tries JSON first, then bool/int/float, and falls back to str.
 30:     """
 31:     try:
 32:         return json.loads(value)
 33:     except json.JSONDecodeError:
 34:         pass
 35:     if value.lower() == "true":
 36:         return True
 37:     if value.lower() == "false":
 38:         return False
 39:     if value.lstrip("-").isdigit():
 40:         return int(value)
 41:     try:
 42:         return float(value)
 43:     except ValueError:
 44:         return value
 45: 
 46: 
 47: def parse_overrides(args: argparse.Namespace) -> dict[str, Any]:
 48:     """Parse CLI override arguments into flat dotted-key overrides.
 49: 
 50:     Returns a dict with dotted keys (e.g. ``{"data.K": 5}``) that
 51:     ``merge_overrides`` can apply leaf-by-leaf without clobbering
 52:     sibling config entries.
 53: 
 54:     Parameters
 55:     ----------
 56:     args : argparse.Namespace
 57:         Parsed CLI arguments.  Positional ``overrides`` are
 58:         ``key=value`` strings where *key* uses dot-notation
 59:         (e.g. ``data.K=5``).
 60: 
 61:     Returns
 62:     -------
 63:     dict[str, Any]
 64:         Flat dotted-key overrides ready for ``merge_overrides()``.
 65:     """
 66:     overrides: dict[str, Any] = {}
 67:     if hasattr(args, "overrides") and args.overrides:
 68:         for token in args.overrides:
 69:             if "=" not in token:
 70:                 continue
 71:             key, value = token.split("=", 1)
 72:             overrides[key] = _parse_value(value)
 73:     return overrides
 74: DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
 75: ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
 76: 
 77: 
 78: def load_config(config_path: str | None = None, smoke: bool = False) -> dict[str, Any]:
 79:     """Load YAML configuration from a file path.
 80: 
 81:     Parameters
 82:     ----------
 83:     config_path : str or None
 84:         Path to YAML config file. If None, loads ``configs/default.yaml``.
 85: 
 86:     Returns
 87:     -------
 88:     dict[str, Any]
 89:         Parsed config dict with nested structure (data, likelihood,
 90:         environment, ppo, etc.).
 91:     """
 92:     return load_yaml_config(config_path, smoke=smoke)
 93: 
 94: 
 95: def build_likelihood_model(config: dict[str, Any], mc_questions: list[MCQuestion]):
 96:     """Build a likelihood model with shared TF-IDF corpus handling."""
 97:     corpus = None
 98:     if config["likelihood"].get("model") == "tfidf":
 99:         corpus = [q.question for q in mc_questions] + [
100:             profile
101:             for question in mc_questions
102:             for profile in question.option_profiles
103:         ]
104:     return build_likelihood_from_config(config, corpus_texts=corpus)
105: 
106: 
107: def ensure_dir(path: str | Path) -> Path:
108:     """Create a directory (and parents) if it does not exist.
109: 
110:     Parameters
111:     ----------
112:     path : str or Path
113:         Directory path to create.
114: 
115:     Returns
116:     -------
117:     Path
118:         The created (or existing) directory path.
119:     """
120:     p = Path(path)
121:     p.mkdir(parents=True, exist_ok=True)
122:     return p
123: 
124: 
125: def to_serializable(item: Any) -> Any:
126:     """Recursively convert dataclasses to dicts for JSON serialization.
127: 
128:     Parameters
129:     ----------
130:     item : Any
131:         Object to convert. Dataclasses are converted via ``asdict()``,
132:         dicts and lists are processed recursively.
133: 
134:     Returns
135:     -------
136:     Any
137:         JSON-serializable version of the input.
138:     """
139:     if is_dataclass(item):
140:         return asdict(item)
141:     if isinstance(item, dict):
142:         return {k: to_serializable(v) for k, v in item.items()}
143:     if isinstance(item, list):
144:         return [to_serializable(v) for v in item]
145:     return item
146: 
147: 
148: def save_json(path: str | Path, data: Any) -> Path:
149:     """Save data to a JSON file, creating parent directories as needed.
150: 
151:     Applies ``to_serializable`` to convert dataclasses before writing.
152: 
153:     Parameters
154:     ----------
155:     path : str or Path
156:         Output file path.
157:     data : Any
158:         Data to serialize. Dataclasses are converted to dicts automatically.
159: 
160:     Returns
161:     -------
162:     Path
163:         The path where the JSON was written.
164:     """
165:     p = Path(path)
166:     p.parent.mkdir(parents=True, exist_ok=True)
167:     with p.open("w", encoding="utf-8") as f:
168:         json.dump(to_serializable(data), f, indent=2)
169:     return p
170: 
171: 
172: def load_json(path: str | Path) -> Any:
173:     """Load data from a JSON file.
174: 
175:     Parameters
176:     ----------
177:     path : str or Path
178:         Path to JSON file.
179: 
180:     Returns
181:     -------
182:     Any
183:         Parsed JSON data.
184:     """
185:     with Path(path).open("r", encoding="utf-8") as f:
186:         return json.load(f)
187: 
188: 
189: def mc_question_from_dict(row: dict[str, Any]) -> MCQuestion:
190:     """Reconstruct an MCQuestion dataclass from a JSON-deserialized dict.
191: 
192:     Parameters
193:     ----------
194:     row : dict[str, Any]
195:         Dictionary with all MCQuestion fields.
196: 
197:     Returns
198:     -------
199:     MCQuestion
200:         Reconstructed MCQuestion instance.
201:     """
202:     return MCQuestion(
203:         qid=row["qid"],
204:         question=row["question"],
205:         tokens=list(row["tokens"]),
206:         answer_primary=row["answer_primary"],
207:         clean_answers=list(row["clean_answers"]),
208:         run_indices=list(row["run_indices"]),
209:         human_buzz_positions=row.get("human_buzz_positions"),
210:         category=row.get("category", ""),
211:         cumulative_prefixes=list(row["cumulative_prefixes"]),
212:         options=list(row["options"]),
213:         gold_index=int(row["gold_index"]),
214:         option_profiles=list(row["option_profiles"]),
215:         option_answer_primary=list(row["option_answer_primary"]),
216:         distractor_strategy=row.get("distractor_strategy", "unknown"),
217:     )
218: 
219: 
220: def load_mc_questions(path: str | Path) -> list[MCQuestion]:
221:     """Load and deserialize a list of MCQuestions from a JSON file.
222: 
223:     Parameters
224:     ----------
225:     path : str or Path
226:         Path to JSON file containing a list of serialized MCQuestion dicts.
227: 
228:     Returns
229:     -------
230:     list[MCQuestion]
231:         List of reconstructed MCQuestion instances.
232:     """
233:     raw = load_json(path)
234:     return [mc_question_from_dict(item) for item in raw]
235: 
236: 
237: # ------------------------------------------------------------------ #
238: # Embedding cache persistence helpers
239: # ------------------------------------------------------------------ #
240: 
241: 
242: def embedding_cache_path(config: dict[str, Any]) -> Path:
243:     """Return the resolved embedding cache file path from config.
244: 
245:     Uses ``config['likelihood']['cache_dir']`` (default ``'cache/embeddings'``)
246:     and appends ``'embedding_cache.npz'``.
247: 
248:     Parameters
249:     ----------
250:     config : dict
251:         Full YAML config dict.
252: 
253:     Returns
254:     -------
255:     Path
256:         Absolute path to the embedding cache ``.npz`` file.
257:     """
258:     cache_dir = config.get("likelihood", {}).get("cache_dir", "cache/embeddings")
259:     return PROJECT_ROOT / cache_dir / "embedding_cache.npz"
260: 
261: 
262: def load_embedding_cache(model: LikelihoodModel, config: dict[str, Any]) -> None:
263:     """Load persisted embedding cache into model if file exists.
264: 
265:     Parameters
266:     ----------
267:     model : LikelihoodModel
268:         Likelihood model whose embedding_cache will be populated.
269:     config : dict
270:         Full YAML config dict (used to resolve cache path).
271:     """
272:     path = embedding_cache_path(config)
273:     n = model.load_cache(path)
274:     if n > 0:
275:         print(f"Loaded {n} cached embeddings from {path}")
276: 
277: 
278: def save_embedding_cache(model: LikelihoodModel, config: dict[str, Any]) -> None:
279:     """Persist model's embedding cache to disk.
280: 
281:     Parameters
282:     ----------
283:     model : LikelihoodModel
284:         Likelihood model whose embedding_cache will be saved.
285:     config : dict
286:         Full YAML config dict (used to resolve cache path).
287:     """
288:     path = embedding_cache_path(config)
289:     n = model.save_cache(path)
290:     if n > 0:
291:         print(f"Saved {n} embeddings to {path}")
````

## File: scripts/ci.sh
````bash
 1: #!/usr/bin/env bash
 2: # CI entry point -- runs the full pytest suite from the project venv.
 3: # Exit nonzero on any failure so CI gates catch regressions.
 4: #
 5: # Usage:
 6: #   bash scripts/ci.sh              # full suite
 7: #   bash scripts/ci.sh -k "not t5"  # skip T5-dependent tests
 8: set -euo pipefail
 9: 
10: REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
11: 
12: if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
13:     # shellcheck disable=SC1091
14:     source "$REPO_ROOT/.venv/bin/activate"
15: elif ! command -v pytest &>/dev/null; then
16:     echo "ERROR: No .venv found and pytest not on PATH." >&2
17:     echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
18:     exit 1
19: fi
20: 
21: pytest tests/ "$@"
````

## File: scripts/train_t5_policy.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Train T5 policy with supervised warm-start then PPO fine-tuning.
  4: 
  5: End-to-end pipeline for training a T5PolicyModel on quiz bowl questions:
  6: 1. Supervised warm-start: Train answer selection on complete questions
  7: 2. PPO fine-tuning: Optimize wait/answer policy on incremental episodes
  8: 
  9: Usage:
 10:     # Full pipeline (supervised + PPO)
 11:     python scripts/train_t5_policy.py --config configs/t5_policy.yaml
 12: 
 13:     # Quick smoke test (t5-small, few epochs)
 14:     python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke
 15: 
 16:     # Skip supervised, load pretrained for PPO only
 17:     python scripts/train_t5_policy.py --config configs/t5_policy.yaml \
 18:         --skip-supervised --model-path checkpoints/supervised/best_model
 19: 
 20:     # Custom number of PPO iterations
 21:     python scripts/train_t5_policy.py --config configs/t5_policy.yaml \
 22:         --ppo-iterations 50
 23: """
 24: 
 25: from __future__ import annotations
 26: 
 27: import argparse
 28: import sys
 29: from pathlib import Path
 30: 
 31: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 32: if str(PROJECT_ROOT) not in sys.path:
 33:     sys.path.insert(0, str(PROJECT_ROOT))
 34: 
 35: import yaml
 36: 
 37: from qb_data.config import merge_overrides
 38: from scripts._common import ARTIFACT_DIR, load_mc_questions, parse_overrides
 39: 
 40: 
 41: def parse_args() -> argparse.Namespace:
 42:     """Parse command-line arguments.
 43: 
 44:     Returns
 45:     -------
 46:     argparse.Namespace
 47:         Parsed arguments for training configuration.
 48:     """
 49:     parser = argparse.ArgumentParser(
 50:         description="Train T5 policy with supervised warm-start then PPO.",
 51:     )
 52:     parser.add_argument(
 53:         "--config",
 54:         type=str,
 55:         default=str(PROJECT_ROOT / "configs" / "t5_policy.yaml"),
 56:         help="Path to YAML config file (default: configs/t5_policy.yaml).",
 57:     )
 58:     parser.add_argument(
 59:         "--smoke",
 60:         action="store_true",
 61:         help="Quick test run: uses t5-small, 2 epochs, 4 batch size.",
 62:     )
 63:     parser.add_argument(
 64:         "--skip-supervised",
 65:         action="store_true",
 66:         help="Skip supervised training phase.",
 67:     )
 68:     parser.add_argument(
 69:         "--model-path",
 70:         type=str,
 71:         default=None,
 72:         help="Path to pretrained model checkpoint (required if --skip-supervised).",
 73:     )
 74:     parser.add_argument(
 75:         "--mc-path",
 76:         type=str,
 77:         default=None,
 78:         help="Path to MC dataset JSON file.",
 79:     )
 80:     parser.add_argument(
 81:         "--ppo-iterations",
 82:         type=int,
 83:         default=None,
 84:         help="Override number of PPO iterations from config.",
 85:     )
 86:     parser.add_argument(
 87:         "--hazard-pretrain",
 88:         action="store_true",
 89:         help="Enable the experimental hazard pretraining bridge before PPO.",
 90:     )
 91:     parser.add_argument(
 92:         "--beta-terminal",
 93:         type=float,
 94:         default=1.0,
 95:         help="Terminal survival penalty used by the hazard bridge.",
 96:     )
 97:     parser.add_argument(
 98:         "--freeze-answer-head",
 99:         action="store_true",
100:         help="Freeze the answer head during the hazard bridge phase.",
101:     )
102:     parser.add_argument(
103:         "overrides",
104:         nargs="*",
105:         help="Config overrides: key=value (e.g. model.model_name=t5-base)",
106:     )
107:     return parser.parse_args()
108: 
109: 
110: def load_config_with_overrides(args: argparse.Namespace) -> dict:
111:     """Load YAML config and apply smoke/CLI overrides.
112: 
113:     Parameters
114:     ----------
115:     args : argparse.Namespace
116:         Parsed command-line arguments.
117: 
118:     Returns
119:     -------
120:     dict
121:         Configuration dictionary with overrides applied.
122:     """
123:     with open(args.config) as f:
124:         config = yaml.safe_load(f)
125: 
126:     if args.smoke:
127:         smoke = config.get("smoke", {})
128:         # Override model settings
129:         if "model" in smoke:
130:             for key, val in smoke["model"].items():
131:                 config["model"][key] = val
132:         # Override supervised settings
133:         if "supervised" in smoke:
134:             for key, val in smoke["supervised"].items():
135:                 config["supervised"][key] = val
136:         # Override PPO settings
137:         if "ppo" in smoke:
138:             for key, val in smoke["ppo"].items():
139:                 config["ppo"][key] = val
140:         # Override data settings
141:         if "data" in smoke:
142:             for key, val in smoke["data"].items():
143:                 config["data"][key] = val
144: 
145:     if args.ppo_iterations is not None:
146:         config["ppo"]["iterations"] = args.ppo_iterations
147: 
148:     return config
149: 
150: 
151: def flatten_config(config: dict) -> dict:
152:     """Flatten nested config sections into a single dict for trainer APIs.
153: 
154:     Parameters
155:     ----------
156:     config : dict
157:         Nested config dict with sections (model, supervised, ppo, data).
158: 
159:     Returns
160:     -------
161:     dict
162:         Flat config dict with prefixed keys for each trainer.
163:     """
164:     flat = {}
165: 
166:     # Model section
167:     model = config.get("model", {})
168:     flat["model_name"] = model.get("model_name", "t5-large")
169:     device = model.get("device", "auto")
170:     if device == "auto":
171:         import torch
172:         if torch.cuda.is_available():
173:             device = "cuda"
174:         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
175:             device = "mps"
176:         else:
177:             device = "cpu"
178:     flat["device"] = device
179:     flat["max_input_length"] = model.get("max_input_length", 512)
180:     flat["num_choices"] = model.get("num_choices", config.get("data", {}).get("K", 4))
181: 
182:     # Supervised section
183:     sup = config.get("supervised", {})
184:     flat["supervised_lr"] = sup.get("lr", 3e-4)
185:     flat["supervised_epochs"] = sup.get("epochs", 10)
186:     flat["supervised_batch_size"] = sup.get("batch_size", 8)
187:     flat["supervised_grad_accum_steps"] = sup.get("grad_accum_steps", 4)
188:     flat["max_grad_norm"] = sup.get("max_grad_norm", 1.0)
189:     flat["weight_decay"] = sup.get("weight_decay", 0.01)
190:     flat["checkpoint_dir"] = sup.get("checkpoint_dir", "checkpoints")
191: 
192:     # PPO section
193:     ppo = config.get("ppo", {})
194:     flat["ppo_lr"] = ppo.get("lr", 1e-5)
195:     flat["ppo_iterations"] = ppo.get("iterations", 100)
196:     flat["ppo_batch_size"] = ppo.get("batch_size", 8)
197:     flat["ppo_epochs_per_iter"] = ppo.get("epochs_per_iter", 4)
198:     flat["ppo_gamma"] = ppo.get("gamma", 0.99)
199:     flat["ppo_gae_lambda"] = ppo.get("gae_lambda", 0.95)
200:     flat["ppo_clip_ratio"] = ppo.get("clip_ratio", 0.2)
201:     flat["ppo_value_coef"] = ppo.get("value_coef", 0.5)
202:     flat["ppo_entropy_coef"] = ppo.get("entropy_coef", 0.01)
203:     flat["ppo_max_grad_norm"] = ppo.get("max_grad_norm", 0.5)
204:     flat["ppo_episodes_per_iter"] = ppo.get("episodes_per_iter", 16)
205:     flat["eval_interval"] = ppo.get("eval_interval", 10)
206:     flat["save_interval"] = ppo.get("save_interval", 20)
207: 
208:     return flat
209: 
210: 
211: def load_questions(args: argparse.Namespace, config: dict) -> list:
212:     """Load MC questions from file or fallback paths.
213: 
214:     Parameters
215:     ----------
216:     args : argparse.Namespace
217:         Parsed arguments, may have mc_path override.
218:     config : dict
219:         Config dict with data section.
220: 
221:     Returns
222:     -------
223:     list
224:         List of MCQuestion instances.
225:     """
226:     if args.mc_path:
227:         mc_path = Path(args.mc_path)
228:     else:
229:         # Try standard locations
230:         candidates = [
231:             ARTIFACT_DIR / "main" / "mc_dataset.json",
232:             ARTIFACT_DIR / "smoke" / "mc_dataset.json",
233:             PROJECT_ROOT / "data" / "processed" / "mc_dataset.json",
234:         ]
235:         mc_path = None
236:         for candidate in candidates:
237:             if candidate.exists():
238:                 mc_path = candidate
239:                 break
240: 
241:         if mc_path is None:
242:             print("ERROR: No MC dataset found. Run build_mc_dataset.py first.")
243:             print("Searched locations:")
244:             for c in candidates:
245:                 print(f"  {c}")
246:             sys.exit(1)
247: 
248:     print(f"Loading MC questions from: {mc_path}")
249:     questions = load_mc_questions(mc_path)
250:     print(f"Loaded {len(questions)} questions")
251: 
252:     # Apply max_questions limit (smoke mode)
253:     max_questions = config.get("data", {}).get("max_questions", None)
254:     if max_questions and len(questions) > max_questions:
255:         questions = questions[:max_questions]
256:         print(f"Limited to {max_questions} questions (smoke mode)")
257: 
258:     return questions
259: 
260: 
261: def validate_args(args: argparse.Namespace) -> None:
262:     """Validate CLI arguments and reject unsupported bridge paths."""
263:     if args.skip_supervised and args.model_path is None:
264:         print("ERROR: --model-path is required when using --skip-supervised")
265:         sys.exit(1)
266:     if args.hazard_pretrain:
267:         raise NotImplementedError(
268:             "Hazard pretraining loop not yet implemented. "
269:             "The math utilities are available in training/hazard_pretrain.py, "
270:             "but the end-to-end bridge has not been wired into train_t5_policy.py yet."
271:         )
272: 
273: 
274: def split_questions(questions: list, config: dict) -> tuple:
275:     """Split questions into train/val/test sets.
276: 
277:     Parameters
278:     ----------
279:     questions : list
280:         Full list of MCQuestion instances.
281:     config : dict
282:         Config dict with data section (train_size, val_size, test_size, seed).
283: 
284:     Returns
285:     -------
286:     tuple[list, list, list]
287:         Train, validation, and test question lists.
288:     """
289:     import random
290: 
291:     data = config.get("data", {})
292:     seed = data.get("seed", 42)
293:     train_size = data.get("train_size", 0.7)
294:     val_size = data.get("val_size", 0.15)
295: 
296:     rng = random.Random(seed)
297:     shuffled = questions[:]
298:     rng.shuffle(shuffled)
299: 
300:     n = len(shuffled)
301:     n_train = int(n * train_size)
302:     n_val = int(n * val_size)
303: 
304:     train_questions = shuffled[:n_train]
305:     val_questions = shuffled[n_train : n_train + n_val]
306:     test_questions = shuffled[n_train + n_val :]
307: 
308:     print(f"Split: {len(train_questions)} train, {len(val_questions)} val, {len(test_questions)} test")
309:     return train_questions, val_questions, test_questions
310: 
311: 
312: def main() -> None:
313:     """Run the full T5 policy training pipeline."""
314:     args = parse_args()
315:     validate_args(args)
316: 
317:     # Load config with overrides
318:     config = load_config_with_overrides(args)
319:     overrides = parse_overrides(args)
320:     if overrides:
321:         config = merge_overrides(config, overrides)
322:     flat_config = flatten_config(config)
323: 
324:     # Load and split dataset
325:     questions = load_questions(args, config)
326:     train_questions, val_questions, test_questions = split_questions(questions, config)
327: 
328:     # Import training modules (lazy to avoid loading transformers until needed)
329:     from training.train_supervised_t5 import run_supervised_training
330:     from training.train_ppo_t5 import run_ppo_training
331: 
332:     # Phase 1: Supervised warm-start (optional)
333:     supervised_model_path = None
334:     if not args.skip_supervised:
335:         print("\n" + "=" * 60)
336:         print("PHASE 1: SUPERVISED WARM-START")
337:         print("=" * 60)
338: 
339:         model, trainer = run_supervised_training(
340:             config=flat_config,
341:             train_questions=train_questions,
342:             val_questions=val_questions,
343:         )
344:         supervised_model_path = str(
345:             trainer.checkpoint_dir / "best_model"
346:         )
347:         print(f"Supervised model saved to: {supervised_model_path}")
348:     else:
349:         supervised_model_path = args.model_path
350:         print(f"\nSkipping supervised training, using model: {supervised_model_path}")
351: 
352:     # Phase 2: PPO fine-tuning
353:     print("\n" + "=" * 60)
354:     print("PHASE 2: PPO FINE-TUNING (T5 Policy)")
355:     print("=" * 60)
356: 
357:     model, trainer = run_ppo_training(
358:         config=flat_config,
359:         train_questions=train_questions,
360:         val_questions=val_questions,
361:         test_questions=test_questions,
362:         pretrained_model_path=supervised_model_path,
363:     )
364: 
365:     print("\n" + "=" * 60)
366:     print("TRAINING COMPLETE")
367:     print("=" * 60)
368:     print(f"Best PPO model saved to: {trainer.checkpoint_dir / 'best_model'}")
369:     print(f"Training history: {trainer.checkpoint_dir / 'history.json'}")
370: 
371: 
372: if __name__ == "__main__":
373:     main()
````

## File: tests/test_dspy_answer_profiles.py
````python
 1: """Tests for qb_data/dspy_answer_profiles.py."""
 2: 
 3: from __future__ import annotations
 4: 
 5: import pytest
 6: 
 7: 
 8: class TestBuildDspyProfiles:
 9:     def test_module_importable_without_dspy(self) -> None:
10:         """The module imports cleanly even when dspy is not installed."""
11:         from qb_data.dspy_answer_profiles import build_dspy_profiles
12:         assert callable(build_dspy_profiles)
13: 
14:     def test_runtime_call_without_dspy_raises(self) -> None:
15:         """Calling build_dspy_profiles without dspy raises ImportError."""
16:         try:
17:             import dspy
18:             pytest.skip("dspy is installed; cannot test import failure")
19:         except ImportError:
20:             from qb_data.dspy_answer_profiles import build_dspy_profiles
21:             with pytest.raises(ImportError, match="dspy"):
22:                 build_dspy_profiles(
23:                     answers=["A"],
24:                     existing_profiles={"A": "existing"},
25:                     dspy_config={"model": "test"},
26:                 )
27: 
28:     def test_with_dspy_installed(self) -> None:
29:         """When dspy IS installed, the function is callable."""
30:         dspy = pytest.importorskip("dspy", reason="dspy not installed")
31:         from qb_data.dspy_answer_profiles import build_dspy_profiles
32:         assert callable(build_dspy_profiles)
````

## File: tests/test_dspy_likelihood.py
````python
 1: """Tests for models/dspy_likelihood.py — DSPy-backed scorer with cache."""
 2: 
 3: from __future__ import annotations
 4: 
 5: import tempfile
 6: from pathlib import Path
 7: 
 8: import numpy as np
 9: import pytest
10: 
11: from models.dspy_likelihood import DSPyLikelihood, _score_cache_key
12: 
13: 
14: def _fake_scorer(clue: str, options: list[str]) -> list[float]:
15:     """Return uniform scores sized to the option list."""
16:     return [1.0 / len(options)] * len(options)
17: 
18: 
19: class TestDSPyLikelihood:
20:     def test_score_returns_ndarray_k(self) -> None:
21:         model = DSPyLikelihood(scorer=_fake_scorer)
22:         scores = model.score("clue text", ["A", "B", "C", "D"])
23:         assert scores.shape == (4,)
24:         assert scores.dtype == np.float32
25: 
26:     def test_repeated_call_hits_cache(self) -> None:
27:         call_count = 0
28: 
29:         def counting_scorer(clue, options):
30:             nonlocal call_count
31:             call_count += 1
32:             return [1.0] * len(options)
33: 
34:         model = DSPyLikelihood(scorer=counting_scorer)
35:         model.score("clue", ["A", "B"])
36:         model.score("clue", ["A", "B"])
37:         assert call_count == 1
38: 
39:     def test_changed_fingerprint_invalidates(self) -> None:
40:         """Different fingerprints produce different cache keys for same input."""
41:         key_v1 = _score_cache_key("clue", ["A", "B"], "v1")
42:         key_v2 = _score_cache_key("clue", ["A", "B"], "v2")
43:         assert key_v1 != key_v2, "Fingerprint must affect cache key"
44: 
45:         model = DSPyLikelihood(scorer=_fake_scorer, program_fingerprint="v1")
46:         model.score("clue", ["A", "B"])
47:         assert key_v1 in model._score_cache
48:         assert key_v2 not in model._score_cache
49: 
50:     def test_persistence_roundtrip(self) -> None:
51:         with tempfile.TemporaryDirectory() as tmpdir:
52:             path = Path(tmpdir) / "cache.npz"
53:             model = DSPyLikelihood(scorer=_fake_scorer)
54:             model.score("clue", ["A", "B", "C"])
55:             saved = model.save_cache(path)
56:             assert saved == 1
57: 
58:             model2 = DSPyLikelihood(scorer=_fake_scorer)
59:             loaded = model2.load_cache(path)
60:             assert loaded == 1
61:             np.testing.assert_array_equal(
62:                 model2.score("clue", ["A", "B", "C"]),
63:                 model.score("clue", ["A", "B", "C"]),
64:             )
65: 
66:     def test_embed_batch_raises(self) -> None:
67:         model = DSPyLikelihood(scorer=_fake_scorer)
68:         with pytest.raises(NotImplementedError):
69:             model._embed_batch(["text"])
70: 
71:     def test_cache_memory_bytes(self) -> None:
72:         model = DSPyLikelihood(scorer=_fake_scorer)
73:         assert model.cache_memory_bytes == 0
74:         model.score("c", ["A"])
75:         assert model.cache_memory_bytes > 0
76: 
77:     def test_score_shape_validation(self) -> None:
78:         """Scorer returning wrong length raises ValueError."""
79:         def bad_scorer(clue, options):
80:             return [1.0, 2.0]  # always 2, ignoring len(options)
81: 
82:         model = DSPyLikelihood(scorer=bad_scorer)
83:         with pytest.raises(ValueError, match="expected"):
84:             model.score("clue", ["A", "B", "C", "D"])
85: 
86:     def test_isinstance_likelihood_model(self) -> None:
87:         """DSPyLikelihood is a proper LikelihoodModel subclass."""
88:         from models.likelihoods import LikelihoodModel
89:         model = DSPyLikelihood(scorer=_fake_scorer)
90:         assert isinstance(model, LikelihoodModel)
````

## File: tests/test_factories.py
````python
  1: """Test suite for factory functions — build_likelihood_from_config and make_env_from_config.
  2: 
  3: Covers:
  4: - LIK-06: build_likelihood_from_config dispatches on config["likelihood"]["model"]
  5: - CFG-02: make_env_from_config constructs TossupMCEnv from YAML config
  6: """
  7: 
  8: from __future__ import annotations
  9: 
 10: import numpy as np
 11: import pytest
 12: 
 13: from models.likelihoods import (
 14:     LikelihoodModel,
 15:     SBERTLikelihood,
 16:     TfIdfLikelihood,
 17:     build_likelihood_from_config,
 18: )
 19: from qb_data.mc_builder import MCQuestion
 20: from qb_env.tossup_env import TossupMCEnv, make_env_from_config
 21: 
 22: 
 23: # ------------------------------------------------------------------ #
 24: # Tests: build_likelihood_from_config (LIK-06)
 25: # ------------------------------------------------------------------ #
 26: 
 27: 
 28: class TestBuildLikelihoodFromConfig:
 29:     """Tests for likelihood model factory function."""
 30: 
 31:     @pytest.fixture
 32:     def stub_sbert_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
 33:         """Stub SBERT model loading so factory tests stay offline-safe."""
 34: 
 35:         def fake_init(self, model_name: str = "all-MiniLM-L6-v2") -> None:
 36:             LikelihoodModel.__init__(self)
 37:             self.model_name = model_name
 38:             self.encoder = object()
 39: 
 40:         monkeypatch.setattr(SBERTLikelihood, "__init__", fake_init)
 41: 
 42:     def test_likelihood_factory_sbert(
 43:         self, sample_config: dict, stub_sbert_init: None
 44:     ) -> None:
 45:         """Config with model='sbert' creates SBERTLikelihood."""
 46:         sample_config["likelihood"]["model"] = "sbert"
 47:         model = build_likelihood_from_config(sample_config)
 48:         assert isinstance(model, SBERTLikelihood), (
 49:             f"Expected SBERTLikelihood, got {type(model).__name__}"
 50:         )
 51: 
 52:     def test_likelihood_factory_tfidf(
 53:         self, sample_config: dict, sample_corpus: list[str]
 54:     ) -> None:
 55:         """Config with model='tfidf' creates TfIdfLikelihood (fitted)."""
 56:         sample_config["likelihood"]["model"] = "tfidf"
 57:         model = build_likelihood_from_config(sample_config, corpus_texts=sample_corpus)
 58:         assert isinstance(model, TfIdfLikelihood), (
 59:             f"Expected TfIdfLikelihood, got {type(model).__name__}"
 60:         )
 61:         assert model._is_fit is True, "TF-IDF model should be fitted after construction"
 62: 
 63:     def test_likelihood_factory_tfidf_missing_corpus(
 64:         self, sample_config: dict
 65:     ) -> None:
 66:         """TF-IDF factory without corpus_texts raises ValueError."""
 67:         sample_config["likelihood"]["model"] = "tfidf"
 68:         with pytest.raises(ValueError, match="corpus_texts"):
 69:             build_likelihood_from_config(sample_config)
 70: 
 71:     def test_likelihood_factory_unknown_model(self, sample_config: dict) -> None:
 72:         """Unknown model name raises ValueError."""
 73:         sample_config["likelihood"]["model"] = "unknown_model"
 74:         with pytest.raises(ValueError, match="Unknown likelihood model"):
 75:             build_likelihood_from_config(sample_config)
 76: 
 77:     def test_likelihood_factory_sbert_name_override(
 78:         self, sample_config: dict, stub_sbert_init: None
 79:     ) -> None:
 80:         """sbert_name config key overrides default model name."""
 81:         sample_config["likelihood"]["model"] = "sbert"
 82:         sample_config["likelihood"]["sbert_name"] = "all-MiniLM-L6-v2"
 83:         model = build_likelihood_from_config(sample_config)
 84:         assert isinstance(model, SBERTLikelihood)
 85:         assert model.model_name == "all-MiniLM-L6-v2", (
 86:             f"Expected all-MiniLM-L6-v2, got {model.model_name}"
 87:         )
 88: 
 89:     def test_likelihood_factory_embedding_model_key(
 90:         self, sample_config: dict, stub_sbert_init: None
 91:     ) -> None:
 92:         """embedding_model config key works as fallback for sbert_name."""
 93:         sample_config["likelihood"]["model"] = "sbert"
 94:         sample_config["likelihood"]["embedding_model"] = "all-MiniLM-L6-v2"
 95:         # Remove sbert_name if present to test fallback
 96:         sample_config["likelihood"].pop("sbert_name", None)
 97:         model = build_likelihood_from_config(sample_config)
 98:         assert isinstance(model, SBERTLikelihood)
 99:         assert model.model_name == "all-MiniLM-L6-v2"
100: 
101: 
102: # ------------------------------------------------------------------ #
103: # Tests: make_env_from_config (CFG-02)
104: # ------------------------------------------------------------------ #
105: 
106: 
107: class TestMakeEnvFromConfig:
108:     """Tests for environment factory function."""
109: 
110:     def _make_model_and_env(
111:         self, mc_question: MCQuestion, config: dict
112:     ) -> TossupMCEnv:
113:         """Helper to create a model and env from config."""
114:         corpus = mc_question.option_profiles[:]
115:         model = TfIdfLikelihood(corpus_texts=corpus)
116:         return make_env_from_config([mc_question], model, config)
117: 
118:     def test_env_factory_creates_tossup_env(
119:         self, sample_mc_question: MCQuestion, sample_config: dict
120:     ) -> None:
121:         """Factory creates a TossupMCEnv instance."""
122:         env = self._make_model_and_env(sample_mc_question, sample_config)
123:         assert isinstance(env, TossupMCEnv), (
124:             f"Expected TossupMCEnv, got {type(env).__name__}"
125:         )
126: 
127:     def test_env_factory_config_values(
128:         self, sample_mc_question: MCQuestion, sample_config: dict
129:     ) -> None:
130:         """Factory correctly extracts config values."""
131:         env = self._make_model_and_env(sample_mc_question, sample_config)
132:         assert env.K == 4, f"Expected K=4, got {env.K}"
133:         assert env.reward_mode == "simple", (
134:             f"Expected 'simple', got '{env.reward_mode}'"
135:         )
136:         assert env.belief_mode == "from_scratch", (
137:             f"Expected 'from_scratch', got '{env.belief_mode}'"
138:         )
139:         assert env.beta == 5.0, f"Expected beta=5.0, got {env.beta}"
140: 
141:     def test_env_factory_reward_mode_override(
142:         self, sample_mc_question: MCQuestion, sample_config: dict
143:     ) -> None:
144:         """Config overrides reward mode."""
145:         sample_config["environment"]["reward"] = "human_grounded"
146:         env = self._make_model_and_env(sample_mc_question, sample_config)
147:         assert env.reward_mode == "human_grounded", (
148:             f"Expected 'human_grounded', got '{env.reward_mode}'"
149:         )
150: 
151:     def test_env_factory_beta_override(
152:         self, sample_mc_question: MCQuestion, sample_config: dict
153:     ) -> None:
154:         """Config overrides beta value."""
155:         sample_config["likelihood"]["beta"] = 10.0
156:         env = self._make_model_and_env(sample_mc_question, sample_config)
157:         assert env.beta == 10.0, f"Expected beta=10.0, got {env.beta}"
158: 
159:     def test_env_factory_wait_penalty_override(
160:         self, sample_mc_question: MCQuestion, sample_config: dict
161:     ) -> None:
162:         """Config overrides wait_penalty value."""
163:         sample_config["environment"]["wait_penalty"] = 0.05
164:         env = self._make_model_and_env(sample_mc_question, sample_config)
165:         assert env.wait_penalty == 0.05, (
166:             f"Expected wait_penalty=0.05, got {env.wait_penalty}"
167:         )
168: 
169:     def test_env_factory_reset_works(
170:         self, sample_mc_question: MCQuestion, sample_config: dict
171:     ) -> None:
172:         """Factory-created env can reset and produce valid observation."""
173:         env = self._make_model_and_env(sample_mc_question, sample_config)
174:         obs, info = env.reset()
175:         assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
176:         assert "qid" in info, "Info should contain 'qid'"
177:         assert np.all(np.isfinite(obs)), "All observations should be finite"
178: 
179:     def test_env_factory_step_works(
180:         self, sample_mc_question: MCQuestion, sample_config: dict
181:     ) -> None:
182:         """Factory-created env can step and return valid results."""
183:         env = self._make_model_and_env(sample_mc_question, sample_config)
184:         env.reset()
185:         obs, reward, terminated, truncated, info = env.step(0)
186:         assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
187:         assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
188:         assert terminated is False, "WAIT should not terminate"
189: 
190:     def test_env_factory_reward_mode_key_fallback(
191:         self, sample_mc_question: MCQuestion
192:     ) -> None:
193:         """Factory supports 'reward_mode' key (default.yaml uses this)."""
194:         config = {
195:             "data": {"K": 4},
196:             "environment": {
197:                 "reward_mode": "time_penalty",
198:                 "wait_penalty": 0.1,
199:                 "buzz_correct": 1.0,
200:                 "buzz_incorrect": -0.5,
201:             },
202:             "likelihood": {"beta": 5.0},
203:         }
204:         corpus = sample_mc_question.option_profiles[:]
205:         model = TfIdfLikelihood(corpus_texts=corpus)
206:         env = make_env_from_config([sample_mc_question], model, config)
207:         assert env.reward_mode == "time_penalty", (
208:             f"Expected 'time_penalty', got '{env.reward_mode}'"
209:         )
210: 
211:     def test_env_factory_end_mode_and_no_buzz_reward(
212:         self, sample_mc_question: MCQuestion
213:     ) -> None:
214:         """Factory reads end_mode and no_buzz_reward from config."""
215:         config = {
216:             "data": {"K": 4},
217:             "environment": {
218:                 "reward_mode": "simple",
219:                 "end_mode": "no_buzz",
220:                 "no_buzz_reward": 0.25,
221:                 "wait_penalty": 0.1,
222:                 "buzz_correct": 1.0,
223:                 "buzz_incorrect": -0.5,
224:             },
225:             "likelihood": {"beta": 5.0},
226:         }
227:         corpus = sample_mc_question.option_profiles[:]
228:         model = TfIdfLikelihood(corpus_texts=corpus)
229:         env = make_env_from_config([sample_mc_question], model, config)
230:         assert getattr(env, "end_mode") == "no_buzz"
231:         assert getattr(env, "no_buzz_reward") == 0.25
232: 
233: 
234: class TestDSPyFactoryIntegration:
235:     """Factory dispatches to DSPyLikelihood when configured."""
236: 
237:     def test_factory_returns_dspy_likelihood(self):
238:         from models.dspy_likelihood import DSPyLikelihood
239: 
240:         config = {
241:             "likelihood": {"model": "dspy"},
242:             "dspy": {"cache_dir": None, "program_fingerprint": "test"},
243:         }
244:         model = build_likelihood_from_config(config)
245:         assert isinstance(model, DSPyLikelihood)
246: 
247:     def test_default_paths_unchanged(self, sample_corpus):
248:         config = {"likelihood": {"model": "tfidf"}}
249:         model = build_likelihood_from_config(config, corpus_texts=sample_corpus)
250:         assert isinstance(model, TfIdfLikelihood)
````

## File: tests/test_ppo_buzzer.py
````python
  1: """Test suite for scripts/_common.py and agents/ppo_buzzer.py.
  2: 
  3: Covers:
  4: - AGT-01: PPOBuzzer training, save, load, episode execution
  5: - AGT-07: Shared utilities (config, JSON, MCQuestion serialization)
  6: - S_q metric support: c_trace, g_trace, entropy_trace generation
  7: 
  8: Uses TF-IDF likelihood for fast test execution (< 10 seconds total).
  9: """
 10: 
 11: from __future__ import annotations
 12: 
 13: from dataclasses import asdict
 14: from pathlib import Path
 15: 
 16: import numpy as np
 17: import pytest
 18: 
 19: from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
 20: from qb_data.mc_builder import MCQuestion
 21: from qb_env.tossup_env import TossupMCEnv
 22: from scripts._common import (
 23:     ARTIFACT_DIR,
 24:     PROJECT_ROOT,
 25:     load_config,
 26:     load_json,
 27:     mc_question_from_dict,
 28:     save_json,
 29:     to_serializable,
 30: )
 31: 
 32: 
 33: # ------------------------------------------------------------------ #
 34: # Tests: _common utilities (AGT-07)
 35: # ------------------------------------------------------------------ #
 36: 
 37: 
 38: class TestLoadConfig:
 39:     """Tests for config loading utility."""
 40: 
 41:     def test_load_config_default(self) -> None:
 42:         """load_config() without args loads default.yaml with expected keys."""
 43:         cfg = load_config()
 44:         assert isinstance(cfg, dict)
 45:         assert "data" in cfg
 46:         assert "ppo" in cfg
 47:         assert "environment" in cfg
 48:         assert "likelihood" in cfg
 49: 
 50:     def test_load_config_smoke(self) -> None:
 51:         """load_config() can load smoke.yaml with reduced settings."""
 52:         smoke_path = str(PROJECT_ROOT / "configs" / "smoke.yaml")
 53:         cfg = load_config(smoke_path)
 54:         assert cfg["data"]["max_questions"] == 50
 55:         assert cfg["ppo"]["total_timesteps"] == 3000
 56: 
 57: 
 58: class TestJsonUtilities:
 59:     """Tests for JSON save/load round-trip."""
 60: 
 61:     def test_save_load_json_roundtrip(self, tmp_path: Path) -> None:
 62:         """save_json/load_json round-trips nested dicts."""
 63:         data = {"a": 1, "b": [2, 3], "c": {"d": "hello"}}
 64:         path = tmp_path / "test.json"
 65:         save_json(path, data)
 66:         loaded = load_json(path)
 67:         assert loaded == data
 68: 
 69:     def test_save_json_creates_parent_dirs(self, tmp_path: Path) -> None:
 70:         """save_json creates missing parent directories."""
 71:         path = tmp_path / "sub" / "dir" / "test.json"
 72:         save_json(path, {"x": 1})
 73:         assert path.exists()
 74: 
 75: 
 76: class TestMCQuestionSerialization:
 77:     """Tests for MCQuestion serialization and deserialization."""
 78: 
 79:     def test_to_serializable_on_mcquestion(
 80:         self, sample_mc_question: MCQuestion
 81:     ) -> None:
 82:         """to_serializable converts MCQuestion to a dict."""
 83:         result = to_serializable(sample_mc_question)
 84:         assert isinstance(result, dict)
 85:         assert result["qid"] == "test_q1"
 86:         assert result["gold_index"] == 0
 87:         assert len(result["options"]) == 4
 88: 
 89:     def test_mc_question_roundtrip(
 90:         self, sample_mc_question: MCQuestion
 91:     ) -> None:
 92:         """MCQuestion survives serialization -> deserialization round-trip."""
 93:         serialized = to_serializable(sample_mc_question)
 94:         restored = mc_question_from_dict(serialized)
 95:         assert restored.qid == sample_mc_question.qid
 96:         assert restored.gold_index == sample_mc_question.gold_index
 97:         assert restored.options == sample_mc_question.options
 98:         assert restored.tokens == sample_mc_question.tokens
 99: 
100:     def test_mc_question_json_roundtrip(
101:         self, sample_mc_question: MCQuestion, tmp_path: Path
102:     ) -> None:
103:         """MCQuestion survives save_json -> load_json -> mc_question_from_dict."""
104:         path = tmp_path / "mc.json"
105:         save_json(path, [sample_mc_question])
106:         raw = load_json(path)
107:         restored = mc_question_from_dict(raw[0])
108:         assert restored.qid == sample_mc_question.qid
109:         assert restored.answer_primary == sample_mc_question.answer_primary
110: 
111: 
112: class TestArtifactDir:
113:     """Tests for path constants."""
114: 
115:     def test_artifact_dir_constant(self) -> None:
116:         """ARTIFACT_DIR points to project/artifacts."""
117:         assert ARTIFACT_DIR.name == "artifacts"
118:         assert ARTIFACT_DIR.parent == PROJECT_ROOT
119: 
120: 
121: # ------------------------------------------------------------------ #
122: # Tests: PPOBuzzer initialization (AGT-01)
123: # ------------------------------------------------------------------ #
124: 
125: 
126: class TestPPOBuzzerInit:
127:     """Tests for PPOBuzzer construction."""
128: 
129:     def test_ppo_buzzer_init(self, sample_tfidf_env: TossupMCEnv) -> None:
130:         """PPOBuzzer instantiates with default hyperparameters."""
131:         buzzer = PPOBuzzer(env=sample_tfidf_env)
132:         assert buzzer.model is not None
133:         assert buzzer.env is sample_tfidf_env
134: 
135:     def test_ppo_buzzer_custom_policy_kwargs(
136:         self, sample_tfidf_env: TossupMCEnv
137:     ) -> None:
138:         """PPOBuzzer accepts custom policy_kwargs."""
139:         buzzer = PPOBuzzer(
140:             env=sample_tfidf_env,
141:             policy_kwargs={"net_arch": [128, 128, 64]},
142:         )
143:         assert buzzer.model is not None
144: 
145: 
146: # ------------------------------------------------------------------ #
147: # Tests: Episode trace generation
148: # ------------------------------------------------------------------ #
149: 
150: 
151: class TestActionProbabilities:
152:     """Tests for action probability extraction."""
153: 
154:     def test_action_probabilities_shape(
155:         self, sample_tfidf_env: TossupMCEnv
156:     ) -> None:
157:         """action_probabilities returns K+1 probabilities that sum to 1."""
158:         buzzer = PPOBuzzer(env=sample_tfidf_env)
159:         obs, _ = sample_tfidf_env.reset(seed=42)
160:         probs = buzzer.action_probabilities(obs)
161:         K = sample_tfidf_env.K
162:         assert probs.shape == (K + 1,), f"Expected ({K + 1},), got {probs.shape}"
163:         assert abs(probs.sum() - 1.0) < 1e-5, f"Probabilities sum to {probs.sum()}"
164:         assert (probs >= 0).all(), "All probabilities should be non-negative"
165: 
166:     def test_c_t_computation(self, sample_tfidf_env: TossupMCEnv) -> None:
167:         """c_t returns buzz probability in [0, 1]."""
168:         buzzer = PPOBuzzer(env=sample_tfidf_env)
169:         obs, _ = sample_tfidf_env.reset(seed=42)
170:         c_val = buzzer.c_t(obs)
171:         assert 0.0 <= c_val <= 1.0, f"c_t={c_val} out of range"
172: 
173:     def test_g_t_computation(self, sample_tfidf_env: TossupMCEnv) -> None:
174:         """g_t returns correctness probability, handles near-zero c_t."""
175:         buzzer = PPOBuzzer(env=sample_tfidf_env)
176:         obs, _ = sample_tfidf_env.reset(seed=42)
177:         gold_index = sample_tfidf_env.question.gold_index
178:         g_val = buzzer.g_t(obs, gold_index)
179:         assert g_val >= 0.0, f"g_t={g_val} should be non-negative"
180:         # g_t can be > 1.0 if P(gold) > P(buzz) in early steps, but
181:         # mathematically g_t = P(gold) / c_t <= 1.0 since P(gold) <= c_t
182:         # (gold action is one of the buzz actions)
183:         assert g_val <= 1.0 + 1e-5, f"g_t={g_val} should be <= 1.0"
184: 
185: 
186: class TestRunEpisode:
187:     """Tests for full episode execution with traces."""
188: 
189:     def test_run_episode_generates_traces(
190:         self, sample_tfidf_env: TossupMCEnv
191:     ) -> None:
192:         """run_episode returns PPOEpisodeTrace with matching trace lengths."""
193:         buzzer = PPOBuzzer(env=sample_tfidf_env)
194:         trace = buzzer.run_episode(seed=42)
195: 
196:         assert isinstance(trace, PPOEpisodeTrace)
197:         assert len(trace.c_trace) == len(trace.g_trace)
198:         assert len(trace.c_trace) == len(trace.top_p_trace)
199:         assert len(trace.c_trace) == len(trace.entropy_trace)
200:         assert len(trace.c_trace) > 0, "Episode should have at least one step"
201: 
202:     def test_run_episode_trace_values(
203:         self, sample_tfidf_env: TossupMCEnv
204:     ) -> None:
205:         """Trace values are in valid ranges."""
206:         buzzer = PPOBuzzer(env=sample_tfidf_env)
207:         trace = buzzer.run_episode(seed=42)
208: 
209:         for c_val in trace.c_trace:
210:             assert 0.0 <= c_val <= 1.0, f"c_trace value {c_val} out of [0,1]"
211:         for g_val in trace.g_trace:
212:             assert g_val >= 0.0, f"g_trace value {g_val} should be non-negative"
213:         for top_p in trace.top_p_trace:
214:             assert 0.0 <= top_p <= 1.0, f"top_p_trace value {top_p} out of [0,1]"
215:         for ent in trace.entropy_trace:
216:             assert ent >= 0.0, f"entropy {ent} should be non-negative"
217: 
218:     def test_ppo_calibration_uses_top_p_trace(
219:         self, sample_tfidf_env: TossupMCEnv
220:     ) -> None:
221:         """calibration_at_buzz on PPO traces uses top_p_trace, not c_trace."""
222:         from dataclasses import asdict
223:         from evaluation.metrics import calibration_at_buzz
224: 
225:         buzzer = PPOBuzzer(env=sample_tfidf_env)
226:         trace = buzzer.run_episode(seed=42)
227:         assert len(trace.top_p_trace) > 0, "top_p_trace must be populated"
228: 
229:         cal = calibration_at_buzz([asdict(trace)])
230:         assert cal["n_calibration"] == 1.0
231:         # Confidence should be top_p_trace[buzz_step], not c_trace[buzz_step]
232:         idx = min(max(0, trace.buzz_step), len(trace.top_p_trace) - 1)
233:         expected_conf = trace.top_p_trace[idx]
234:         expected_brier = (expected_conf - (1.0 if trace.correct else 0.0)) ** 2
235:         assert abs(cal["brier"] - expected_brier) < 1e-9
236: 
237:     def test_run_episode_deterministic(
238:         self, sample_tfidf_env: TossupMCEnv
239:     ) -> None:
240:         """Deterministic episodes with same seed produce same traces."""
241:         buzzer = PPOBuzzer(env=sample_tfidf_env)
242:         trace1 = buzzer.run_episode(deterministic=True, seed=42)
243:         trace2 = buzzer.run_episode(deterministic=True, seed=42)
244: 
245:         assert trace1.buzz_step == trace2.buzz_step
246:         assert trace1.buzz_index == trace2.buzz_index
247:         np.testing.assert_allclose(trace1.c_trace, trace2.c_trace, atol=1e-6)
248: 
249:     def test_run_episode_has_qid(
250:         self, sample_tfidf_env: TossupMCEnv
251:     ) -> None:
252:         """Episode trace includes the question ID."""
253:         buzzer = PPOBuzzer(env=sample_tfidf_env)
254:         trace = buzzer.run_episode(seed=42)
255:         assert trace.qid != "", "qid should not be empty"
256: 
257:     def test_run_episode_correct_field(
258:         self, sample_tfidf_env: TossupMCEnv
259:     ) -> None:
260:         """correct field matches buzz_index vs gold_index."""
261:         buzzer = PPOBuzzer(env=sample_tfidf_env)
262:         trace = buzzer.run_episode(seed=42)
263:         assert trace.correct == (trace.buzz_index == trace.gold_index)
264: 
265:     def test_run_episode_stop_only_uses_env_chosen_idx(
266:         self, sample_tfidf_env: TossupMCEnv
267:     ) -> None:
268:         """Stop-only episodes must use the env-selected answer index."""
269:         sample_obs, _ = sample_tfidf_env.reset(seed=42)
270:         buzzer = PPOBuzzer(env=sample_tfidf_env)
271: 
272:         class FakeStopOnlyEnv:
273:             def __init__(self, obs_shape):
274:                 self.obs_shape = obs_shape
275:                 self.question = type("Question", (), {"gold_index": 2})()
276:                 self.belief = np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32)
277: 
278:             def reset(self, seed=None, options=None):
279:                 self.question = type("Question", (), {"gold_index": 2})()
280:                 self.belief = np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32)
281:                 return np.zeros(self.obs_shape, dtype=np.float32), {"qid": "stop_only_q"}
282: 
283:             def step(self, action):
284:                 assert action == 1
285:                 return (
286:                     np.zeros(self.obs_shape, dtype=np.float32),
287:                     1.0,
288:                     True,
289:                     False,
290:                     {"qid": "stop_only_q", "step_idx": 0, "chosen_idx": 2, "correct": True},
291:                 )
292: 
293:         buzzer.env = FakeStopOnlyEnv(sample_obs.shape)
294:         buzzer.action_probabilities = lambda _obs: np.array([0.1, 0.9], dtype=np.float32)
295: 
296:         trace = buzzer.run_episode(deterministic=True, seed=42)
297: 
298:         assert trace.buzz_index == 2
299:         assert trace.gold_index == 2
300:         assert trace.correct is True
301:         assert trace.g_trace == pytest.approx([0.6])
302: 
303:     def test_run_episode_no_buzz_keeps_buzz_step_unset(
304:         self, sample_mc_question: MCQuestion
305:     ) -> None:
306:         """no_buzz truncations stay distinct from voluntary buzz episodes."""
307:         from models.likelihoods import TfIdfLikelihood
308: 
309:         corpus = sample_mc_question.option_profiles[:]
310:         model = TfIdfLikelihood(corpus_texts=corpus)
311:         env = TossupMCEnv(
312:             questions=[sample_mc_question],
313:             likelihood_model=model,
314:             K=4,
315:             reward_mode="simple",
316:             end_mode="no_buzz",
317:             no_buzz_reward=0.0,
318:         )
319:         buzzer = PPOBuzzer(env=env)
320:         buzzer.action_probabilities = lambda _obs: np.array(
321:             [1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
322:         )
323: 
324:         trace = buzzer.run_episode(deterministic=True, seed=42)
325: 
326:         assert trace.buzz_step == -1
327:         assert trace.buzz_index == -1
328:         assert trace.correct is False
329: 
330: 
331: # ------------------------------------------------------------------ #
332: # Tests: Checkpoint save/load
333: # ------------------------------------------------------------------ #
334: 
335: 
336: class TestCheckpointSaveLoad:
337:     """Tests for PPOBuzzer model persistence."""
338: 
339:     def test_ppo_checkpoint_save_load(
340:         self, sample_tfidf_env: TossupMCEnv, tmp_path: Path
341:     ) -> None:
342:         """PPOBuzzer saves and loads from checkpoint."""
343:         buzzer = PPOBuzzer(env=sample_tfidf_env)
344:         save_path = tmp_path / "ppo_test"
345:         buzzer.save(save_path)
346: 
347:         # SB3 appends .zip
348:         assert (tmp_path / "ppo_test.zip").exists(), "Model file should exist"
349: 
350:         loaded = PPOBuzzer.load(save_path, env=sample_tfidf_env)
351:         assert loaded.model is not None
352: 
353:         # Verify loaded model produces valid probabilities
354:         obs, _ = sample_tfidf_env.reset(seed=42)
355:         probs = loaded.action_probabilities(obs)
356:         assert probs.shape == (sample_tfidf_env.K + 1,)
357:         assert abs(probs.sum() - 1.0) < 1e-5
358: 
359: 
360: class TestMaskablePPO:
361:     """Tests for optional MaskablePPO path."""
362: 
363:     def test_default_ppo_unchanged(self, sample_tfidf_env) -> None:
364:         buzzer = PPOBuzzer(env=sample_tfidf_env, use_maskable_ppo=False)
365:         assert not buzzer._use_maskable
366:         trace = buzzer.run_episode(seed=42)
367:         assert len(trace.c_trace) > 0
368: 
369:     def test_maskable_import_error(self, sample_tfidf_env) -> None:
370:         sb3_contrib = pytest.importorskip("sb3_contrib", reason="sb3-contrib not installed")
371:         buzzer = PPOBuzzer(env=sample_tfidf_env, use_maskable_ppo=True)
372:         assert buzzer._use_maskable
````

## File: configs/smoke.yaml
````yaml
 1: # Smoke test configuration - quick testing with reduced data
 2: # Inherits from default.yaml and overrides key settings
 3: 
 4: # Data settings for quick testing
 5: data:
 6:   csv_path: "questions.csv"
 7:   K: 4
 8:   distractor_strategy: "category_random"  # Faster than sbert_profile
 9:   train_ratio: 0.7
10:   val_ratio: 0.15
11:   test_ratio: 0.15
12:   max_questions: 50  # Use only 50 questions for smoke test
13:   shuffle_seed: 42
14: 
15: answer_profiles:
16:   max_tokens_per_profile: 500  # Reduced for speed
17:   min_questions_per_answer: 1
18:   leave_one_out: false  # Skip for smoke test
19: 
20: likelihood:
21:   model: "tfidf"  # Use TF-IDF for fastest smoke testing (<5 seconds)
22:   embedding_model: "all-MiniLM-L6-v2"
23:   beta: 5.0  # Softmax temperature for belief distribution
24:   cache_embeddings: true
25:   cache_dir: "cache/embeddings"
26:   batch_size: 4  # Smaller batch for memory
27:   max_length: 256  # Shorter sequences
28: 
29: environment:
30:   reward_mode: "time_penalty"
31:   seed: 13
32:   wait_penalty: 0.05
33:   early_buzz_penalty: 0.2
34:   buzz_correct: 1.0
35:   buzz_incorrect: -1.0
36:   max_steps: 10  # Fewer steps for quick testing
37:   opponent_buzz_model:
38:     type: "none"
39: 
40: mc_guards:
41:   alias_edit_distance_threshold: 0.2
42:   duplicate_token_overlap_threshold: 0.8
43:   max_length_ratio: 3.0
44: 
45: bayesian:  # Reduced sweep for smoke testing
46:   threshold_sweep: [0.5, 0.7, 0.9]
47:   alpha: 10.0
48: 
49: ppo:  # Reduced for smoke testing
50:   seed: 13
51:   total_timesteps: 3000
52:   learning_rate: 3e-4
53:   n_steps: 32  # Smaller rollout
54:   batch_size: 8  # Smaller batch
55:   n_epochs: 2  # Fewer epochs
56:   gamma: 0.99
57:   gae_lambda: 0.95
58:   clip_ratio: 0.2
59:   value_coef: 0.5
60:   entropy_coef: 0.01
61:   max_grad_norm: 0.5
62:   target_kl: 0.03
63:   policy_kwargs:
64:     net_arch: [32, 32]  # Smaller network
65: 
66: evaluation:
67:   metrics:
68:     - accuracy
69:     - reward
70:   compute_sq: false  # Skip expensive metrics
71:   run_choices_only: false  # Skip control experiments
72:   run_shuffle: false
73:   bootstrap_ci_samples: 0  # No bootstrap for smoke test
74:   save_predictions: false
75:   prediction_dir: "results/predictions"
76: 
77: # Supervised settings for smoke test
78: supervised:
79:   epochs: 2  # Very few epochs
80:   batch_size: 4
81:   gradient_accumulation_steps: 1  # No accumulation for speed
82:   learning_rate: 1e-4
83:   warmup_steps: 10
84:   eval_steps: 20
85:   save_steps: 100
86:   save_total_limit: 1
87:   checkpoint_dir: "checkpoints/supervised_smoke"
````

## File: scripts/build_mc_dataset.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Build multiple-choice dataset from QANTA quiz bowl questions.
  4: 
  5: This script orchestrates the complete data pipeline:
  6: 1. Load questions from CSV or HuggingFace
  7: 2. Build answer profiles from training data
  8: 3. Generate MC questions with anti-artifact guards
  9: 4. Create stratified train/val/test splits
 10: 5. Save processed datasets as JSON
 11: 
 12: Usage:
 13:     python scripts/build_mc_dataset.py
 14:     python scripts/build_mc_dataset.py --smoke  # Quick test with 50 questions in artifacts/smoke
 15:     python scripts/build_mc_dataset.py --config configs/custom.yaml
 16:     python scripts/build_mc_dataset.py --data.K=5 --data.distractor_strategy=tfidf_profile
 17: """
 18: 
 19: import argparse
 20: import json
 21: import sys
 22: import time
 23: from pathlib import Path
 24: from typing import Any, List, Optional
 25: 
 26: # Add parent directory to path for imports
 27: sys.path.insert(0, str(Path(__file__).parent.parent))
 28: 
 29: from qb_data import TossupQuestion
 30: from qb_data.answer_profiles import AnswerProfileBuilder
 31: from qb_data.config import load_config, merge_overrides, resolve_data_loading_options
 32: from qb_data.data_loader import QANTADatasetLoader
 33: from qb_data.dataset_splits import create_stratified_splits
 34: from qb_data.huggingface_loader import load_from_huggingface
 35: from qb_data.mc_builder import MCBuilder, MCQuestion
 36: from scripts._common import parse_overrides
 37: 
 38: DEFAULT_OUTPUT_DIR = Path("data/processed")
 39: SMOKE_OUTPUT_DIR = Path("artifacts/smoke")
 40: 
 41: 
 42: def resolve_output_dir(output_dir: Optional[str], smoke: bool) -> Path:
 43:     """Resolve the dataset output directory from CLI inputs."""
 44:     if output_dir is not None:
 45:         return Path(output_dir)
 46:     return SMOKE_OUTPUT_DIR if smoke else DEFAULT_OUTPUT_DIR
 47: 
 48: 
 49: def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
 50:     """Parse CLI arguments for dataset construction."""
 51:     parser = argparse.ArgumentParser(
 52:         description="Build multiple-choice dataset from QANTA questions",
 53:         formatter_class=argparse.RawDescriptionHelpFormatter,
 54:         epilog=__doc__,
 55:     )
 56: 
 57:     parser.add_argument(
 58:         '--config',
 59:         type=str,
 60:         default=None,
 61:         help=(
 62:             "Path to YAML configuration file. Defaults to configs/default.yaml, "
 63:             "or the smoke config path selected by load_config() when --smoke is set."
 64:         ),
 65:     )
 66:     parser.add_argument(
 67:         '--smoke',
 68:         action='store_true',
 69:         help='Use smoke test settings (50 questions, quick run, outputs to artifacts/smoke by default).',
 70:     )
 71:     parser.add_argument(
 72:         '--output-dir',
 73:         type=str,
 74:         default=None,
 75:         help='Directory to save processed datasets. Defaults to data/processed, or artifacts/smoke when --smoke is set.',
 76:     )
 77:     parser.add_argument(
 78:         'overrides',
 79:         nargs='*',
 80:         help='Config overrides in format: data.K=5 data.distractor_strategy=tfidf_profile',
 81:     )
 82: 
 83:     return parser.parse_args(argv)
 84: 
 85: 
 86: def save_json(path: Path, data: List[Any]) -> None:
 87:     """
 88:     Save dataclass objects to JSON file.
 89: 
 90:     Parameters
 91:     ----------
 92:     path : Path
 93:         Output file path
 94:     data : List[Any]
 95:         List of dataclass objects (TossupQuestion or MCQuestion)
 96:     """
 97:     path.parent.mkdir(parents=True, exist_ok=True)
 98: 
 99:     # Convert dataclasses to dictionaries
100:     if data and hasattr(data[0], '__dataclass_fields__'):
101:         # It's a dataclass, use asdict
102:         from dataclasses import asdict
103:         json_data = [asdict(item) for item in data]
104:     else:
105:         json_data = data
106: 
107:     with open(path, 'w', encoding='utf-8') as f:
108:         json.dump(json_data, f, indent=2, ensure_ascii=False)
109: 
110:     print(f"Saved {len(data)} items to {path}")
111: 
112: 
113: def print_statistics(
114:     train: List[MCQuestion],
115:     val: List[MCQuestion],
116:     test: List[MCQuestion],
117:     profile_builder: Optional[AnswerProfileBuilder] = None,
118:     mc_builder: Optional[MCBuilder] = None
119: ) -> None:
120:     """
121:     Print dataset statistics.
122: 
123:     Parameters
124:     ----------
125:     train : List[MCQuestion]
126:         Training split
127:     val : List[MCQuestion]
128:         Validation split
129:     test : List[MCQuestion]
130:         Test split
131:     profile_builder : Optional[AnswerProfileBuilder]
132:         Answer profile builder for profile stats
133:     mc_builder : Optional[MCBuilder]
134:         MC builder for guard rejection stats
135:     """
136:     print("\n" + "="*60)
137:     print("Dataset Construction Complete")
138:     print("="*60)
139: 
140:     # Split statistics
141:     total = len(train) + len(val) + len(test)
142:     print(f"\nTotal MC questions: {total}")
143:     print(f"  Train: {len(train)} ({100*len(train)/total:.1f}%)")
144:     print(f"  Val:   {len(val)} ({100*len(val)/total:.1f}%)")
145:     print(f"  Test:  {len(test)} ({100*len(test)/total:.1f}%)")
146: 
147:     # Category distribution
148:     def get_categories(questions):
149:         return set(q.category for q in questions if q.category)
150: 
151:     all_categories = get_categories(train) | get_categories(val) | get_categories(test)
152:     print(f"\nCategories: {len(all_categories)}")
153: 
154:     # Sample categories
155:     sample_cats = sorted(all_categories)[:5]
156:     print("Sample categories:", ", ".join(sample_cats))
157: 
158:     # Answer profile statistics
159:     if profile_builder and hasattr(profile_builder, '_grouped'):
160:         print(f"\nAnswer profiles: {len(profile_builder._grouped)}")
161:         # Get average questions per answer
162:         avg_questions = sum(len(items) for items in profile_builder._grouped.values()) / len(profile_builder._grouped)
163:         print(f"Average questions per answer: {avg_questions:.1f}")
164: 
165:     # Guard rejection statistics
166:     if mc_builder and hasattr(mc_builder, 'guard_stats'):
167:         stats = mc_builder.guard_stats
168:         if stats:
169:             print("\nGuard rejection statistics:")
170:             for guard_name, count in stats.items():
171:                 print(f"  {guard_name}: {count} rejections")
172: 
173:     # Sample MC question
174:     if train:
175:         sample = train[0]
176:         print(f"\nSample MC question:")
177:         # Get first sentence from the question
178:         first_sentence = sample.question[:100] + "..." if len(sample.question) > 100 else sample.question
179:         print(f"  Question: {first_sentence}")
180:         print(f"  Correct answer: {sample.answer_primary}")
181:         print(f"  Options: {', '.join(sample.options[:3])}...")
182:         print(f"  Category: {sample.category}")
183: 
184: 
185: def main(argv: Optional[list[str]] = None):
186:     """Main entry point for dataset construction."""
187:     args = parse_args(argv)
188: 
189:     # Start timing
190:     start_time = time.time()
191: 
192:     # Load configuration
193:     print("Loading configuration...")
194:     config = load_config(args.config, smoke=args.smoke)
195: 
196:     # Apply overrides
197:     overrides = parse_overrides(args)
198:     if overrides:
199:         print(f"Applying overrides: {overrides}")
200:         config = merge_overrides(config, overrides)
201: 
202:     # Create output directory
203:     output_dir = resolve_output_dir(args.output_dir, smoke=args.smoke)
204:     output_dir.mkdir(parents=True, exist_ok=True)
205: 
206:     # Load questions
207:     print("\nLoading questions...")
208:     questions = None
209:     data_opts = resolve_data_loading_options(config, smoke=args.smoke)
210: 
211:     # Try CSV first
212:     csv_path = data_opts.get('csv_path')
213:     if csv_path and Path(csv_path).exists():
214:         print(f"Loading from CSV: {csv_path}")
215:         loader = QANTADatasetLoader()
216:         questions = loader.load_from_csv(csv_path)
217:         print(f"Loaded {len(questions)} questions from CSV")
218: 
219:     # Fallback to HuggingFace if configured
220:     if questions is None and data_opts.get('use_huggingface'):
221:         print("CSV not found, falling back to HuggingFace")
222:         dataset_name = data_opts.get('dataset') or 'qanta-challenge/acf-co24-tossups'
223:         questions = load_from_huggingface(
224:             dataset_name,
225:             config_name=data_opts.get('dataset_config'),
226:             split=data_opts.get('split', 'eval'),
227:         )
228:         print(f"Loaded {len(questions)} questions from HuggingFace")
229: 
230:     if questions is None:
231:         raise FileNotFoundError(f"Could not load questions from {csv_path} and HuggingFace fallback not enabled")
232: 
233:     # Apply configured limit after loading
234:     max_questions = data_opts.get('max_questions')
235:     if max_questions is not None and len(questions) > int(max_questions):
236:         print(f"Limiting dataset to {int(max_questions)} questions")
237:         questions = questions[: int(max_questions)]
238: 
239:     # Build answer profiles
240:     print("\nBuilding answer profiles...")
241:     profile_builder = AnswerProfileBuilder(
242:         max_tokens_per_profile=config['answer_profiles']['max_tokens_per_profile'],
243:         min_questions_per_answer=config['answer_profiles']['min_questions_per_answer']
244:     )
245:     profile_builder.fit(questions)
246:     print(f"Built {len(profile_builder._grouped)} answer profiles")
247: 
248:     # Construct MC questions with guards
249:     print("\nConstructing MC questions...")
250:     data_cfg = config['data']
251:     mc_builder = MCBuilder(
252:         K=data_cfg['K'],
253:         strategy=data_cfg['distractor_strategy'],
254:         embedding_model=config['likelihood'].get(
255:             'sbert_name',
256:             config['likelihood'].get('embedding_model', 'all-MiniLM-L6-v2'),
257:         ),
258:         openai_model=config['likelihood'].get('openai_model', 'text-embedding-3-small'),
259:         variable_K=bool(data_cfg.get('variable_K', False)),
260:         min_K=int(data_cfg.get('min_K', 2)),
261:         max_K=int(data_cfg['max_K']) if data_cfg.get('max_K') is not None else None,
262:         **config['mc_guards']
263:     )
264: 
265:     # Track guard statistics
266:     mc_builder.guard_stats = {}
267: 
268:     mc_questions = mc_builder.build(questions, profile_builder)
269:     print(f"Generated {len(mc_questions)} MC questions")
270: 
271:     if len(mc_questions) < len(questions):
272:         print(f"Note: {len(questions) - len(mc_questions)} questions filtered by guards")
273: 
274:     # Create stratified splits
275:     print("\nCreating stratified splits...")
276:     ratios = [
277:         config['data']['train_ratio'],
278:         config['data']['val_ratio'],
279:         config['data']['test_ratio']
280:     ]
281: 
282:     train, val, test = create_stratified_splits(mc_questions, ratios=ratios)
283: 
284:     # Save datasets
285:     print("\nSaving datasets...")
286:     save_json(output_dir / "mc_dataset.json", mc_questions)
287:     save_json(output_dir / "train_dataset.json", train)
288:     save_json(output_dir / "val_dataset.json", val)
289:     save_json(output_dir / "test_dataset.json", test)
290: 
291:     # Save answer profiles for debugging
292:     if profile_builder._grouped:
293:         profiles_dict = {
294:             answer: {
295:                 'question_count': len(items),
296:                 'sample_qids': [qid for qid, _ in items[:5]]  # First 5 question IDs
297:             }
298:             for answer, items in profile_builder._grouped.items()
299:         }
300:         with open(output_dir / "answer_profiles.json", 'w') as f:
301:             json.dump(profiles_dict, f, indent=2)
302:         print(f"Saved answer profiles to {output_dir / 'answer_profiles.json'}")
303: 
304:     # Print statistics
305:     print_statistics(train, val, test, profile_builder, mc_builder)
306: 
307:     # Print timing
308:     elapsed = time.time() - start_time
309:     print(f"\nTotal time: {elapsed:.1f} seconds")
310: 
311:     if args.smoke:
312:         # Print sample MC questions for verification
313:         print("\n" + "="*60)
314:         print("Sample MC Questions (Smoke Test)")
315:         print("="*60)
316: 
317:         for i, q in enumerate(train[:3], 1):
318:             print(f"\nQuestion {i}:")
319:             # Get first clue from cumulative_prefixes if available
320:             if q.cumulative_prefixes:
321:                 first_clue = q.cumulative_prefixes[0][:100] + "..." if len(q.cumulative_prefixes[0]) > 100 else q.cumulative_prefixes[0]
322:             else:
323:                 first_clue = q.question[:100] + "..." if len(q.question) > 100 else q.question
324:             print(f"  First clue: {first_clue}")
325:             print(f"  Category: {q.category}")
326:             print(f"  Correct: {q.answer_primary}")
327:             print(f"  Options: {', '.join(q.options[:3])}...")
328: 
329:     print("\nDataset construction complete!")
330:     return 0
331: 
332: 
333: if __name__ == '__main__':
334:     sys.exit(main())
````

## File: scripts/compare_policies.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Compare T5-as-likelihood (MLP policy) vs T5-as-policy (end-to-end).
  4: 
  5: Evaluates both approaches on the same test set using the same metric
  6: functions (accuracy, S_q, ECE, Brier score, buzz position).
  7: 
  8: **Important caveats for numeric comparison:**
  9: 
 10: The two evaluation paths are *not* fully apples-to-apples:
 11: 
 12: - The MLP path uses config-driven environment settings (e.g. wait_penalty
 13:   from default.yaml or smoke.yaml).
 14: - The T5 path uses its own hardcoded reward settings (wait_penalty=0.1,
 15:   matching the T5 pipeline's default).
 16: - The MLP path builds TF-IDF from test questions + all option profiles.
 17:   The T5 path builds TF-IDF from profiles of the first 100 questions
 18:   only (lightweight env reward computation — the T5 policy does not
 19:   consume TF-IDF likelihoods).
 20: - S_q semantics differ: for MLP, c_trace is a sigmoid confidence proxy
 21:   over belief max; for T5, c_trace is the wait-head buzz probability.
 22: 
 23: These differences are inherent to the two architectures.  Accuracy and
 24: buzz-position comparisons are directly meaningful.  ECE and Brier are
 25: computed identically (both use top_p at buzz time).  S_q and reward
 26: comparisons should be interpreted qualitatively.
 27: 
 28: MLP Policy (Phase 4):
 29:     T5/TF-IDF computes likelihood scores -> belief features -> MLP
 30:     policy decides.  Uses SB3 PPO with belief-feature observations.
 31: 
 32: T5 Policy (Phase 6):
 33:     T5 encoder processes text directly -> PolicyHead decides.
 34:     Uses custom PPO with text observations via TextObservationWrapper.
 35: 
 36: Usage:
 37:     python scripts/compare_policies.py \\
 38:         --mlp-checkpoint checkpoints/ppo/best_model \\
 39:         --t5-checkpoint checkpoints/ppo_t5/best_model \\
 40:         --output results/t5_comparison.json
 41: 
 42:     python scripts/compare_policies.py \\
 43:         --t5-checkpoint checkpoints/ppo_t5/best_model \\
 44:         --t5-only
 45: """
 46: 
 47: from __future__ import annotations
 48: 
 49: import argparse
 50: import sys
 51: from pathlib import Path
 52: from typing import Any
 53: 
 54: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 55: if str(PROJECT_ROOT) not in sys.path:
 56:     sys.path.insert(0, str(PROJECT_ROOT))
 57: 
 58: import numpy as np
 59: 
 60: from evaluation.metrics import (
 61:     expected_calibration_error,
 62:     brier_score,
 63:     summarize_buzz_metrics,
 64:     system_score,
 65: )
 66: from scripts._common import ARTIFACT_DIR, load_config, load_mc_questions, save_json
 67: 
 68: 
 69: def evaluate_mlp_policy(
 70:     checkpoint_path: str,
 71:     test_questions: list,
 72:     config: dict,
 73: ) -> dict[str, Any]:
 74:     """Evaluate Phase 4 MLP policy with T5/TF-IDF likelihood on belief features.
 75: 
 76:     Loads a PPOBuzzer from an SB3 checkpoint, runs deterministic episodes
 77:     on each test question, and computes accuracy, S_q, ECE, and buzz
 78:     position metrics.
 79: 
 80:     Parameters
 81:     ----------
 82:     checkpoint_path : str
 83:         Path to SB3 PPO model checkpoint (`.zip`` file).
 84:     test_questions : list
 85:         List of MCQuestion instances to evaluate on.
 86:     config : dict
 87:         YAML config dict with environment, likelihood, and data sections.
 88: 
 89:     Returns
 90:     -------
 91:     dict[str, Any]
 92:         Evaluation results: accuracy, mean_sq, ece, brier, avg_buzz_pos,
 93:         n_questions.
 94:     """
 95:     from agents.ppo_buzzer import PPOBuzzer
 96:     from models.likelihoods import TfIdfLikelihood
 97:     from qb_env.tossup_env import make_env_from_config
 98: 
 99:     # Build likelihood model
100:     corpus = (
101:         [q.question for q in test_questions]
102:         + [p for q in test_questions for p in q.option_profiles]
103:     )
104:     likelihood_model = TfIdfLikelihood(corpus_texts=corpus)
105: 
106:     # Build environment with all test questions
107:     env = make_env_from_config(
108:         mc_questions=test_questions,
109:         likelihood_model=likelihood_model,
110:         config=config,
111:     )
112: 
113:     # Load trained agent
114:     agent = PPOBuzzer.load(checkpoint_path, env=env)
115: 
116:     # Run episodes
117:     results = []
118:     for _ in range(len(test_questions)):
119:         trace = agent.run_episode(deterministic=True)
120:         results.append(trace)
121: 
122:     # Compute metrics
123:     buzz_metrics = summarize_buzz_metrics(results)
124: 
125:     # Extract confidences and outcomes for calibration — use top_p
126:     from dataclasses import asdict
127: 
128:     rows = [asdict(r) for r in results]
129:     confidences = []
130:     outcomes = []
131:     buzz_positions = []
132:     for row in rows:
133:         top_p_trace = list(row.get("top_p_trace", []))
134:         c_trace = list(row.get("c_trace", []))
135:         conf_trace = top_p_trace if top_p_trace else c_trace
136:         buzz_step = int(row.get("buzz_step", max(0, len(conf_trace) - 1)))
137:         if conf_trace:
138:             idx = min(max(0, buzz_step), len(conf_trace) - 1)
139:             confidences.append(float(conf_trace[idx]))
140:             outcomes.append(1 if bool(row.get("correct", False)) else 0)
141:         buzz_positions.append(buzz_step)
142: 
143:     ece = expected_calibration_error(confidences, outcomes)
144:     brier = brier_score(confidences, outcomes)
145: 
146:     return {
147:         "accuracy": buzz_metrics["buzz_accuracy"],
148:         "mean_sq": buzz_metrics["mean_sq"],
149:         "ece": ece,
150:         "brier": brier,
151:         "avg_buzz_pos": float(np.mean(buzz_positions)) if buzz_positions else 0.0,
152:         "mean_reward": buzz_metrics["mean_reward_like"],
153:         "n_questions": len(test_questions),
154:     }
155: 
156: 
157: def evaluate_t5_policy(
158:     checkpoint_path: str,
159:     test_questions: list,
160:     config: dict,
161: ) -> dict[str, Any]:
162:     """Evaluate Phase 6 T5 end-to-end policy on text observations.
163: 
164:     Loads a T5PolicyModel from checkpoint, runs deterministic episodes
165:     on each test question using TextObservationWrapper, and computes the
166:     same metrics as evaluate_mlp_policy for fair comparison.
167: 
168:     Parameters
169:     ----------
170:     checkpoint_path : str
171:         Path to T5PolicyModel checkpoint directory.
172:     test_questions : list
173:         List of MCQuestion instances to evaluate on.
174:     config : dict
175:         YAML config dict.
176: 
177:     Returns
178:     -------
179:     dict[str, Any]
180:         Evaluation results: accuracy, mean_sq, ece, brier, avg_buzz_pos,
181:         n_questions.
182:     """
183:     import torch
184:     from models.t5_policy import T5PolicyModel
185:     from models.likelihoods import TfIdfLikelihood
186:     from qb_env.text_wrapper import TextObservationWrapper
187:     from qb_env.tossup_env import TossupMCEnv
188: 
189:     # Load T5 policy model
190:     model = T5PolicyModel.load_pretrained(checkpoint_path)
191:     model.eval()
192: 
193:     # Build lightweight likelihood for environment reward computation
194:     corpus = []
195:     for q in test_questions[:100]:
196:         corpus.extend(q.option_profiles)
197:     likelihood_model = TfIdfLikelihood(corpus_texts=corpus)
198: 
199:     correct_count = 0
200:     total_count = 0
201:     sq_scores = []
202:     confidences = []
203:     outcomes = []
204:     buzz_positions = []
205: 
206:     with torch.no_grad():
207:         for question in test_questions:
208:             env = TossupMCEnv(
209:                 questions=[question],
210:                 likelihood_model=likelihood_model,
211:                 K=len(question.options),
212:                 reward_mode="time_penalty",
213:                 wait_penalty=0.1,
214:                 belief_mode="from_scratch",
215:             )
216:             wrapped_env = TextObservationWrapper(env)
217: 
218:             obs, info = wrapped_env.reset()
219:             done = False
220:             c_trace = []
221:             g_trace = []
222:             top_p_trace = []
223:             episode_reward = 0.0
224:             step_count = 0
225: 
226:             while not done:
227:                 inputs = model.tokenizer(
228:                     obs,
229:                     return_tensors="pt",
230:                     padding=True,
231:                     truncation=True,
232:                     max_length=512,
233:                 )
234:                 actions, act_info = model.select_action(
235:                     inputs["input_ids"],
236:                     inputs["attention_mask"],
237:                     deterministic=True,
238:                 )
239: 
240:                 action = actions.item()
241: 
242:                 wait_probs = act_info["wait_probs"]
243:                 buzz_prob = wait_probs[0, 1].item()
244:                 c_trace.append(buzz_prob)
245: 
246:                 answer_probs = act_info["answer_probs"]
247:                 gold_prob = answer_probs[0, question.gold_index].item()
248:                 g_trace.append(gold_prob)
249: 
250:                 top_p = float(answer_probs[0].max().item())
251:                 top_p_trace.append(top_p)
252: 
253:                 obs, reward, terminated, truncated, step_info = (
254:                     wrapped_env.step(action)
255:                 )
256:                 done = terminated or truncated
257:                 episode_reward += reward
258:                 step_count += 1
259: 
260:             sq = system_score(c_trace, g_trace)
261:             sq_scores.append(sq)
262: 
263:             is_correct = step_info.get("correct", False) or step_info.get(
264:                 "forced_correct", False
265:             )
266:             if is_correct:
267:                 correct_count += 1
268:             total_count += 1
269: 
270:             # Calibration: use top_p (max answer prob) for consistency
271:             # with belief-feature agents
272:             if top_p_trace:
273:                 buzz_step = step_count - 1
274:                 confidences.append(top_p_trace[-1])
275:                 outcomes.append(1 if is_correct else 0)
276:                 buzz_positions.append(buzz_step)
277: 
278:     accuracy = correct_count / max(1, total_count)
279:     mean_sq = float(np.mean(sq_scores)) if sq_scores else 0.0
280:     ece = expected_calibration_error(confidences, outcomes)
281:     brier_val = brier_score(confidences, outcomes)
282:     avg_buzz_pos = float(np.mean(buzz_positions)) if buzz_positions else 0.0
283: 
284:     return {
285:         "accuracy": accuracy,
286:         "mean_sq": mean_sq,
287:         "ece": ece,
288:         "brier": brier_val,
289:         "avg_buzz_pos": avg_buzz_pos,
290:         "mean_reward": 0.0,  # Not tracked per-episode for T5 policy eval
291:         "n_questions": total_count,
292:     }
293: 
294: 
295: def print_comparison(
296:     mlp_results: dict[str, Any] | None,
297:     t5_results: dict[str, Any],
298:     test_size: int,
299: ) -> dict[str, Any]:
300:     """Print and return comparison summary.
301: 
302:     Parameters
303:     ----------
304:     mlp_results : dict or None
305:         MLP policy evaluation results. None if --t5-only.
306:     t5_results : dict
307:         T5 policy evaluation results.
308:     test_size : int
309:         Number of test questions evaluated.
310: 
311:     Returns
312:     -------
313:     dict[str, Any]
314:         Complete comparison dict for JSON serialization.
315:     """
316:     print("\n" + "=" * 70)
317:     print("COMPARISON RESULTS: T5-as-Likelihood vs T5-as-Policy")
318:     print("=" * 70)
319:     print(f"Test set size: {test_size}")
320:     print()
321: 
322:     if mlp_results is not None:
323:         print(f"{'Metric':<20} {'MLP (T5-likelihood)':>20} {'T5 (end-to-end)':>20} {'Difference':>15}")
324:         print("-" * 75)
325:         for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]:
326:             mlp_val = mlp_results.get(metric, 0.0)
327:             t5_val = t5_results.get(metric, 0.0)
328:             diff = t5_val - mlp_val
329:             print(f"{metric:<20} {mlp_val:>20.4f} {t5_val:>20.4f} {diff:>+15.4f}")
330:     else:
331:         print("T5 Policy (end-to-end) results:")
332:         print("-" * 40)
333:         for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]:
334:             val = t5_results.get(metric, 0.0)
335:             print(f"  {metric:<20}: {val:.4f}")
336: 
337:     # Build comparison dict
338:     comparison: dict[str, Any] = {
339:         "test_size": test_size,
340:         "t5_policy": t5_results,
341:     }
342:     if mlp_results is not None:
343:         comparison["mlp_policy"] = mlp_results
344:         comparison["difference"] = {
345:             metric: t5_results.get(metric, 0.0) - mlp_results.get(metric, 0.0)
346:             for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]
347:         }
348: 
349:     return comparison
350: 
351: 
352: def parse_compare_args() -> argparse.Namespace:
353:     """Parse comparison script arguments.
354: 
355:     Returns
356:     -------
357:     argparse.Namespace
358:         Parsed arguments.
359:     """
360:     parser = argparse.ArgumentParser(
361:         description="Compare T5-as-likelihood (MLP) vs T5-as-policy.",
362:     )
363:     parser.add_argument(
364:         "--mlp-checkpoint",
365:         type=str,
366:         default=None,
367:         help="Path to Phase 4 MLP policy checkpoint.",
368:     )
369:     parser.add_argument(
370:         "--t5-checkpoint",
371:         type=str,
372:         required=True,
373:         help="Path to Phase 6 T5 policy checkpoint.",
374:     )
375:     parser.add_argument(
376:         "--config",
377:         type=str,
378:         default=None,
379:         help="Path to YAML config file.",
380:     )
381:     parser.add_argument(
382:         "--mc-path",
383:         type=str,
384:         default=None,
385:         help="Path to MC dataset JSON file.",
386:     )
387:     parser.add_argument(
388:         "--output",
389:         type=str,
390:         default="results/t5_comparison.json",
391:         help="Path for output JSON results.",
392:     )
393:     parser.add_argument(
394:         "--smoke",
395:         action="store_true",
396:         help="Quick test with first 50 questions.",
397:     )
398:     parser.add_argument(
399:         "--t5-only",
400:         action="store_true",
401:         help="Only evaluate T5 policy (skip MLP comparison).",
402:     )
403:     return parser.parse_args()
404: 
405: 
406: def main() -> None:
407:     """Run the comparison experiment."""
408:     args = parse_compare_args()
409: 
410:     # Load config
411:     config = load_config(args.config)
412: 
413:     # Load test questions
414:     if args.mc_path:
415:         mc_path = Path(args.mc_path)
416:     else:
417:         candidates = [
418:             ARTIFACT_DIR / "main" / "mc_dataset.json",
419:             ARTIFACT_DIR / "smoke" / "mc_dataset.json",
420:             PROJECT_ROOT / "data" / "processed" / "mc_dataset.json",
421:         ]
422:         mc_path = None
423:         for candidate in candidates:
424:             if candidate.exists():
425:                 mc_path = candidate
426:                 break
427:         if mc_path is None:
428:             print("ERROR: No MC dataset found. Run build_mc_dataset.py first.")
429:             sys.exit(1)
430: 
431:     print(f"Loading questions from: {mc_path}")
432:     all_questions = load_mc_questions(mc_path)
433:     print(f"Loaded {len(all_questions)} questions")
434: 
435:     # Prefer the persisted test split if it exists alongside mc_dataset.json
436:     test_split_path = mc_path.parent / "test_dataset.json"
437:     if test_split_path.exists():
438:         test_questions = load_mc_questions(test_split_path)
439:         print(f"Using persisted test split: {len(test_questions)} questions")
440:     else:
441:         import random
442:         rng = random.Random(42)
443:         shuffled = all_questions[:]
444:         rng.shuffle(shuffled)
445:         test_start = int(len(shuffled) * 0.85)
446:         test_questions = shuffled[test_start:]
447:         print(f"No test_dataset.json found; using random 15% split: {len(test_questions)} questions")
448: 
449:     if args.smoke:
450:         test_questions = test_questions[:50]
451: 
452:     print(f"Test set: {len(test_questions)} questions")
453: 
454:     # Evaluate MLP policy (if checkpoint provided and not t5-only)
455:     mlp_results = None
456:     if args.mlp_checkpoint and not args.t5_only:
457:         print("\n" + "-" * 40)
458:         print("Evaluating MLP policy (T5-as-likelihood)...")
459:         print("-" * 40)
460:         mlp_results = evaluate_mlp_policy(
461:             args.mlp_checkpoint, test_questions, config
462:         )
463:         print(f"  Accuracy: {mlp_results['accuracy']:.4f}")
464:         print(f"  Mean S_q: {mlp_results['mean_sq']:.4f}")
465: 
466:     # Evaluate T5 policy
467:     print("\n" + "-" * 40)
468:     print("Evaluating T5 policy (end-to-end)...")
469:     print("-" * 40)
470:     t5_results = evaluate_t5_policy(
471:         args.t5_checkpoint, test_questions, config
472:     )
473:     print(f"  Accuracy: {t5_results['accuracy']:.4f}")
474:     print(f"  Mean S_q: {t5_results['mean_sq']:.4f}")
475: 
476:     # Print comparison
477:     comparison = print_comparison(mlp_results, t5_results, len(test_questions))
478: 
479:     # Save results
480:     output_path = Path(args.output)
481:     output_path.parent.mkdir(parents=True, exist_ok=True)
482:     save_json(output_path, comparison)
483:     print(f"\nResults saved to {output_path}")
484: 
485: 
486: if __name__ == "__main__":
487:     main()
````

## File: scripts/optimize_dspy.py
````python
  1: #!/usr/bin/env python3
  2: """Offline DSPy compile/optimize workflow.
  3: 
  4: Compiles a DSPy scorer program against quiz bowl training data.
  5: Does NOT integrate with PPO rollouts — this is pure offline tooling.
  6: 
  7: Usage:
  8:     python scripts/optimize_dspy.py --config configs/default.yaml
  9:     python scripts/optimize_dspy.py --config configs/default.yaml --optimizer MIPROv2
 10: """
 11: 
 12: from __future__ import annotations
 13: 
 14: import argparse
 15: import hashlib
 16: import json
 17: import sys
 18: from pathlib import Path
 19: from typing import Any
 20: 
 21: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 22: if str(PROJECT_ROOT) not in sys.path:
 23:     sys.path.insert(0, str(PROJECT_ROOT))
 24: 
 25: 
 26: def build_dspy_trainset(
 27:     mc_questions: list,
 28:     max_examples: int = 50,
 29: ) -> list[dict[str, Any]]:
 30:     """Build training examples for DSPy optimization.
 31: 
 32:     Each example contains a clue prefix, option profiles, and the gold
 33:     answer index — suitable for ``dspy.Example``.
 34: 
 35:     Parameters
 36:     ----------
 37:     mc_questions : list
 38:         MC question objects with cumulative_prefixes, option_profiles,
 39:         and gold_index.
 40:     max_examples : int
 41:         Cap on the number of examples.
 42: 
 43:     Returns
 44:     -------
 45:     list[dict]
 46:         Training examples.
 47:     """
 48:     examples = []
 49:     for q in mc_questions[:max_examples]:
 50:         mid = len(q.cumulative_prefixes) // 2
 51:         prefix = q.cumulative_prefixes[mid] if q.cumulative_prefixes else q.question
 52:         examples.append({
 53:             "clue_prefix": prefix,
 54:             "option_profiles": q.option_profiles,
 55:             "gold_index": q.gold_index,
 56:         })
 57:     return examples
 58: 
 59: 
 60: def _score_metric(example, prediction, _trace=None):
 61:     """Compare predicted scores against gold target via argmax match.
 62: 
 63:     Used as the optimization metric for DSPy ``BootstrapFewShot`` and
 64:     ``MIPROv2``.  Returns 1.0 when the argmax of the predicted scores
 65:     matches the argmax of the target scores, 0.0 otherwise.
 66:     """
 67:     try:
 68:         pred_scores = json.loads(prediction.scores)
 69:         target_scores = json.loads(example.scores)
 70:     except (json.JSONDecodeError, AttributeError):
 71:         return 0.0
 72:     if not pred_scores or not target_scores:
 73:         return 0.0
 74:     return 1.0 if (
 75:         max(range(len(pred_scores)), key=lambda i: pred_scores[i])
 76:         == max(range(len(target_scores)), key=lambda i: target_scores[i])
 77:     ) else 0.0
 78: 
 79: 
 80: def compile_dspy_scorer(
 81:     trainset: list[dict[str, Any]],
 82:     dspy_config: dict[str, Any],
 83: ) -> dict[str, Any]:
 84:     """Compile a DSPy scorer program.
 85: 
 86:     Requires the ``dspy`` package to be installed.
 87: 
 88:     Parameters
 89:     ----------
 90:     trainset : list[dict]
 91:         Training examples from ``build_dspy_trainset()``.
 92:     dspy_config : dict
 93:         DSPy configuration section from YAML.
 94: 
 95:     Returns
 96:     -------
 97:     dict
 98:         Compilation result with ``program_fingerprint`` and metadata.
 99:     """
100:     try:
101:         import dspy
102:     except ImportError as exc:
103:         raise ImportError(
104:             "DSPy optimization requires the dspy package. "
105:             "Install with: pip install -e '.[dspy]'"
106:         ) from exc
107: 
108:     lm_name = dspy_config.get("model", "openai/gpt-4o-mini")
109:     optimizer_name = dspy_config.get("optimizer", "BootstrapFewShot")
110: 
111:     lm = dspy.LM(lm_name)
112:     dspy.configure(lm=lm)
113: 
114:     class MCScoreSignature(dspy.Signature):
115:         """Score how well each answer option matches the quiz clue."""
116:         clue_prefix: str = dspy.InputField(desc="partial quiz question clue text")
117:         options: str = dspy.InputField(desc="JSON list of answer option profile texts")
118:         scores: str = dspy.OutputField(desc="JSON list of float scores, one per option")
119: 
120:     scorer = dspy.Predict(MCScoreSignature)
121: 
122:     examples = []
123:     for ex in trainset:
124:         gold = ex["gold_index"]
125:         target_scores = [0.0] * len(ex["option_profiles"])
126:         target_scores[gold] = 1.0
127:         examples.append(dspy.Example(
128:             clue_prefix=ex["clue_prefix"],
129:             options=json.dumps(ex["option_profiles"]),
130:             scores=json.dumps(target_scores),
131:         ).with_inputs("clue_prefix", "options"))
132: 
133:     if optimizer_name == "MIPROv2":
134:         optimizer = dspy.MIPROv2(metric=_score_metric)
135:     else:
136:         optimizer = dspy.BootstrapFewShot(metric=_score_metric)
137: 
138:     compiled = optimizer.compile(scorer, trainset=examples)
139: 
140:     fingerprint = hashlib.md5(
141:         json.dumps(dspy_config, sort_keys=True).encode()
142:     ).hexdigest()[:12]
143: 
144:     return {
145:         "program_fingerprint": fingerprint,
146:         "optimizer": optimizer_name,
147:         "n_examples": len(examples),
148:         "compiled_program": compiled,
149:     }
150: 
151: 
152: def main() -> None:
153:     parser = argparse.ArgumentParser(description="Offline DSPy optimization")
154:     parser.add_argument("--config", type=str, default="configs/default.yaml")
155:     parser.add_argument("--optimizer", type=str, default=None)
156:     parser.add_argument("--max-examples", type=int, default=None)
157:     args = parser.parse_args()
158: 
159:     from scripts._common import load_config, load_mc_questions, ARTIFACT_DIR
160: 
161:     config = load_config(args.config)
162:     dspy_cfg = config.get("dspy", {})
163:     if args.optimizer:
164:         dspy_cfg["optimizer"] = args.optimizer
165:     max_ex = args.max_examples or int(dspy_cfg.get("max_examples", 50))
166: 
167:     # Use the train split to avoid leaking val/test data into DSPy compilation
168:     train_path = ARTIFACT_DIR / "smoke" / "train_dataset.json"
169:     if not train_path.exists():
170:         train_path = ARTIFACT_DIR / "main" / "train_dataset.json"
171:     if not train_path.exists():
172:         # Fallback to combined dataset with warning
173:         train_path = ARTIFACT_DIR / "smoke" / "mc_dataset.json"
174:         if not train_path.exists():
175:             train_path = ARTIFACT_DIR / "main" / "mc_dataset.json"
176:         print(f"Warning: train split not found, using combined dataset: {train_path}")
177:     questions = load_mc_questions(train_path)
178:     trainset = build_dspy_trainset(questions, max_examples=max_ex)
179: 
180:     print(f"Built {len(trainset)} training examples")
181:     print(f"Compiling with {dspy_cfg.get('optimizer', 'BootstrapFewShot')}...")
182:     result = compile_dspy_scorer(trainset, dspy_cfg)
183:     print(f"Compiled. Fingerprint: {result['program_fingerprint']}")
184: 
185: 
186: if __name__ == "__main__":
187:     main()
````

## File: scripts/run_baselines.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Run non-RL baseline agents and save episode traces + summary artifacts.
  4: 
  5: Executes four baseline agent types across a threshold sweep:
  6: 1. ThresholdBuzzer -- buzzes when top belief exceeds threshold
  7: 2. SoftmaxProfileBuzzer -- softmax belief from scratch at each step
  8: 3. SequentialBayesBuzzer -- Bayesian belief update with sequential fragments
  9: 4. AlwaysBuzzFinalBuzzer -- always waits until last clue, then buzzes
 10: 
 11: Results are saved to artifacts/{smoke,main}/ as JSON files with per-episode
 12: traces and aggregated summary metrics (accuracy, S_q, ECE, Brier score).
 13: 
 14: Usage:
 15:     python scripts/run_baselines.py              # Full run (default config)
 16:     python scripts/run_baselines.py --smoke      # Quick smoke test (~50 questions)
 17:     python scripts/run_baselines.py --config configs/custom.yaml
 18:     python scripts/run_baselines.py --mc-path artifacts/main/mc_dataset.json
 19: 
 20: Ported from qb-rl reference implementation (scripts/run_baselines.py).
 21: """
 22: 
 23: from __future__ import annotations
 24: 
 25: import argparse
 26: import sys
 27: import time
 28: from dataclasses import asdict
 29: from pathlib import Path
 30: 
 31: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 32: if str(PROJECT_ROOT) not in sys.path:
 33:     sys.path.insert(0, str(PROJECT_ROOT))
 34: 
 35: from agents.bayesian_buzzer import (
 36:     precompute_sequential_beliefs,
 37:     sweep_sequential_thresholds,
 38: )
 39: from agents.threshold_buzzer import (
 40:     _always_final_from_precomputed,
 41:     _softmax_episode_from_precomputed,
 42:     precompute_beliefs,
 43:     sweep_thresholds,
 44: )
 45: from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
 46: from qb_data.config import merge_overrides
 47: from scripts._common import (
 48:     ARTIFACT_DIR,
 49:     build_likelihood_model,
 50:     load_config,
 51:     load_embedding_cache,
 52:     load_mc_questions,
 53:     parse_overrides,
 54:     save_embedding_cache,
 55:     save_json,
 56: )
 57: 
 58: 
 59: def parse_args() -> argparse.Namespace:
 60:     """Parse command-line arguments.
 61: 
 62:     Returns
 63:     -------
 64:     argparse.Namespace
 65:         Parsed arguments with config, smoke, and mc_path fields.
 66:     """
 67:     parser = argparse.ArgumentParser(description="Run non-RL baseline agents.")
 68:     parser.add_argument(
 69:         "--config",
 70:         type=str,
 71:         default=None,
 72:         help="Path to YAML config file (default: configs/default.yaml).",
 73:     )
 74:     parser.add_argument(
 75:         "--smoke",
 76:         action="store_true",
 77:         help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
 78:     )
 79:     parser.add_argument(
 80:         "--mc-path",
 81:         type=str,
 82:         default=None,
 83:         help="Optional MC dataset JSON path (overrides config-derived path).",
 84:     )
 85:     parser.add_argument(
 86:         "overrides",
 87:         nargs="*",
 88:         help="Config overrides: key=value (e.g. likelihood.model=tfidf)",
 89:     )
 90:     return parser.parse_args()
 91: 
 92: 
 93: def summarize(results: list[dict]) -> dict:
 94:     """Combine buzz metrics and calibration into a single summary dict.
 95: 
 96:     Parameters
 97:     ----------
 98:     results : list[dict]
 99:         List of episode trace dicts (from asdict(EpisodeResult)).
100: 
101:     Returns
102:     -------
103:     dict
104:         Merged summary with accuracy, S_q, ECE, Brier, etc.
105:     """
106:     return {
107:         **summarize_buzz_metrics(results),
108:         **calibration_at_buzz(results),
109:     }
110: 
111: 
112: def main() -> None:
113:     """Run all baseline agents and save artifacts."""
114:     start_time = time.time()
115: 
116:     args = parse_args()
117: 
118:     config = load_config(args.config, smoke=args.smoke)
119:     overrides = parse_overrides(args)
120:     if overrides:
121:         print(f"Applying overrides: {overrides}")
122:         config = merge_overrides(config, overrides)
123: 
124:     split = "smoke" if args.smoke else "main"
125:     out_dir = ARTIFACT_DIR / split
126: 
127:     # Determine MC dataset path
128:     mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"
129: 
130:     # Fallback: check data/processed/ if artifacts path doesn't exist
131:     if not mc_path.exists():
132:         fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
133:         if fallback.exists():
134:             print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
135:             mc_path = fallback
136: 
137:     print(f"Loading MC questions from: {mc_path}")
138:     mc_questions = load_mc_questions(mc_path)
139:     print(f"Loaded {len(mc_questions)} MC questions")
140: 
141:     # Build likelihood model
142:     print(f"Building likelihood model: {config['likelihood']['model']}")
143:     likelihood_model = build_likelihood_model(config, mc_questions)
144:     load_embedding_cache(likelihood_model, config)
145: 
146:     # Extract hyperparameters
147:     beta = float(config["likelihood"].get("beta", 5.0))
148:     alpha = float(config["bayesian"].get("alpha", 10.0))
149:     thresholds = [float(x) for x in config["bayesian"]["threshold_sweep"]]
150: 
151:     print(f"Beta: {beta}, Alpha: {alpha}")
152:     print(f"Thresholds: {thresholds}")
153: 
154:     # --- Pre-compute all embeddings once (batched) ---
155:     all_texts: list[str] = []
156:     for q in mc_questions:
157:         all_texts.extend(q.cumulative_prefixes)
158:         all_texts.extend(q.option_profiles)
159:         for step_idx in range(len(q.run_indices)):
160:             prev_idx = q.run_indices[step_idx - 1] if step_idx > 0 else -1
161:             all_texts.append(" ".join(q.tokens[prev_idx + 1 : q.run_indices[step_idx] + 1]))
162:     print(f"\nPre-computing embeddings for {len(set(all_texts)):,} unique texts...")
163:     likelihood_model.precompute_embeddings(all_texts, batch_size=64)
164:     save_embedding_cache(likelihood_model, config)
165: 
166:     # --- Pre-compute beliefs (one model pass, all steps) ---
167:     precomputed = precompute_beliefs(mc_questions, likelihood_model, beta)
168: 
169:     # --- Threshold sweep (pure numpy, instant) ---
170:     print("\nRunning ThresholdBuzzer sweep...")
171:     threshold_runs = sweep_thresholds(
172:         questions=mc_questions,
173:         likelihood_model=likelihood_model,
174:         thresholds=thresholds,
175:         beta=beta,
176:         alpha=alpha,
177:         precomputed=precomputed,
178:     )
179: 
180:     threshold_payload: dict[str, list[dict]] = {}
181:     threshold_summary: dict[str, dict] = {}
182:     for threshold, runs in threshold_runs.items():
183:         rows = [asdict(r) for r in runs]
184:         threshold_payload[str(threshold)] = rows
185:         threshold_summary[str(threshold)] = summarize(rows)
186: 
187:     # --- Softmax profile sweep (reuse from_scratch precomputed beliefs) ---
188:     print("\nRunning SoftmaxProfile sweep (precomputed)...")
189:     softmax_payload: dict[str, list[dict]] = {}
190:     softmax_summary: dict[str, dict] = {}
191:     for threshold in thresholds:
192:         results = [
193:             asdict(_softmax_episode_from_precomputed(pq, threshold, alpha))
194:             for pq in precomputed
195:         ]
196:         softmax_payload[str(threshold)] = results
197:         softmax_summary[str(threshold)] = summarize(results)
198: 
199:     # --- Sequential Bayes sweep (one belief pass, pure numpy threshold sweep) ---
200:     print("Pre-computing sequential Bayes beliefs...")
201:     seq_precomputed = precompute_sequential_beliefs(mc_questions, likelihood_model, beta)
202:     print("Running SequentialBayes sweep (precomputed)...")
203:     seq_results = sweep_sequential_thresholds(
204:         questions=mc_questions,
205:         likelihood_model=likelihood_model,
206:         thresholds=thresholds,
207:         beta=beta,
208:         alpha=alpha,
209:         precomputed=seq_precomputed,
210:     )
211:     sequential_payload: dict[str, list[dict]] = {}
212:     sequential_summary: dict[str, dict] = {}
213:     for threshold, runs in seq_results.items():
214:         rows = [asdict(r) for r in runs]
215:         sequential_payload[str(threshold)] = rows
216:         sequential_summary[str(threshold)] = summarize(rows)
217: 
218:     # --- AlwaysBuzzFinal (reuse from_scratch precomputed beliefs) ---
219:     print("Running AlwaysBuzzFinal baseline (precomputed)...")
220:     floor_runs = [asdict(_always_final_from_precomputed(pq)) for pq in precomputed]
221:     floor_summary = summarize(floor_runs)
222: 
223:     # --- Save artifacts ---
224:     print(f"\nSaving artifacts to: {out_dir}")
225:     save_json(out_dir / "baseline_threshold_runs.json", threshold_payload)
226:     save_json(out_dir / "baseline_softmax_profile_runs.json", softmax_payload)
227:     save_json(out_dir / "baseline_sequential_bayes_runs.json", sequential_payload)
228:     save_json(out_dir / "baseline_floor_runs.json", floor_runs)
229: 
230:     summary = {
231:         "threshold": threshold_summary,
232:         "softmax_profile": softmax_summary,
233:         "sequential_bayes": sequential_summary,
234:         "always_final": floor_summary,
235:     }
236:     save_json(out_dir / "baseline_summary.json", summary)
237: 
238:     elapsed = time.time() - start_time
239:     print(f"\nWrote baseline outputs to: {out_dir}")
240:     print(f"Total time: {elapsed:.1f} seconds")
241: 
242:     # Print summary highlights
243:     print("\n--- Summary ---")
244:     for agent_name, agent_summary in summary.items():
245:         if isinstance(agent_summary, dict) and "buzz_accuracy" in agent_summary:
246:             # Single-threshold agent (always_final)
247:             print(f"  {agent_name}: accuracy={agent_summary['buzz_accuracy']:.3f}, "
248:                   f"mean_sq={agent_summary.get('mean_sq', 0):.3f}")
249:         elif isinstance(agent_summary, dict):
250:             # Multi-threshold agent
251:             for thr, metrics in agent_summary.items():
252:                 if isinstance(metrics, dict) and "buzz_accuracy" in metrics:
253:                     print(f"  {agent_name}[{thr}]: accuracy={metrics['buzz_accuracy']:.3f}, "
254:                           f"mean_sq={metrics.get('mean_sq', 0):.3f}")
255: 
256: 
257: if __name__ == "__main__":
258:     main()
````

## File: tests/test_agents.py
````python
  1: """Test suite for agents/ -- baseline agent execution and episode result schemas.
  2: 
  3: Covers:
  4: - AGT-02: ThresholdBuzzer execution and buzzing logic
  5: - AGT-03: AlwaysBuzzFinalBuzzer wait-then-buzz behavior
  6: - AGT-04: SoftmaxProfileBuzzer from-scratch belief recomputation
  7: - AGT-05: SequentialBayesBuzzer incremental Bayesian updates
  8: - AGT-06: EpisodeResult and SoftmaxEpisodeResult schema validation
  9: - Threshold sweep utility tests
 10: """
 11: 
 12: from __future__ import annotations
 13: 
 14: import warnings
 15: 
 16: import numpy as np
 17: import pytest
 18: 
 19: from agents import (
 20:     AlwaysBuzzFinalBuzzer,
 21:     EpisodeResult,
 22:     SequentialBayesBuzzer,
 23:     SoftmaxEpisodeResult,
 24:     SoftmaxProfileBuzzer,
 25:     ThresholdBuzzer,
 26:     result_to_dict,
 27:     sweep_thresholds,
 28: )
 29: from agents._math import sigmoid
 30: from models.likelihoods import TfIdfLikelihood
 31: from qb_data.mc_builder import MCQuestion
 32: 
 33: 
 34: # ------------------------------------------------------------------ #
 35: # Helpers
 36: # ------------------------------------------------------------------ #
 37: 
 38: 
 39: def _make_likelihood(corpus: list[str]) -> TfIdfLikelihood:
 40:     """Create a fitted TF-IDF likelihood model from a corpus.
 41: 
 42:     Uses TF-IDF (fast) for agent logic tests so tests run quickly.
 43:     """
 44:     return TfIdfLikelihood(corpus_texts=corpus)
 45: 
 46: 
 47: class TestSigmoidMath:
 48:     """Tests for stable scalar sigmoid helper."""
 49: 
 50:     def test_sigmoid_handles_extreme_inputs_without_warning(self) -> None:
 51:         with warnings.catch_warnings():
 52:             warnings.simplefilter("error", RuntimeWarning)
 53:             assert sigmoid(1000.0) == pytest.approx(1.0)
 54:             assert sigmoid(-1000.0) == pytest.approx(0.0)
 55: 
 56: 
 57: # ------------------------------------------------------------------ #
 58: # ThresholdBuzzer tests (AGT-02)
 59: # ------------------------------------------------------------------ #
 60: 
 61: 
 62: class TestThresholdBuzzer:
 63:     """Tests for ThresholdBuzzer execution and buzzing logic."""
 64: 
 65:     def test_threshold_buzzer_executes(
 66:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
 67:     ) -> None:
 68:         """ThresholdBuzzer runs an episode without error and returns EpisodeResult."""
 69:         likelihood = _make_likelihood(sample_corpus)
 70:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
 71:         result = agent.run_episode(sample_mc_question)
 72: 
 73:         assert isinstance(result, EpisodeResult)
 74:         assert result.qid == sample_mc_question.qid
 75:         assert len(result.c_trace) > 0
 76: 
 77:     def test_threshold_buzzer_buzzes_on_threshold(
 78:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
 79:     ) -> None:
 80:         """ThresholdBuzzer buzzes when top_p >= threshold.
 81: 
 82:         With threshold=0.0, the agent should buzz immediately at step 0
 83:         because any non-negative top_p will meet the threshold.
 84:         """
 85:         likelihood = _make_likelihood(sample_corpus)
 86:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.0)
 87:         result = agent.run_episode(sample_mc_question)
 88: 
 89:         # With threshold 0.0, should buzz at step 0
 90:         assert result.buzz_step == 0, (
 91:             f"Expected buzz at step 0 with threshold=0.0, got step {result.buzz_step}"
 92:         )
 93: 
 94:     def test_threshold_buzzer_waits_on_low_confidence(
 95:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
 96:     ) -> None:
 97:         """ThresholdBuzzer waits when top_p < threshold.
 98: 
 99:         With threshold=1.0 (impossible for softmax to reach exactly 1.0 in
100:         practice), the agent should wait until the final step.
101:         """
102:         likelihood = _make_likelihood(sample_corpus)
103:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=1.0)
104:         result = agent.run_episode(sample_mc_question)
105: 
106:         # With threshold 1.0, should wait until the last step
107:         expected_final = len(sample_mc_question.cumulative_prefixes) - 1
108:         assert result.buzz_step == expected_final, (
109:             f"Expected buzz at final step {expected_final} with threshold=1.0, "
110:             f"got step {result.buzz_step}"
111:         )
112: 
113:     def test_threshold_buzzer_buzzes_at_final(
114:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
115:     ) -> None:
116:         """ThresholdBuzzer always buzzes on final step regardless of threshold.
117: 
118:         Even with threshold=1.0 (unreachable), the agent must buzz at the
119:         final step as a forced fallback.
120:         """
121:         likelihood = _make_likelihood(sample_corpus)
122:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=1.0)
123:         result = agent.run_episode(sample_mc_question)
124: 
125:         final_step = len(sample_mc_question.cumulative_prefixes) - 1
126:         assert result.buzz_step == final_step
127:         assert result.buzz_index in range(len(sample_mc_question.options))
128: 
129:     def test_threshold_buzzer_traces_valid(
130:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
131:     ) -> None:
132:         """c_trace and g_trace have correct and matching lengths.
133: 
134:         Traces should have length equal to buzz_step + 1 (one entry per
135:         step from 0 to buzz_step inclusive).
136:         """
137:         likelihood = _make_likelihood(sample_corpus)
138:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
139:         result = agent.run_episode(sample_mc_question)
140: 
141:         trace_len = result.buzz_step + 1
142:         assert len(result.c_trace) == trace_len, (
143:             f"c_trace length {len(result.c_trace)} != expected {trace_len}"
144:         )
145:         assert len(result.g_trace) == trace_len, (
146:             f"g_trace length {len(result.g_trace)} != expected {trace_len}"
147:         )
148:         assert len(result.top_p_trace) == trace_len
149:         assert len(result.entropy_trace) == trace_len
150: 
151:     def test_threshold_buzzer_confidence_proxy(
152:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
153:     ) -> None:
154:         """c_t values in [0, 1] via sigmoid transformation."""
155:         likelihood = _make_likelihood(sample_corpus)
156:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
157:         result = agent.run_episode(sample_mc_question)
158: 
159:         for c_t in result.c_trace:
160:             assert 0.0 <= c_t <= 1.0, (
161:                 f"Confidence proxy {c_t} outside [0, 1]"
162:             )
163: 
164:     def test_threshold_buzzer_custom_params(
165:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
166:     ) -> None:
167:         """ThresholdBuzzer accepts custom beta and alpha parameters."""
168:         likelihood = _make_likelihood(sample_corpus)
169:         agent = ThresholdBuzzer(
170:             likelihood_model=likelihood,
171:             threshold=0.5,
172:             beta=10.0,
173:             alpha=20.0,
174:         )
175:         assert agent.beta == 10.0
176:         assert agent.alpha == 20.0
177: 
178:         result = agent.run_episode(sample_mc_question)
179:         assert isinstance(result, EpisodeResult)
180: 
181:     def test_threshold_buzzer_confidence_proxy_stable_extremes(
182:         self, sample_corpus: list[str]
183:     ) -> None:
184:         likelihood = _make_likelihood(sample_corpus)
185:         agent = ThresholdBuzzer(
186:             likelihood_model=likelihood,
187:             threshold=-100.0,
188:             alpha=100.0,
189:         )
190: 
191:         with warnings.catch_warnings():
192:             warnings.simplefilter("error", RuntimeWarning)
193:             assert agent._confidence_proxy(1.0) == pytest.approx(1.0)
194: 
195:         agent = ThresholdBuzzer(
196:             likelihood_model=likelihood,
197:             threshold=100.0,
198:             alpha=100.0,
199:         )
200:         with warnings.catch_warnings():
201:             warnings.simplefilter("error", RuntimeWarning)
202:             assert agent._confidence_proxy(0.0) == pytest.approx(0.0)
203: 
204:     def test_threshold_buzzer_top_p_in_range(
205:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
206:     ) -> None:
207:         """top_p_trace values are valid probabilities in [0, 1]."""
208:         likelihood = _make_likelihood(sample_corpus)
209:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
210:         result = agent.run_episode(sample_mc_question)
211: 
212:         for p in result.top_p_trace:
213:             assert 0.0 <= p <= 1.0, f"top_p {p} outside [0, 1]"
214: 
215:     def test_threshold_buzzer_entropy_nonnegative(
216:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
217:     ) -> None:
218:         """Entropy values are non-negative (Shannon entropy >= 0)."""
219:         likelihood = _make_likelihood(sample_corpus)
220:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
221:         result = agent.run_episode(sample_mc_question)
222: 
223:         for h in result.entropy_trace:
224:             assert h >= 0.0, f"Entropy {h} is negative"
225: 
226: 
227: # ------------------------------------------------------------------ #
228: # AlwaysBuzzFinalBuzzer tests (AGT-03)
229: # ------------------------------------------------------------------ #
230: 
231: 
232: class TestAlwaysBuzzFinalBuzzer:
233:     """Tests for AlwaysBuzzFinalBuzzer wait-then-buzz behavior."""
234: 
235:     def test_always_buzz_final_waits(
236:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
237:     ) -> None:
238:         """All c_trace entries except the last are 0.0 (agent waits)."""
239:         likelihood = _make_likelihood(sample_corpus)
240:         agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
241:         result = agent.run_episode(sample_mc_question)
242: 
243:         # All entries except last should be 0.0
244:         for c_t in result.c_trace[:-1]:
245:             assert c_t == 0.0, f"Expected c_t=0.0 for waiting, got {c_t}"
246: 
247:     def test_always_buzz_final_buzzes_last(
248:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
249:     ) -> None:
250:         """The last c_trace entry is 1.0 (agent buzzes at final step)."""
251:         likelihood = _make_likelihood(sample_corpus)
252:         agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
253:         result = agent.run_episode(sample_mc_question)
254: 
255:         assert result.c_trace[-1] == 1.0, (
256:             f"Expected c_trace[-1]=1.0, got {result.c_trace[-1]}"
257:         )
258: 
259:     def test_always_buzz_final_computes_beliefs(
260:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
261:     ) -> None:
262:         """Beliefs are computed at each step (not skipped).
263: 
264:         All top_p_trace entries should have valid probability values,
265:         demonstrating the model computed beliefs at every step.
266:         """
267:         likelihood = _make_likelihood(sample_corpus)
268:         agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
269:         result = agent.run_episode(sample_mc_question)
270: 
271:         n_steps = len(sample_mc_question.cumulative_prefixes)
272:         assert len(result.top_p_trace) == n_steps, (
273:             f"Expected {n_steps} top_p entries, got {len(result.top_p_trace)}"
274:         )
275:         for p in result.top_p_trace:
276:             assert 0.0 <= p <= 1.0, f"top_p {p} outside [0, 1]"
277: 
278:     def test_always_buzz_final_buzz_step(
279:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
280:     ) -> None:
281:         """buzz_step equals len(cumulative_prefixes) - 1 (last step)."""
282:         likelihood = _make_likelihood(sample_corpus)
283:         agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
284:         result = agent.run_episode(sample_mc_question)
285: 
286:         expected = len(sample_mc_question.cumulative_prefixes) - 1
287:         assert result.buzz_step == expected, (
288:             f"Expected buzz_step={expected}, got {result.buzz_step}"
289:         )
290: 
291:     def test_always_buzz_final_full_trace(
292:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
293:     ) -> None:
294:         """All traces have length equal to number of cumulative prefixes."""
295:         likelihood = _make_likelihood(sample_corpus)
296:         agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
297:         result = agent.run_episode(sample_mc_question)
298: 
299:         n = len(sample_mc_question.cumulative_prefixes)
300:         assert len(result.c_trace) == n
301:         assert len(result.g_trace) == n
302:         assert len(result.top_p_trace) == n
303:         assert len(result.entropy_trace) == n
304: 
305: 
306: # ------------------------------------------------------------------ #
307: # SoftmaxProfileBuzzer tests (AGT-04)
308: # ------------------------------------------------------------------ #
309: 
310: 
311: class TestSoftmaxProfileBuzzer:
312:     """Tests for SoftmaxProfileBuzzer from-scratch belief computation."""
313: 
314:     def test_softmax_profile_executes(
315:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
316:     ) -> None:
317:         """SoftmaxProfileBuzzer runs an episode without error."""
318:         likelihood = _make_likelihood(sample_corpus)
319:         agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
320:         result = agent.run_episode(sample_mc_question)
321: 
322:         assert isinstance(result, SoftmaxEpisodeResult)
323:         assert result.qid == sample_mc_question.qid
324: 
325:     def test_softmax_profile_recomputes_belief(
326:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
327:     ) -> None:
328:         """SoftmaxProfileBuzzer calls _belief_from_scratch each step.
329: 
330:         Verifies the method exists and the agent stores beliefs, confirming
331:         from-scratch recomputation (not incremental Bayesian updates).
332:         """
333:         likelihood = _make_likelihood(sample_corpus)
334:         agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
335: 
336:         # Verify the from-scratch method exists
337:         assert hasattr(agent, "_belief_from_scratch")
338: 
339:         result = agent.run_episode(sample_mc_question)
340: 
341:         # After episode, agent should have a stored belief
342:         assert agent.belief is not None
343:         assert isinstance(agent.belief, np.ndarray)
344:         assert agent.belief.shape == (len(sample_mc_question.options),)
345: 
346:     def test_softmax_profile_result_schema(
347:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
348:     ) -> None:
349:         """SoftmaxProfileBuzzer returns SoftmaxEpisodeResult, not EpisodeResult."""
350:         likelihood = _make_likelihood(sample_corpus)
351:         agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
352:         result = agent.run_episode(sample_mc_question)
353: 
354:         assert isinstance(result, SoftmaxEpisodeResult)
355:         # SoftmaxEpisodeResult should NOT be an EpisodeResult (different dataclass)
356:         assert not isinstance(result, EpisodeResult)
357: 
358:     def test_softmax_profile_confidence_proxy(
359:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
360:     ) -> None:
361:         """SoftmaxProfileBuzzer c_t values in [0, 1] via sigmoid."""
362:         likelihood = _make_likelihood(sample_corpus)
363:         agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
364:         result = agent.run_episode(sample_mc_question)
365: 
366:         for c_t in result.c_trace:
367:             assert 0.0 <= c_t <= 1.0, f"c_t {c_t} outside [0, 1]"
368: 
369:     def test_softmax_profile_threshold_behavior(
370:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
371:     ) -> None:
372:         """SoftmaxProfileBuzzer respects threshold for buzzing."""
373:         likelihood = _make_likelihood(sample_corpus)
374: 
375:         # With threshold 0.0, should buzz immediately
376:         agent_low = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.0)
377:         result_low = agent_low.run_episode(sample_mc_question)
378:         assert result_low.buzz_step == 0
379: 
380:         # With threshold 1.0, should wait until the end
381:         agent_high = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=1.0)
382:         result_high = agent_high.run_episode(sample_mc_question)
383:         assert result_high.buzz_step == len(sample_mc_question.cumulative_prefixes) - 1
384: 
385:     def test_softmax_profile_confidence_proxy_stable_extremes(
386:         self, sample_corpus: list[str]
387:     ) -> None:
388:         likelihood = _make_likelihood(sample_corpus)
389:         agent = SoftmaxProfileBuzzer(
390:             likelihood_model=likelihood,
391:             threshold=-100.0,
392:             alpha=100.0,
393:         )
394: 
395:         with warnings.catch_warnings():
396:             warnings.simplefilter("error", RuntimeWarning)
397:             assert agent.confidence_proxy(1.0) == pytest.approx(1.0)
398: 
399:         agent = SoftmaxProfileBuzzer(
400:             likelihood_model=likelihood,
401:             threshold=100.0,
402:             alpha=100.0,
403:         )
404:         with warnings.catch_warnings():
405:             warnings.simplefilter("error", RuntimeWarning)
406:             assert agent.confidence_proxy(0.0) == pytest.approx(0.0)
407: 
408: 
409: # ------------------------------------------------------------------ #
410: # SequentialBayesBuzzer tests (AGT-05)
411: # ------------------------------------------------------------------ #
412: 
413: 
414: class TestSequentialBayesBuzzer:
415:     """Tests for SequentialBayesBuzzer incremental Bayesian update."""
416: 
417:     def test_sequential_bayes_executes(
418:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
419:     ) -> None:
420:         """SequentialBayesBuzzer runs an episode without error."""
421:         likelihood = _make_likelihood(sample_corpus)
422:         agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
423:         result = agent.run_episode(sample_mc_question)
424: 
425:         assert isinstance(result, SoftmaxEpisodeResult)
426:         assert result.qid == sample_mc_question.qid
427: 
428:     def test_sequential_bayes_uses_run_indices(
429:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
430:     ) -> None:
431:         """SequentialBayesBuzzer requires question.run_indices field.
432: 
433:         The agent iterates over run_indices to extract token fragments,
434:         not over cumulative_prefixes. The number of trace entries should
435:         match the number of run_indices steps processed.
436:         """
437:         likelihood = _make_likelihood(sample_corpus)
438:         agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
439:         result = agent.run_episode(sample_mc_question)
440: 
441:         # Trace length should be <= len(run_indices)
442:         assert len(result.c_trace) <= len(sample_mc_question.run_indices), (
443:             f"Trace length {len(result.c_trace)} > run_indices length "
444:             f"{len(sample_mc_question.run_indices)}"
445:         )
446: 
447:     def test_sequential_bayes_bayesian_update(
448:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
449:     ) -> None:
450:         """Belief is posterior proportional to prior * likelihood.
451: 
452:         Verify the _step_update method produces valid posterior:
453:         all entries >= 0 and sum to 1.
454:         """
455:         likelihood = _make_likelihood(sample_corpus)
456:         agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
457: 
458:         K = len(sample_mc_question.options)
459:         prior = np.ones(K, dtype=np.float32) / K
460:         fragment = "first president"
461:         profiles = sample_mc_question.option_profiles
462: 
463:         posterior = agent._step_update(prior, fragment, profiles)
464: 
465:         assert posterior.shape == (K,), f"Expected shape ({K},), got {posterior.shape}"
466:         assert all(posterior >= 0), "Posterior has negative entries"
467:         np.testing.assert_almost_equal(
468:             posterior.sum(), 1.0, decimal=5,
469:             err_msg="Posterior should sum to 1.0",
470:         )
471: 
472:     def test_sequential_bayes_result_schema(
473:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
474:     ) -> None:
475:         """SequentialBayesBuzzer returns SoftmaxEpisodeResult."""
476:         likelihood = _make_likelihood(sample_corpus)
477:         agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
478:         result = agent.run_episode(sample_mc_question)
479: 
480:         assert isinstance(result, SoftmaxEpisodeResult)
481:         assert not isinstance(result, EpisodeResult)
482: 
483:     def test_sequential_bayes_fragments(
484:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
485:     ) -> None:
486:         """SequentialBayesBuzzer processes token fragments, not full prefixes.
487: 
488:         With threshold 1.0 (never buzzes early), all run_indices should be
489:         processed, producing traces of length len(run_indices).
490:         """
491:         likelihood = _make_likelihood(sample_corpus)
492:         agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=1.0)
493:         result = agent.run_episode(sample_mc_question)
494: 
495:         n_steps = len(sample_mc_question.run_indices)
496:         assert len(result.c_trace) == n_steps, (
497:             f"Expected {n_steps} trace entries, got {len(result.c_trace)}"
498:         )
499: 
500: 
501: # ------------------------------------------------------------------ #
502: # Episode result schema tests (AGT-06)
503: # ------------------------------------------------------------------ #
504: 
505: 
506: class TestEpisodeResultSchema:
507:     """Tests for EpisodeResult and SoftmaxEpisodeResult dataclass schemas."""
508: 
509:     def test_episode_result_fields(self) -> None:
510:         """EpisodeResult has all required fields."""
511:         result = EpisodeResult(
512:             qid="test_q",
513:             buzz_step=3,
514:             buzz_index=1,
515:             gold_index=0,
516:             correct=False,
517:             reward_like=-0.5,
518:             c_trace=[0.1, 0.2, 0.3, 0.4],
519:             g_trace=[0.0, 0.0, 0.0, 1.0],
520:             top_p_trace=[0.3, 0.4, 0.5, 0.6],
521:             entropy_trace=[1.4, 1.2, 1.0, 0.8],
522:         )
523:         assert result.qid == "test_q"
524:         assert result.buzz_step == 3
525:         assert result.buzz_index == 1
526:         assert result.gold_index == 0
527:         assert result.correct is False
528:         assert result.reward_like == -0.5
529: 
530:     def test_softmax_episode_result_fields(self) -> None:
531:         """SoftmaxEpisodeResult has all required fields."""
532:         result = SoftmaxEpisodeResult(
533:             qid="test_q",
534:             buzz_step=2,
535:             buzz_index=0,
536:             gold_index=0,
537:             correct=True,
538:             c_trace=[0.1, 0.5, 0.9],
539:             g_trace=[1.0, 1.0, 1.0],
540:             top_p_trace=[0.4, 0.6, 0.9],
541:             entropy_trace=[1.2, 0.8, 0.3],
542:         )
543:         assert result.qid == "test_q"
544:         assert result.buzz_step == 2
545:         assert result.correct is True
546: 
547:     def test_traces_same_length(
548:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
549:     ) -> None:
550:         """len(c_trace) == len(g_trace) for all agents."""
551:         likelihood = _make_likelihood(sample_corpus)
552: 
553:         agents = [
554:             ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
555:             AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
556:             SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
557:             SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
558:         ]
559: 
560:         for agent in agents:
561:             result = agent.run_episode(sample_mc_question)
562:             agent_name = type(agent).__name__
563:             assert len(result.c_trace) == len(result.g_trace), (
564:                 f"{agent_name}: c_trace ({len(result.c_trace)}) != "
565:                 f"g_trace ({len(result.g_trace)})"
566:             )
567:             assert len(result.c_trace) == len(result.top_p_trace), (
568:                 f"{agent_name}: c_trace ({len(result.c_trace)}) != "
569:                 f"top_p_trace ({len(result.top_p_trace)})"
570:             )
571:             assert len(result.c_trace) == len(result.entropy_trace), (
572:                 f"{agent_name}: c_trace ({len(result.c_trace)}) != "
573:                 f"entropy_trace ({len(result.entropy_trace)})"
574:             )
575: 
576:     def test_g_trace_binary(
577:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
578:     ) -> None:
579:         """g_trace values are 0.0 or 1.0 (correctness is binary)."""
580:         likelihood = _make_likelihood(sample_corpus)
581: 
582:         agents = [
583:             ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
584:             AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
585:             SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
586:             SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
587:         ]
588: 
589:         for agent in agents:
590:             result = agent.run_episode(sample_mc_question)
591:             agent_name = type(agent).__name__
592:             for g_t in result.g_trace:
593:                 assert g_t in (0.0, 1.0), (
594:                     f"{agent_name}: g_t={g_t} not in {{0.0, 1.0}}"
595:                 )
596: 
597:     def test_buzz_index_valid(
598:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
599:     ) -> None:
600:         """buzz_index in range(K) where K = len(options)."""
601:         likelihood = _make_likelihood(sample_corpus)
602:         K = len(sample_mc_question.options)
603: 
604:         agents = [
605:             ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
606:             AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
607:             SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
608:             SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
609:         ]
610: 
611:         for agent in agents:
612:             result = agent.run_episode(sample_mc_question)
613:             agent_name = type(agent).__name__
614:             assert 0 <= result.buzz_index < K, (
615:                 f"{agent_name}: buzz_index={result.buzz_index} not in [0, {K})"
616:             )
617: 
618:     def test_result_to_dict(
619:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
620:     ) -> None:
621:         """result_to_dict() converts EpisodeResult to dict."""
622:         likelihood = _make_likelihood(sample_corpus)
623:         agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
624:         result = agent.run_episode(sample_mc_question)
625: 
626:         d = result_to_dict(result)
627:         assert isinstance(d, dict)
628:         assert d["qid"] == sample_mc_question.qid
629:         assert "buzz_step" in d
630:         assert "buzz_index" in d
631:         assert "gold_index" in d
632:         assert "correct" in d
633:         assert "reward_like" in d
634:         assert "c_trace" in d
635:         assert "g_trace" in d
636:         assert isinstance(d["c_trace"], list)
637: 
638: 
639: # ------------------------------------------------------------------ #
640: # Threshold sweep utility tests
641: # ------------------------------------------------------------------ #
642: 
643: 
644: class TestSweepThresholds:
645:     """Tests for sweep_thresholds utility function."""
646: 
647:     def test_sweep_thresholds_runs(
648:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
649:     ) -> None:
650:         """sweep_thresholds() returns dict[float, list[EpisodeResult]]."""
651:         likelihood = _make_likelihood(sample_corpus)
652:         results = sweep_thresholds(
653:             questions=[sample_mc_question],
654:             likelihood_model=likelihood,
655:             thresholds=[0.7],
656:         )
657: 
658:         assert isinstance(results, dict)
659:         assert 0.7 in results
660:         assert len(results[0.7]) == 1
661:         assert isinstance(results[0.7][0], EpisodeResult)
662: 
663:     def test_sweep_thresholds_multiple_values(
664:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
665:     ) -> None:
666:         """Sweeps over [0.6, 0.7, 0.8, 0.9] and returns results for each."""
667:         likelihood = _make_likelihood(sample_corpus)
668:         thresholds = [0.6, 0.7, 0.8, 0.9]
669:         results = sweep_thresholds(
670:             questions=[sample_mc_question],
671:             likelihood_model=likelihood,
672:             thresholds=thresholds,
673:         )
674: 
675:         assert len(results) == len(thresholds)
676:         for thresh in thresholds:
677:             assert thresh in results, f"Missing results for threshold {thresh}"
678:             assert len(results[thresh]) == 1
679: 
680:     def test_sweep_thresholds_monotonic_buzz_step(
681:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
682:     ) -> None:
683:         """Higher thresholds should produce later or equal buzz steps.
684: 
685:         A higher threshold means the agent needs more confidence to buzz,
686:         so it should wait at least as long as with a lower threshold.
687:         """
688:         likelihood = _make_likelihood(sample_corpus)
689:         thresholds = [0.3, 0.5, 0.7, 0.9]
690:         results = sweep_thresholds(
691:             questions=[sample_mc_question],
692:             likelihood_model=likelihood,
693:             thresholds=thresholds,
694:         )
695: 
696:         buzz_steps = [results[t][0].buzz_step for t in thresholds]
697:         for i in range(len(buzz_steps) - 1):
698:             assert buzz_steps[i] <= buzz_steps[i + 1], (
699:                 f"Buzz step not monotonic: threshold {thresholds[i]} "
700:                 f"(step {buzz_steps[i]}) > threshold {thresholds[i+1]} "
701:                 f"(step {buzz_steps[i+1]})"
702:             )
703: 
704: 
705: # ------------------------------------------------------------------ #
706: # Precomputed equivalence tests
707: # ------------------------------------------------------------------ #
708: 
709: 
710: class TestPrecomputedEquivalence:
711:     """Prove precomputed-path functions are numerically identical to live agents."""
712: 
713:     def test_softmax_precomputed_matches_live(
714:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
715:     ) -> None:
716:         """_softmax_episode_from_precomputed matches SoftmaxProfileBuzzer.run_episode."""
717:         from agents.threshold_buzzer import (
718:             _softmax_episode_from_precomputed,
719:             precompute_beliefs,
720:         )
721: 
722:         likelihood = _make_likelihood(sample_corpus)
723:         threshold, beta, alpha = 0.7, 5.0, 10.0
724: 
725:         # Live agent
726:         agent = SoftmaxProfileBuzzer(
727:             likelihood_model=likelihood, threshold=threshold, beta=beta, alpha=alpha
728:         )
729:         live = agent.run_episode(sample_mc_question)
730: 
731:         # Precomputed path
732:         pqs = precompute_beliefs([sample_mc_question], likelihood, beta)
733:         pre = _softmax_episode_from_precomputed(pqs[0], threshold, alpha)
734: 
735:         assert pre.buzz_step == live.buzz_step
736:         assert pre.buzz_index == live.buzz_index
737:         assert pre.correct == live.correct
738:         np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
739:         np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
740:         np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
741:         np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)
742: 
743:     def test_always_final_precomputed_matches_live(
744:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
745:     ) -> None:
746:         """_always_final_from_precomputed matches AlwaysBuzzFinalBuzzer.run_episode."""
747:         from agents.threshold_buzzer import (
748:             _always_final_from_precomputed,
749:             precompute_beliefs,
750:         )
751: 
752:         likelihood = _make_likelihood(sample_corpus)
753:         beta = 5.0
754: 
755:         # Live agent
756:         agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood, beta=beta)
757:         live = agent.run_episode(sample_mc_question)
758: 
759:         # Precomputed path
760:         pqs = precompute_beliefs([sample_mc_question], likelihood, beta)
761:         pre = _always_final_from_precomputed(pqs[0])
762: 
763:         assert pre.buzz_step == live.buzz_step
764:         assert pre.buzz_index == live.buzz_index
765:         assert pre.correct == live.correct
766:         assert pre.reward_like == live.reward_like
767:         np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
768:         np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
769:         np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
770:         np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)
771: 
772:     def test_sequential_precomputed_matches_live(
773:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
774:     ) -> None:
775:         """_sequential_episode_from_precomputed matches SequentialBayesBuzzer.run_episode."""
776:         from agents.bayesian_buzzer import (
777:             _sequential_episode_from_precomputed,
778:             precompute_sequential_beliefs,
779:         )
780: 
781:         likelihood = _make_likelihood(sample_corpus)
782:         threshold, beta, alpha = 0.7, 5.0, 10.0
783: 
784:         # Live agent
785:         agent = SequentialBayesBuzzer(
786:             likelihood_model=likelihood, threshold=threshold, beta=beta, alpha=alpha
787:         )
788:         live = agent.run_episode(sample_mc_question)
789: 
790:         # Precomputed path
791:         pqs = precompute_sequential_beliefs([sample_mc_question], likelihood, beta)
792:         pre = _sequential_episode_from_precomputed(pqs[0], threshold, alpha)
793: 
794:         assert pre.buzz_step == live.buzz_step
795:         assert pre.buzz_index == live.buzz_index
796:         assert pre.correct == live.correct
797:         np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
798:         np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
799:         np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
800:         np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)
801: 
802:     def test_sweep_sequential_matches_per_threshold(
803:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
804:     ) -> None:
805:         """sweep_sequential_thresholds matches per-threshold SequentialBayesBuzzer."""
806:         from agents.bayesian_buzzer import sweep_sequential_thresholds
807: 
808:         likelihood = _make_likelihood(sample_corpus)
809:         thresholds = [0.5, 0.7, 0.9]
810:         beta, alpha = 5.0, 10.0
811: 
812:         # Sweep
813:         sweep = sweep_sequential_thresholds(
814:             questions=[sample_mc_question],
815:             likelihood_model=likelihood,
816:             thresholds=thresholds,
817:             beta=beta,
818:             alpha=alpha,
819:         )
820: 
821:         # Per-threshold live agents
822:         for threshold in thresholds:
823:             agent = SequentialBayesBuzzer(
824:                 likelihood_model=likelihood,
825:                 threshold=threshold,
826:                 beta=beta,
827:                 alpha=alpha,
828:             )
829:             live = agent.run_episode(sample_mc_question)
830:             pre = sweep[float(threshold)][0]
831: 
832:             assert pre.buzz_step == live.buzz_step, (
833:                 f"threshold={threshold}: buzz_step {pre.buzz_step} != {live.buzz_step}"
834:             )
835:             assert pre.buzz_index == live.buzz_index
836:             assert pre.correct == live.correct
837:             np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
838:             np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
839:             np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
840:             np.testing.assert_array_almost_equal(
841:                 pre.entropy_trace, live.entropy_trace
842:             )
843: 
844: 
845: # ------------------------------------------------------------------ #
846: # Shuffle precomputed equivalence tests
847: # ------------------------------------------------------------------ #
848: 
849: 
850: class TestShufflePrecomputedEquivalence:
851:     """Prove precomputed shuffle control matches live rescore shuffle control."""
852: 
853:     def test_shuffle_precomputed_matches_rescore(
854:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
855:     ) -> None:
856:         """Precomputed shuffle control matches live rescore shuffle control."""
857:         from dataclasses import asdict
858: 
859:         from agents.threshold_buzzer import precompute_beliefs
860:         from evaluation.controls import (
861:             run_shuffle_control,
862:             run_shuffle_control_precomputed,
863:         )
864:         from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
865: 
866:         likelihood = _make_likelihood(sample_corpus)
867:         threshold, beta, alpha = 0.7, 5.0, 10.0
868:         questions = [sample_mc_question]
869: 
870:         # Live rescore path
871:         def evaluator(qset):
872:             agent = SoftmaxProfileBuzzer(
873:                 likelihood_model=likelihood,
874:                 threshold=threshold,
875:                 beta=beta,
876:                 alpha=alpha,
877:             )
878:             runs = [asdict(agent.run_episode(q)) for q in qset]
879:             summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
880:             summary["runs"] = runs
881:             return summary
882: 
883:         live_result = run_shuffle_control(questions, evaluator=evaluator, random_seed=13)
884: 
885:         # Precomputed path
886:         precomputed = precompute_beliefs(questions, likelihood, beta)
887:         pre_result = run_shuffle_control_precomputed(
888:             precomputed, threshold, alpha, random_seed=13
889:         )
890: 
891:         # Compare summary metrics
892:         assert live_result["mean_sq"] == pytest.approx(pre_result["mean_sq"])
893:         assert live_result["buzz_accuracy"] == pytest.approx(pre_result["buzz_accuracy"])
894: 
895:         # Compare per-run results
896:         for live_run, pre_run in zip(live_result["runs"], pre_result["runs"]):
897:             assert live_run["buzz_step"] == pre_run["buzz_step"]
898:             assert live_run["buzz_index"] == pre_run["buzz_index"]
899:             assert live_run["correct"] == pre_run["correct"]
900:             np.testing.assert_array_almost_equal(
901:                 live_run["c_trace"], pre_run["c_trace"]
902:             )
903:             np.testing.assert_array_almost_equal(
904:                 live_run["g_trace"], pre_run["g_trace"]
905:             )
906:             np.testing.assert_array_almost_equal(
907:                 live_run["top_p_trace"], pre_run["top_p_trace"]
908:             )
909:             np.testing.assert_array_almost_equal(
910:                 live_run["entropy_trace"], pre_run["entropy_trace"]
911:             )
912: 
913:     def test_permutation_consistency(
914:         self, sample_mc_question: MCQuestion, sample_corpus: list[str]
915:     ) -> None:
916:         """Permutation applied to beliefs matches permutation applied to gold_index."""
917:         import random as random_mod
918: 
919:         from agents.threshold_buzzer import _PrecomputedQuestion, precompute_beliefs
920:         from evaluation.controls import shuffled_option_copy
921: 
922:         likelihood = _make_likelihood(sample_corpus)
923:         beta = 5.0
924:         questions = [sample_mc_question]
925:         precomputed = precompute_beliefs(questions, likelihood, beta)
926: 
927:         # Reproduce the permutation that shuffled_option_copy would use
928:         rng_live = random_mod.Random(13)
929:         shuffled_q = shuffled_option_copy(sample_mc_question, rng_live)
930: 
931:         # Reproduce the same permutation for precomputed
932:         rng_pre = random_mod.Random(13)
933:         pq = precomputed[0]
934:         perm = list(range(pq.num_options))
935:         rng_pre.shuffle(perm)
936:         new_gold = perm.index(pq.gold_index)
937: 
938:         # The gold index should match
939:         assert new_gold == shuffled_q.gold_index
940: 
941: 
942: class TestBaselineAgentsVariableK:
943:     """Baseline agents work on non-K=4 questions (K-agnostic check)."""
944: 
945:     def test_threshold_buzzer_k3(self, sample_corpus):
946:         from agents.threshold_buzzer import ThresholdBuzzer
947:         from dataclasses import replace
948:         from models.likelihoods import TfIdfLikelihood
949:         from tests.conftest import sample_mc_question as _  # reuse fixture pattern
950: 
951:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
952:         q4 = MCQuestion(
953:             qid="q_k3",
954:             question="Who was the first president?",
955:             tokens=["Who", "was", "the", "first", "president"],
956:             answer_primary="George Washington",
957:             clean_answers=["George Washington"],
958:             run_indices=[1, 3, 4],
959:             human_buzz_positions=[],
960:             category="History",
961:             cumulative_prefixes=["Who was", "Who was the first", "Who was the first president"],
962:             options=["George Washington", "Thomas Jefferson", "John Adams"],
963:             gold_index=0,
964:             option_profiles=[
965:                 "George Washington first president",
966:                 "Thomas Jefferson third president",
967:                 "John Adams second president",
968:             ],
969:             option_answer_primary=["George Washington", "Thomas Jefferson", "John Adams"],
970:             distractor_strategy="test",
971:         )
972:         buzzer = ThresholdBuzzer(
973:             likelihood_model=model, threshold=0.5, beta=5.0, alpha=10.0,
974:         )
975:         result = buzzer.run_episode(q4)
976:         assert len(result.c_trace) > 0
977:         assert 0 <= result.buzz_index < 3
````

## File: tests/test_dspy_optimize.py
````python
 1: """Tests for scripts/optimize_dspy.py — offline DSPy compilation."""
 2: 
 3: from __future__ import annotations
 4: 
 5: import pytest
 6: 
 7: from scripts.optimize_dspy import build_dspy_trainset, _score_metric
 8: 
 9: 
10: def _make_mc_question():
11:     from qb_data.mc_builder import MCQuestion
12: 
13:     return MCQuestion(
14:         qid="q1",
15:         question="Who was the first president?",
16:         tokens=["Who", "was", "the", "first", "president"],
17:         answer_primary="George Washington",
18:         clean_answers=["George Washington"],
19:         run_indices=[1, 3, 4],
20:         human_buzz_positions=[],
21:         category="History",
22:         cumulative_prefixes=["Who was", "Who was the first", "Who was the first president"],
23:         options=["George Washington", "Thomas Jefferson"],
24:         gold_index=0,
25:         option_profiles=["Washington profile", "Jefferson profile"],
26:         option_answer_primary=["George Washington", "Thomas Jefferson"],
27:         distractor_strategy="test",
28:     )
29: 
30: 
31: class TestBuildDspyTrainset:
32:     def test_trainset_structure(self) -> None:
33:         mc = [_make_mc_question()]
34:         trainset = build_dspy_trainset(mc, max_examples=10)
35:         assert len(trainset) == 1
36:         ex = trainset[0]
37:         assert "clue_prefix" in ex
38:         assert "option_profiles" in ex
39:         assert "gold_index" in ex
40: 
41:     def test_trainset_caps_at_max(self) -> None:
42:         mc = [_make_mc_question()] * 100
43:         trainset = build_dspy_trainset(mc, max_examples=5)
44:         assert len(trainset) == 5
45: 
46:     def test_trainset_empty(self) -> None:
47:         assert build_dspy_trainset([], max_examples=10) == []
48: 
49: 
50: class TestCompileDspyScorer:
51:     def test_compile_requires_dspy(self) -> None:
52:         pytest.importorskip("dspy", reason="dspy not installed")
53:         from scripts.optimize_dspy import compile_dspy_scorer
54:         assert callable(compile_dspy_scorer)
55: 
56:     def test_score_metric_logic(self) -> None:
57:         """The _score_metric used by compile_dspy_scorer is argmax-based."""
58:         import json
59:         from unittest.mock import MagicMock
60: 
61:         example = MagicMock()
62:         example.scores = json.dumps([0.0, 1.0, 0.0])
63:         pred_correct = MagicMock()
64:         pred_correct.scores = json.dumps([0.1, 0.8, 0.1])
65:         pred_wrong = MagicMock()
66:         pred_wrong.scores = json.dumps([0.9, 0.05, 0.05])
67:         pred_malformed = MagicMock()
68:         pred_malformed.scores = "not json"
69: 
70:         assert _score_metric(example, pred_correct) == 1.0
71:         assert _score_metric(example, pred_wrong) == 0.0
72:         assert _score_metric(example, pred_malformed) == 0.0
73: 
74:     def test_trainset_uses_mid_prefix(self) -> None:
75:         """build_dspy_trainset picks a mid-point cumulative prefix."""
76:         mc = [_make_mc_question()]
77:         trainset = build_dspy_trainset(mc, max_examples=1)
78:         ex = trainset[0]
79:         # The question has 3 prefixes; mid = 3//2 = 1
80:         assert ex["clue_prefix"] == "Who was the first"
````

## File: tests/test_likelihoods.py
````python
  1: """Test suite for models/likelihoods.py — likelihood model interface and implementations.
  2: 
  3: Covers:
  4: - LIK-01: LikelihoodModel ABC contract
  5: - LIK-02: TfIdfLikelihood with corpus fitting and cosine scoring
  6: - LIK-03: SBERTLikelihood with semantic embeddings and caching
  7: - LIK-04: T5Likelihood semantic scoring and embedding shape
  8: - LIK-05: T5 embedding cache reuse and factory construction
  9: """
 10: 
 11: from __future__ import annotations
 12: 
 13: from pathlib import Path
 14: 
 15: import numpy as np
 16: import pytest
 17: 
 18: from models.likelihoods import (
 19:     LikelihoodModel,
 20:     SBERTLikelihood,
 21:     TfIdfLikelihood,
 22: )
 23: 
 24: 
 25: # ------------------------------------------------------------------ #
 26: # Tests for LikelihoodModel ABC
 27: # ------------------------------------------------------------------ #
 28: 
 29: 
 30: class TestLikelihoodModelABC:
 31:     """Tests for the abstract base class contract."""
 32: 
 33:     def test_abstract_interface_cannot_instantiate(self) -> None:
 34:         """LikelihoodModel ABC cannot be instantiated directly."""
 35:         with pytest.raises(TypeError):
 36:             LikelihoodModel()  # type: ignore[abstract]
 37: 
 38:     def test_embedding_cache_on_subclass(self, sample_corpus: list[str]) -> None:
 39:         """Concrete subclass inherits embedding_cache dict."""
 40:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
 41:         assert hasattr(model, "embedding_cache"), "Missing embedding_cache attribute"
 42:         assert isinstance(model.embedding_cache, dict), "embedding_cache should be dict"
 43: 
 44: 
 45: # ------------------------------------------------------------------ #
 46: # Tests for TfIdfLikelihood
 47: # ------------------------------------------------------------------ #
 48: 
 49: 
 50: class TestTfIdfLikelihood:
 51:     """Tests for TF-IDF based likelihood model."""
 52: 
 53:     def test_tfidf_requires_fit(self) -> None:
 54:         """score() before fit() raises RuntimeError."""
 55:         model = TfIdfLikelihood()
 56:         with pytest.raises(RuntimeError, match="must be fit"):
 57:             model.score("test clue", ["option1", "option2"])
 58: 
 59:     def test_tfidf_embed_requires_fit(self) -> None:
 60:         """_embed_batch() before fit() raises RuntimeError."""
 61:         model = TfIdfLikelihood()
 62:         with pytest.raises(RuntimeError, match="must be fit"):
 63:             model._embed_batch(["test text"])
 64: 
 65:     def test_tfidf_fit_and_score(self, sample_corpus: list[str]) -> None:
 66:         """After fitting, score returns correct shape and dtype.
 67: 
 68:         Also verifies that more relevant text scores higher.
 69:         """
 70:         model = TfIdfLikelihood()
 71:         model.fit(sample_corpus)
 72: 
 73:         scores = model.score(
 74:             "Who was the first president?",
 75:             ["George Washington first president", "Abraham Lincoln Civil War"],
 76:         )
 77:         assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
 78:         assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"
 79:         # Washington should score higher for "first president" clue
 80:         assert scores[0] >= scores[1], (
 81:             f"Washington ({scores[0]:.3f}) should score >= Lincoln ({scores[1]:.3f})"
 82:         )
 83: 
 84:     def test_tfidf_embed_batch(self, sample_corpus: list[str]) -> None:
 85:         """_embed_batch produces dense vectors of correct shape."""
 86:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
 87:         embeddings = model._embed_batch(["test text one", "test text two"])
 88:         assert embeddings.shape[0] == 2, f"Expected 2 rows, got {embeddings.shape[0]}"
 89:         assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
 90:         vocab_size = len(model.vectorizer.vocabulary_)
 91:         assert embeddings.shape[1] == vocab_size, (
 92:             f"Expected {vocab_size} cols, got {embeddings.shape[1]}"
 93:         )
 94: 
 95:     def test_tfidf_corpus_in_constructor(self, sample_corpus: list[str]) -> None:
 96:         """Passing corpus_texts to __init__ auto-fits the model."""
 97:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
 98:         assert model._is_fit is True, "Model should be fit after corpus in constructor"
 99:         # Should work without explicit fit()
100:         scores = model.score("president", ["Washington", "Lincoln"])
101:         assert scores.shape == (2,)
102: 
103:     def test_tfidf_fit_returns_self(self, sample_corpus: list[str]) -> None:
104:         """fit() returns self for method chaining."""
105:         model = TfIdfLikelihood()
106:         result = model.fit(sample_corpus)
107:         assert result is model, "fit() should return self"
108: 
109:     def test_tfidf_score_all_options(self, sample_corpus: list[str]) -> None:
110:         """Score works with 4 options matching K=4 environment setup."""
111:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
112:         scores = model.score(
113:             "first president United States",
114:             [
115:                 "George Washington commander revolutionary",
116:                 "Thomas Jefferson declaration independence",
117:                 "John Adams Massachusetts diplomat",
118:                 "Benjamin Franklin inventor Philadelphia",
119:             ],
120:         )
121:         assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
122:         assert all(np.isfinite(scores)), "All scores should be finite"
123: 
124:     def test_tfidf_embed_batch_normalized(self, sample_corpus: list[str]) -> None:
125:         """_embed_batch returns L2-normalized vectors (row norms ~1.0)."""
126:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
127:         embeddings = model._embed_batch(["George Washington president", "Thomas Jefferson"])
128:         norms = np.linalg.norm(embeddings, axis=1)
129:         np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)
130: 
131:     def test_tfidf_score_uses_cache(self, sample_corpus: list[str]) -> None:
132:         """score() populates embedding_cache via embed_and_cache()."""
133:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
134:         assert len(model.embedding_cache) == 0
135:         model.score("first president", ["Washington profile", "Lincoln profile"])
136:         assert len(model.embedding_cache) == 3  # 1 clue + 2 options
137: 
138:     def test_tfidf_score_cache_hit(self, sample_corpus: list[str]) -> None:
139:         """Repeated score() with same options reuses cache."""
140:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
141:         options = ["George Washington president", "Thomas Jefferson declaration"]
142:         model.score("first president", options)
143:         cache_after_first = len(model.embedding_cache)
144:         model.score("second president", options)
145:         # Only the new clue should be added; options are cached
146:         assert len(model.embedding_cache) == cache_after_first + 1
147: 
148:     def test_tfidf_score_matches_cosine_reference(self, sample_corpus: list[str]) -> None:
149:         """New cached score() matches sklearn cosine_similarity reference."""
150:         from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos
151: 
152:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
153:         clue = "Who was the first president?"
154:         options = [
155:             "George Washington first president commander revolutionary",
156:             "Abraham Lincoln Civil War emancipation",
157:             "Thomas Jefferson declaration independence Virginia",
158:             "Benjamin Franklin inventor Philadelphia diplomat",
159:         ]
160:         # Compute reference via sklearn cosine_similarity (old method)
161:         clue_vec = model.vectorizer.transform([clue])
162:         option_vecs = model.vectorizer.transform(options)
163:         ref_scores = sklearn_cos(clue_vec, option_vecs)[0].astype(np.float32)
164:         # Compute via new cached path
165:         actual_scores = model.score(clue, options)
166:         np.testing.assert_allclose(actual_scores, ref_scores, atol=1e-6)
167: 
168: 
169: # ------------------------------------------------------------------ #
170: # Tests for SBERTLikelihood
171: # ------------------------------------------------------------------ #
172: 
173: 
174: class TestSBERTLikelihood:
175:     """Tests for Sentence-BERT likelihood model."""
176: 
177:     def test_sbert_instantiation(self) -> None:
178:         """SBERTLikelihood can be instantiated with default model."""
179:         model = SBERTLikelihood()
180:         assert hasattr(model, "encoder"), "Missing encoder attribute"
181:         assert model.model_name == "all-MiniLM-L6-v2"
182: 
183:     def test_sbert_score_shape_and_dtype(self) -> None:
184:         """score() returns correct shape and dtype for 4 options."""
185:         model = SBERTLikelihood()
186:         scores = model.score(
187:             "first president United States",
188:             [
189:                 "George Washington first president commander",
190:                 "Thomas Jefferson third president declaration",
191:                 "John Adams second president Massachusetts",
192:                 "Benjamin Franklin inventor diplomat",
193:             ],
194:         )
195:         assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
196:         assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"
197: 
198:     def test_sbert_semantic_ranking(self) -> None:
199:         """SBERT ranks semantically similar text higher."""
200:         model = SBERTLikelihood()
201:         scores = model.score(
202:             "George Washington was the first president of the United States and led the Continental Army",
203:             [
204:                 "George Washington first president commander revolutionary war continental army",
205:                 "The theory of relativity was developed by Albert Einstein in physics",
206:             ],
207:         )
208:         # Washington profile should score much higher than Einstein
209:         assert scores[0] > scores[1], (
210:             f"Washington ({scores[0]:.3f}) should score > Einstein ({scores[1]:.3f})"
211:         )
212: 
213:     def test_sbert_embedding_cache_populated(self) -> None:
214:         """Embedding cache grows after first scoring call."""
215:         model = SBERTLikelihood()
216:         assert len(model.embedding_cache) == 0, "Cache should start empty"
217: 
218:         model.score("test clue", ["option A", "option B"])
219:         cache_after_first = len(model.embedding_cache)
220:         assert cache_after_first > 0, "Cache should be populated after score()"
221: 
222:     def test_sbert_embedding_cache_hit(self) -> None:
223:         """Repeated calls with same text use cache (size unchanged)."""
224:         model = SBERTLikelihood()
225:         scores1 = model.score("test clue", ["option A", "option B"])
226:         cache_size_1 = len(model.embedding_cache)
227: 
228:         scores2 = model.score("test clue", ["option A", "option B"])
229:         cache_size_2 = len(model.embedding_cache)
230: 
231:         assert cache_size_2 == cache_size_1, (
232:             f"Cache grew from {cache_size_1} to {cache_size_2} on repeated call"
233:         )
234:         np.testing.assert_array_almost_equal(
235:             scores1, scores2, decimal=5,
236:             err_msg="Cached results should match original",
237:         )
238: 
239:     def test_sbert_normalized_embeddings(self) -> None:
240:         """SBERT embeddings are L2-normalized (norm ~ 1.0)."""
241:         model = SBERTLikelihood()
242:         embeddings = model._embed_batch(["test sentence one", "test sentence two"])
243:         norms = np.linalg.norm(embeddings, axis=1)
244:         np.testing.assert_array_almost_equal(
245:             norms, np.ones(2), decimal=4,
246:             err_msg="Embeddings should be L2-normalized",
247:         )
248: 
249: 
250: # ------------------------------------------------------------------ #
251: # Tests for T5Likelihood (LIK-04, LIK-05)
252: # ------------------------------------------------------------------ #
253: 
254: 
255: class TestT5Likelihood:
256:     """Tests for T5 encoder likelihood model.
257: 
258:     Uses the sample_t5_model fixture (t5-small, module-scoped) from conftest.py
259:     so the model is loaded once per test file, not per test function.
260:     """
261: 
262:     def test_t5_semantic_scoring(self, sample_t5_model) -> None:
263:         """T5 should score semantically relevant options higher (LIK-04).
264: 
265:         "First president" clue should rank Washington higher than Einstein,
266:         demonstrating that T5 captures semantic similarity between question
267:         content and answer profiles.
268:         """
269:         clue = "This person was the first president of the United States"
270:         options = [
271:             "George Washington first president commander revolutionary war",
272:             "Albert Einstein physicist theory relativity Nobel Prize",
273:         ]
274: 
275:         scores = sample_t5_model.score(clue, options)
276: 
277:         assert isinstance(scores, np.ndarray)
278:         assert scores.dtype == np.float32
279:         assert len(scores) == 2
280:         # Washington should score higher than Einstein for "first president" query
281:         assert scores[0] > scores[1], (
282:             f"Expected Washington > Einstein, got {scores}"
283:         )
284: 
285:     def test_t5_embedding_cache(self, sample_t5_model) -> None:
286:         """T5 should cache embeddings and reuse them (LIK-05).
287: 
288:         After embedding two texts, the cache should contain 2 entries.
289:         Re-embedding the same texts should not grow the cache, and the
290:         returned embeddings should be identical.
291:         """
292:         # Clear cache to get a clean test
293:         sample_t5_model.embedding_cache.clear()
294: 
295:         texts = ["George Washington", "Thomas Jefferson"]
296: 
297:         # First call embeds and caches
298:         emb1 = sample_t5_model.embed_and_cache(texts)
299:         cache_size_1 = len(sample_t5_model.embedding_cache)
300: 
301:         # Second call reuses cache
302:         emb2 = sample_t5_model.embed_and_cache(texts)
303:         cache_size_2 = len(sample_t5_model.embedding_cache)
304: 
305:         np.testing.assert_array_equal(
306:             emb1, emb2, err_msg="Cached embeddings should match"
307:         )
308:         assert cache_size_1 == cache_size_2 == 2, (
309:             f"Cache size should not grow on reuse, got {cache_size_1} -> {cache_size_2}"
310:         )
311: 
312:     def test_t5_score_returns_float32(self, sample_t5_model) -> None:
313:         """T5 score should return float32 array, not probabilities.
314: 
315:         Scores are raw cosine similarities (not softmax probabilities),
316:         so they do not necessarily sum to 1.
317:         """
318:         scores = sample_t5_model.score("test clue", ["option 1", "option 2"])
319:         assert scores.dtype == np.float32
320:         assert scores.shape == (2,)
321:         # Scores are raw similarities, not probabilities (don't sum to 1)
322:         assert all(np.isfinite(scores)), "All scores should be finite"
323: 
324:     def test_build_t5_from_config(self) -> None:
325:         """Factory should construct T5Likelihood from config (LIK-04).
326: 
327:         The build_likelihood_from_config factory should recognize
328:         model="t5" and instantiate a T5Likelihood with the specified
329:         t5_name parameter.
330:         """
331:         from models.likelihoods import T5Likelihood, build_likelihood_from_config
332: 
333:         config = {
334:             "likelihood": {
335:                 "model": "t5",
336:                 "t5_name": "t5-small",
337:             }
338:         }
339: 
340:         model = build_likelihood_from_config(config)
341:         assert isinstance(model, T5Likelihood)
342:         assert model.model_name == "t5-small"
343: 
344:     def test_t5_handles_variable_length(self, sample_t5_model) -> None:
345:         """T5 should handle variable-length texts via attention mask.
346: 
347:         Short and long texts should both embed without error, producing
348:         embeddings of the same hidden dimension regardless of input length.
349:         """
350:         short = "Washington"
351:         long = (
352:             "George Washington was the first president of the United States "
353:             "and commander of the Continental Army during the Revolutionary War"
354:         )
355: 
356:         # Both should embed without error
357:         embs = sample_t5_model.embed_and_cache([short, long])
358:         assert embs.shape == (2, sample_t5_model.encoder.config.d_model), (
359:             f"Expected shape (2, {sample_t5_model.encoder.config.d_model}), "
360:             f"got {embs.shape}"
361:         )
362: 
363: 
364: # ------------------------------------------------------------------ #
365: # Tests for Embedding Cache Persistence
366: # ------------------------------------------------------------------ #
367: 
368: 
369: class TestEmbeddingCachePersistence:
370:     """Tests for save_cache / load_cache disk persistence on LikelihoodModel."""
371: 
372:     def test_save_load_cache_round_trip(self, tmp_path: Path, sample_corpus: list[str]) -> None:
373:         """save_cache writes .npz; load_cache restores identical entries."""
374:         model = SBERTLikelihood()
375:         texts = ["George Washington", "Thomas Jefferson", "Abraham Lincoln"]
376:         model.embed_and_cache(texts)
377:         assert len(model.embedding_cache) == 3
378: 
379:         cache_path = tmp_path / "cache.npz"
380:         saved = model.save_cache(cache_path)
381:         assert saved == 3
382:         assert cache_path.exists()
383: 
384:         model2 = SBERTLikelihood()
385:         assert len(model2.embedding_cache) == 0
386:         loaded = model2.load_cache(cache_path)
387:         assert loaded == 3
388: 
389:         for key in model.embedding_cache:
390:             np.testing.assert_array_equal(
391:                 model.embedding_cache[key],
392:                 model2.embedding_cache[key],
393:                 err_msg=f"Mismatch for key {key}",
394:             )
395: 
396:     def test_load_cache_missing_file(self, tmp_path: Path) -> None:
397:         """load_cache with nonexistent file returns 0 and leaves cache empty."""
398:         model = SBERTLikelihood()
399:         result = model.load_cache(tmp_path / "nonexistent.npz")
400:         assert result == 0
401:         assert len(model.embedding_cache) == 0
402: 
403:     def test_save_cache_empty(self, tmp_path: Path) -> None:
404:         """save_cache with empty cache creates a valid .npz with zero arrays."""
405:         model = SBERTLikelihood()
406:         cache_path = tmp_path / "empty.npz"
407:         saved = model.save_cache(cache_path)
408:         assert saved == 0
409:         assert cache_path.exists()
410: 
411:         # Should be loadable
412:         model2 = SBERTLikelihood()
413:         loaded = model2.load_cache(cache_path)
414:         assert loaded == 0
415: 
416:     def test_tfidf_save_cache_noop(self, sample_corpus: list[str]) -> None:
417:         """TfIdfLikelihood.save_cache is a no-op returning 0."""
418:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
419:         # Populate the cache with some embeddings
420:         model.embed_and_cache(["test text one", "test text two"])
421:         assert len(model.embedding_cache) > 0
422: 
423:         import tempfile
424:         import os
425:         with tempfile.TemporaryDirectory() as td:
426:             path = Path(td) / "should_not_exist.npz"
427:             result = model.save_cache(path)
428:             assert result == 0
429:             assert not path.exists(), "TfIdfLikelihood should NOT write a cache file"
430: 
431:     def test_load_cache_does_not_overwrite(self, tmp_path: Path) -> None:
432:         """load_cache merges without overwriting existing cache entries."""
433:         model = SBERTLikelihood()
434:         texts = ["Hello world"]
435:         model.embed_and_cache(texts)
436: 
437:         # Save this cache
438:         cache_path = tmp_path / "cache.npz"
439:         model.save_cache(cache_path)
440: 
441:         # Create a second model, pre-populate with the same key but different value
442:         model2 = SBERTLikelihood()
443:         from models.likelihoods import _text_key
444:         key = _text_key("Hello world")
445:         original_value = np.ones(384, dtype=np.float32)  # dummy
446:         model2.embedding_cache[key] = original_value
447: 
448:         loaded = model2.load_cache(cache_path)
449:         assert loaded == 0, "Key already present, so nothing should be loaded"
450: 
451:         # Original value should be preserved (not overwritten)
452:         np.testing.assert_array_equal(
453:             model2.embedding_cache[key],
454:             original_value,
455:             err_msg="Existing cache entry was overwritten by load_cache",
456:         )
457: 
458: 
459: class TestCacheMemory:
460:     """Verify cache_memory_bytes property for resource monitoring."""
461: 
462:     def test_tfidf_cache_memory_bytes(self, sample_corpus):
463:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
464:         assert model.cache_memory_bytes == 0
465:         model.embed_and_cache(["George Washington"])
466:         assert model.cache_memory_bytes > 0
467: 
468:     def test_empty_cache_zero_bytes(self, sample_corpus):
469:         model = TfIdfLikelihood(corpus_texts=sample_corpus)
470:         assert model.cache_memory_bytes == 0
````

## File: scripts/train_ppo.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Train PPO buzzer agent on belief-feature observations.
  4: 
  5: Loads MC questions, builds a likelihood model, creates a Gymnasium environment,
  6: trains an MLP policy with SB3 PPO, then evaluates with episode traces and
  7: summary metrics (accuracy, S_q, ECE, Brier score).
  8: 
  9: Usage:
 10:     python scripts/train_ppo.py --smoke              # Quick smoke test
 11:     python scripts/train_ppo.py --smoke --deterministic-eval
 12:     python scripts/train_ppo.py --config configs/custom.yaml
 13:     python scripts/train_ppo.py --timesteps 50000    # Override timesteps
 14: 
 15: Ported from qb-rl reference implementation (scripts/train_ppo.py) with
 16: import path adaptations for the unified qanta-buzzer codebase.
 17: """
 18: 
 19: from __future__ import annotations
 20: 
 21: import argparse
 22: from dataclasses import asdict
 23: from pathlib import Path
 24: import sys
 25: 
 26: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 27: if str(PROJECT_ROOT) not in sys.path:
 28:     sys.path.insert(0, str(PROJECT_ROOT))
 29: 
 30: from agents.ppo_buzzer import PPOBuzzer
 31: from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
 32: from qb_env.stop_only_env import StopOnlyEnv
 33: from qb_env.tossup_env import make_env_from_config, precompute_beliefs
 34: from qb_data.config import merge_overrides
 35: from scripts._common import (
 36:     ARTIFACT_DIR,
 37:     build_likelihood_model,
 38:     load_config,
 39:     load_embedding_cache,
 40:     load_mc_questions,
 41:     parse_overrides,
 42:     save_embedding_cache,
 43:     save_json,
 44: )
 45: 
 46: 
 47: def parse_args() -> argparse.Namespace:
 48:     """Parse command-line arguments.
 49: 
 50:     Returns
 51:     -------
 52:     argparse.Namespace
 53:         Parsed arguments with config, smoke, mc_path, timesteps, and
 54:         deterministic_eval fields.
 55:     """
 56:     parser = argparse.ArgumentParser(description="Train PPO buzzer.")
 57:     parser.add_argument(
 58:         "--config", type=str, default=None,
 59:         help="Path to YAML config file (default: configs/default.yaml).",
 60:     )
 61:     parser.add_argument(
 62:         "--smoke", action="store_true",
 63:         help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
 64:     )
 65:     parser.add_argument(
 66:         "--mc-path", type=str, default=None,
 67:         help="Optional MC dataset JSON path (overrides config-derived path).",
 68:     )
 69:     parser.add_argument(
 70:         "--timesteps", type=int, default=None,
 71:         help="Override total_timesteps from config.",
 72:     )
 73:     parser.add_argument(
 74:         "--seed", type=int, default=None,
 75:         help="Override PPO/environment seed from config.",
 76:     )
 77:     parser.add_argument(
 78:         "--deterministic-eval", action="store_true",
 79:         help="Use deterministic policy for post-training episode evaluation.",
 80:     )
 81:     parser.add_argument(
 82:         "--stochastic-eval", action="store_true",
 83:         help="Force stochastic policy sampling for post-training evaluation.",
 84:     )
 85:     parser.add_argument(
 86:         "--policy-mode",
 87:         type=str,
 88:         choices=["flat_kplus1", "stop_only"],
 89:         default="flat_kplus1",
 90:         help="Policy action space: flat K+1 actions (default) or binary stop_only.",
 91:     )
 92:     parser.add_argument(
 93:         "overrides",
 94:         nargs="*",
 95:         help="Config overrides: key=value (e.g. likelihood.model=tfidf)",
 96:     )
 97:     return parser.parse_args()
 98: 
 99: 
100: def main() -> None:
101:     """Train PPO agent and save model + evaluation artifacts."""
102:     args = parse_args()
103: 
104:     config = load_config(args.config, smoke=args.smoke)
105:     overrides = parse_overrides(args)
106:     if overrides:
107:         print(f"Applying overrides: {overrides}")
108:         config = merge_overrides(config, overrides)
109: 
110:     split = "smoke" if args.smoke else "main"
111:     out_dir = ARTIFACT_DIR / split
112:     mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"
113: 
114:     # Fallback: check data/processed/ if artifacts path doesn't exist
115:     if not mc_path.exists():
116:         fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
117:         if fallback.exists():
118:             print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
119:             mc_path = fallback
120: 
121:     print(f"Loading MC questions from: {mc_path}")
122:     mc_questions = load_mc_questions(mc_path)
123:     print(f"Loaded {len(mc_questions)} MC questions")
124: 
125:     print(f"Building likelihood model: {config['likelihood']['model']}")
126:     likelihood_model = build_likelihood_model(config, mc_questions)
127:     load_embedding_cache(likelihood_model, config)
128: 
129:     env_cfg = config["environment"]
130:     lik_cfg = config["likelihood"]
131: 
132:     print(f"Precomputing belief trajectories for {len(mc_questions)} questions...")
133:     belief_cache = precompute_beliefs(
134:         questions=mc_questions,
135:         likelihood_model=likelihood_model,
136:         belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
137:         beta=float(lik_cfg.get("beta", 5.0)),
138:         K=int(config["data"].get("K", 4)),
139:     )
140:     print(f"Cached {len(belief_cache)} belief vectors")
141:     save_embedding_cache(likelihood_model, config)
142: 
143:     env = make_env_from_config(
144:         mc_questions=mc_questions,
145:         likelihood_model=likelihood_model,
146:         config=config,
147:         precomputed_beliefs=belief_cache,
148:     )
149:     if args.policy_mode == "stop_only":
150:         print("Wrapping environment with StopOnlyEnv (WAIT/BUZZ only)...")
151:         env = StopOnlyEnv(env)
152: 
153:     ppo_cfg = config["ppo"]
154:     train_seed = int(args.seed if args.seed is not None else ppo_cfg.get("seed", 13))
155:     total_timesteps = int(
156:         args.timesteps if args.timesteps is not None else ppo_cfg["total_timesteps"]
157:     )
158: 
159:     print(f"Training PPO for {total_timesteps} timesteps...")
160:     agent = PPOBuzzer(
161:         env=env,
162:         learning_rate=float(ppo_cfg["learning_rate"]),
163:         n_steps=int(ppo_cfg["n_steps"]),
164:         batch_size=int(ppo_cfg["batch_size"]),
165:         n_epochs=int(ppo_cfg["n_epochs"]),
166:         gamma=float(ppo_cfg["gamma"]),
167:         seed=train_seed,
168:         policy_kwargs=ppo_cfg.get("policy_kwargs", {"net_arch": [64, 64]}),
169:         verbose=1,
170:     )
171: 
172:     agent.train(total_timesteps=total_timesteps)
173:     model_path = out_dir / "ppo_model"
174:     agent.save(model_path)
175: 
176:     eval_deterministic = True
177:     if args.stochastic_eval:
178:         eval_deterministic = False
179:     elif args.deterministic_eval:
180:         eval_deterministic = True
181: 
182:     print(
183:         f"Evaluating PPO agent on {len(mc_questions)} questions "
184:         f"(deterministic={eval_deterministic})..."
185:     )
186:     traces = [
187:         asdict(
188:             agent.run_episode(
189:                 deterministic=eval_deterministic,
190:                 question_idx=i,
191:             )
192:         )
193:         for i in range(len(mc_questions))
194:     ]
195:     summary = {**summarize_buzz_metrics(traces), **calibration_at_buzz(traces)}
196: 
197:     save_json(out_dir / "ppo_runs.json", traces)
198:     save_json(out_dir / "ppo_summary.json", summary)
199:     print(f"Saved PPO model to: {model_path}.zip")
200:     print(f"Saved PPO summaries to: {out_dir}")
201: 
202: 
203: if __name__ == "__main__":
204:     main()
````

## File: tests/test_environment.py
````python
  1: """Test suite for qb_env/tossup_env.py — TossupMCEnv Gymnasium environment.
  2: 
  3: Covers:
  4: - ENV-01: Gymnasium interface compliance (reset, step, spaces)
  5: - ENV-02: Action space Discrete(K+1) with WAIT and BUZZ actions
  6: - ENV-04: Reward modes (time_penalty, simple, human_grounded)
  7: - ENV-05: Likelihood model pluggability
  8: """
  9: 
 10: from __future__ import annotations
 11: 
 12: import sys
 13: from unittest.mock import MagicMock
 14: 
 15: import gymnasium as gym
 16: import numpy as np
 17: import pytest
 18: 
 19: from models.likelihoods import SBERTLikelihood, TfIdfLikelihood
 20: from qb_data.mc_builder import MCQuestion
 21: from qb_env.tossup_env import TossupMCEnv, precompute_beliefs
 22: 
 23: 
 24: # ------------------------------------------------------------------ #
 25: # Helpers
 26: # ------------------------------------------------------------------ #
 27: 
 28: 
 29: def _make_env(
 30:     mc_question: MCQuestion,
 31:     corpus: list[str] | None = None,
 32:     reward_mode: str = "simple",
 33:     wait_penalty: float = 0.0,
 34:     buzz_correct: float = 1.0,
 35:     buzz_incorrect: float = -1.0,
 36:     belief_mode: str = "from_scratch",
 37:     beta: float = 5.0,
 38:     use_sbert: bool = False,
 39: ) -> TossupMCEnv:
 40:     """Create a TossupMCEnv with TF-IDF or SBERT likelihood model.
 41: 
 42:     Helper for tests that need a configured environment without going
 43:     through the factory function.
 44:     """
 45:     if use_sbert:
 46:         model = SBERTLikelihood()
 47:     else:
 48:         if corpus is None:
 49:             corpus = mc_question.option_profiles[:]
 50:         model = TfIdfLikelihood(corpus_texts=corpus)
 51:     return TossupMCEnv(
 52:         questions=[mc_question],
 53:         likelihood_model=model,
 54:         K=4,
 55:         reward_mode=reward_mode,
 56:         wait_penalty=wait_penalty,
 57:         buzz_correct=buzz_correct,
 58:         buzz_incorrect=buzz_incorrect,
 59:         belief_mode=belief_mode,
 60:         beta=beta,
 61:     )
 62: 
 63: 
 64: # ------------------------------------------------------------------ #
 65: # Tests: Gymnasium Interface (ENV-01)
 66: # ------------------------------------------------------------------ #
 67: 
 68: 
 69: class TestGymnasiumInterface:
 70:     """Tests for Gymnasium API compliance."""
 71: 
 72:     def test_isinstance_gym_env(self, sample_mc_question: MCQuestion) -> None:
 73:         """TossupMCEnv is a subclass of gym.Env."""
 74:         env = _make_env(sample_mc_question)
 75:         assert isinstance(env, gym.Env), "TossupMCEnv should be a gym.Env subclass"
 76: 
 77:     def test_has_reset_and_step(self, sample_mc_question: MCQuestion) -> None:
 78:         """Environment has reset() and step() methods."""
 79:         env = _make_env(sample_mc_question)
 80:         assert hasattr(env, "reset"), "Missing reset() method"
 81:         assert hasattr(env, "step"), "Missing step() method"
 82:         assert callable(env.reset), "reset should be callable"
 83:         assert callable(env.step), "step should be callable"
 84: 
 85:     def test_action_space_discrete(self, sample_mc_question: MCQuestion) -> None:
 86:         """Action space is Discrete(K+1) = Discrete(5) for K=4."""
 87:         env = _make_env(sample_mc_question)
 88:         assert isinstance(env.action_space, gym.spaces.Discrete), (
 89:             f"Expected Discrete, got {type(env.action_space)}"
 90:         )
 91:         assert env.action_space.n == 5, (
 92:             f"Expected Discrete(5) for K=4, got Discrete({env.action_space.n})"
 93:         )
 94: 
 95:     def test_observation_space_box(self, sample_mc_question: MCQuestion) -> None:
 96:         """Observation space is Box(K+6,) = Box(10,) for K=4."""
 97:         env = _make_env(sample_mc_question)
 98:         assert isinstance(env.observation_space, gym.spaces.Box), (
 99:             f"Expected Box, got {type(env.observation_space)}"
100:         )
101:         assert env.observation_space.shape == (10,), (
102:             f"Expected shape (10,), got {env.observation_space.shape}"
103:         )
104:         assert env.observation_space.dtype == np.float32, (
105:             f"Expected float32, got {env.observation_space.dtype}"
106:         )
107: 
108:     def test_action_space_contains_all_valid_actions(
109:         self, sample_mc_question: MCQuestion
110:     ) -> None:
111:         """All actions 0..K are valid in the action space."""
112:         env = _make_env(sample_mc_question)
113:         for action in range(5):
114:             assert env.action_space.contains(action), (
115:                 f"Action {action} should be valid"
116:             )
117:         assert not env.action_space.contains(5), "Action 5 should be invalid for K=4"
118:         assert not env.action_space.contains(-1), "Action -1 should be invalid"
119: 
120: 
121: # ------------------------------------------------------------------ #
122: # Tests: Episode Flow
123: # ------------------------------------------------------------------ #
124: 
125: 
126: class TestEpisodeFlow:
127:     """Tests for reset/step/termination lifecycle."""
128: 
129:     def test_reset_returns_obs_and_info(self, sample_mc_question: MCQuestion) -> None:
130:         """reset() returns (observation, info) tuple."""
131:         env = _make_env(sample_mc_question)
132:         result = env.reset()
133:         assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
134:         assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
135: 
136:     def test_reset_obs_shape_dtype(self, sample_mc_question: MCQuestion) -> None:
137:         """Observation from reset is (K+6,) float32."""
138:         env = _make_env(sample_mc_question)
139:         obs, info = env.reset()
140:         assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
141:         assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
142: 
143:     def test_reset_info_contains_qid(self, sample_mc_question: MCQuestion) -> None:
144:         """Info dict from reset contains qid."""
145:         env = _make_env(sample_mc_question)
146:         _obs, info = env.reset()
147:         assert "qid" in info, "Info should contain 'qid'"
148:         assert info["qid"] == "test_q1", f"Expected 'test_q1', got {info['qid']}"
149: 
150:     def test_reset_initializes_state(self, sample_mc_question: MCQuestion) -> None:
151:         """After reset, step_idx=0, not terminated, not truncated."""
152:         env = _make_env(sample_mc_question)
153:         env.reset()
154:         assert env.step_idx == 0, f"step_idx should be 0, got {env.step_idx}"
155:         assert env.terminated is False, "terminated should be False"
156:         assert env.truncated is False, "truncated should be False"
157: 
158:     def test_wait_action_advances_step(self, sample_mc_question: MCQuestion) -> None:
159:         """WAIT (action 0) increments step_idx and returns not terminated."""
160:         env = _make_env(sample_mc_question)
161:         env.reset()
162:         obs, reward, terminated, truncated, info = env.step(0)
163:         assert not terminated, "Should not terminate on WAIT"
164:         assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
165:         assert env.step_idx == 1, f"step_idx should be 1, got {env.step_idx}"
166: 
167:     def test_buzz_correct_terminates(self, sample_mc_question: MCQuestion) -> None:
168:         """Buzzing with correct answer (action 1 = option 0 = gold) terminates."""
169:         env = _make_env(sample_mc_question)
170:         env.reset()
171:         obs, reward, terminated, truncated, info = env.step(1)  # gold_index=0, action=1
172:         assert terminated is True, "Should terminate on buzz"
173:         assert truncated is False, "Should not be truncated"
174:         assert info["correct"] is True, "Buzzing with gold should be correct"
175:         assert info["chosen_idx"] == 0, f"chosen_idx should be 0, got {info['chosen_idx']}"
176: 
177:     def test_buzz_incorrect_terminates(self, sample_mc_question: MCQuestion) -> None:
178:         """Buzzing with incorrect answer terminates with correct=False."""
179:         env = _make_env(sample_mc_question)
180:         env.reset()
181:         obs, reward, terminated, truncated, info = env.step(2)  # option 1 = incorrect
182:         assert terminated is True, "Should terminate on buzz"
183:         assert info["correct"] is False, "Buzzing with wrong answer should be incorrect"
184: 
185:     def test_forced_termination(self, sample_mc_question: MCQuestion) -> None:
186:         """Exhausting all clues causes truncation with forced choice."""
187:         env = _make_env(sample_mc_question)
188:         env.reset()
189:         total = env.total_steps  # 6 steps for sample question
190: 
191:         # WAIT until all clues exhausted
192:         for i in range(total):
193:             obs, reward, terminated, truncated, info = env.step(0)
194:             if truncated:
195:                 break
196: 
197:         assert truncated is True, "Should be truncated after exhausting clues"
198:         assert "forced_choice" in info, "Info should contain 'forced_choice'"
199:         assert "forced_correct" in info, "Info should contain 'forced_correct'"
200:         assert isinstance(info["forced_choice"], int), "forced_choice should be int"
201: 
202:     def test_step_before_reset_raises(self, sample_mc_question: MCQuestion) -> None:
203:         """Calling step() before reset() raises RuntimeError."""
204:         env = _make_env(sample_mc_question)
205:         with pytest.raises(RuntimeError, match="reset"):
206:             env.step(0)
207: 
208:     def test_step_after_terminated_raises(self, sample_mc_question: MCQuestion) -> None:
209:         """Calling step() after termination raises RuntimeError."""
210:         env = _make_env(sample_mc_question)
211:         env.reset()
212:         env.step(1)  # buzz to terminate
213:         with pytest.raises(RuntimeError, match="terminated"):
214:             env.step(0)
215: 
216:     def test_invalid_action_raises(self, sample_mc_question: MCQuestion) -> None:
217:         """Invalid action raises ValueError."""
218:         env = _make_env(sample_mc_question)
219:         env.reset()
220:         with pytest.raises(ValueError, match="Invalid action"):
221:             env.step(99)
222: 
223:     def test_default_end_mode_is_force_commit(
224:         self, sample_mc_question: MCQuestion
225:     ) -> None:
226:         """Default env keeps legacy forced-commit behavior."""
227:         env = _make_env(sample_mc_question)
228:         assert env.end_mode == "force_commit"
229: 
230:     def test_no_buzz_end_mode_returns_marker_and_reward(
231:         self, sample_mc_question: MCQuestion
232:     ) -> None:
233:         """no_buzz mode truncates without forcing an answer choice."""
234:         corpus = sample_mc_question.option_profiles[:]
235:         model = TfIdfLikelihood(corpus_texts=corpus)
236:         env = TossupMCEnv(
237:             questions=[sample_mc_question],
238:             likelihood_model=model,
239:             K=4,
240:             reward_mode="simple",
241:             end_mode="no_buzz",
242:             no_buzz_reward=0.25,
243:         )
244:         env.reset()
245: 
246:         for _ in range(env.total_steps):
247:             _obs, reward, _terminated, truncated, info = env.step(0)
248:             if truncated:
249:                 break
250: 
251:         assert truncated is True
252:         assert reward == pytest.approx(0.25)
253:         assert info["no_buzz"] is True
254:         assert info["forced_choice"] == -1
255:         assert info["forced_correct"] is False
256: 
257:     def test_invalid_end_mode_raises_on_terminal_wait(
258:         self, sample_mc_question: MCQuestion
259:     ) -> None:
260:         """Unknown end_mode raises ValueError at horizon."""
261:         corpus = sample_mc_question.option_profiles[:]
262:         model = TfIdfLikelihood(corpus_texts=corpus)
263:         env = TossupMCEnv(
264:             questions=[sample_mc_question],
265:             likelihood_model=model,
266:             K=4,
267:             reward_mode="simple",
268:             end_mode="unknown_mode",
269:         )
270:         env.reset()
271: 
272:         with pytest.raises(ValueError, match="Unknown end_mode"):
273:             for _ in range(env.total_steps):
274:                 env.step(0)
275: 
276: 
277: class TestStopOnlyEnv:
278:     """Tests for the stop-only WAIT/BUZZ wrapper."""
279: 
280:     def test_stop_only_env_has_discrete2_action_space(
281:         self, sample_mc_question: MCQuestion
282:     ) -> None:
283:         """StopOnlyEnv exposes a binary action space."""
284:         from qb_env import StopOnlyEnv
285: 
286:         env = StopOnlyEnv(_make_env(sample_mc_question))
287:         assert isinstance(env.action_space, gym.spaces.Discrete)
288:         assert env.action_space.n == 2
289: 
290:     def test_stop_only_wait_delegates_to_base_env(
291:         self, sample_mc_question: MCQuestion
292:     ) -> None:
293:         """Action 0 remains a WAIT in the wrapped env."""
294:         from qb_env import StopOnlyEnv
295: 
296:         base_env = _make_env(sample_mc_question)
297:         env = StopOnlyEnv(base_env)
298:         env.reset()
299: 
300:         _obs, _reward, terminated, truncated, _info = env.step(0)
301:         assert not terminated
302:         assert not truncated
303:         assert base_env.step_idx == 1
304: 
305:     def test_stop_only_buzz_uses_argmax_belief(
306:         self, sample_mc_question: MCQuestion
307:     ) -> None:
308:         """Action 1 maps to BUZZ with the current belief argmax."""
309:         from qb_env import StopOnlyEnv
310: 
311:         base_env = _make_env(sample_mc_question)
312:         env = StopOnlyEnv(base_env)
313:         env.reset()
314:         base_env.belief = np.array([0.05, 0.8, 0.1, 0.05], dtype=np.float32)
315: 
316:         _obs, _reward, terminated, truncated, info = env.step(1)
317:         assert terminated
318:         assert not truncated
319:         assert info["chosen_idx"] == 1
320:         assert info["correct"] is False
321: 
322:     def test_stop_only_invalid_action_raises(
323:         self, sample_mc_question: MCQuestion
324:     ) -> None:
325:         """StopOnlyEnv rejects actions outside its Discrete(2) contract."""
326:         from qb_env import StopOnlyEnv
327: 
328:         base_env = _make_env(sample_mc_question)
329:         env = StopOnlyEnv(base_env)
330:         env.reset()
331: 
332:         with pytest.raises(ValueError, match="Invalid action"):
333:             env.step(2)
334: 
335:     def test_train_ppo_policy_mode_defaults_flat_kplus1(
336:         self, monkeypatch: pytest.MonkeyPatch
337:     ) -> None:
338:         """train_ppo CLI defaults to flat_kplus1 for compatibility."""
339:         from scripts.train_ppo import parse_args
340: 
341:         monkeypatch.setattr(sys, "argv", ["train_ppo.py"])
342:         args = parse_args()
343:         assert args.policy_mode == "flat_kplus1"
344: 
345: 
346: # ------------------------------------------------------------------ #
347: # Tests: Reward Modes (ENV-04)
348: # ------------------------------------------------------------------ #
349: 
350: 
351: class TestRewardModes:
352:     """Tests for different reward computation modes."""
353: 
354:     def test_reward_simple_correct(self, sample_mc_question: MCQuestion) -> None:
355:         """Simple mode: correct buzz gives +1.0."""
356:         env = _make_env(sample_mc_question, reward_mode="simple")
357:         env.reset()
358:         _obs, reward, _term, _trunc, _info = env.step(1)  # correct buzz
359:         assert reward == 1.0, f"Simple correct reward should be 1.0, got {reward}"
360: 
361:     def test_reward_simple_incorrect(self, sample_mc_question: MCQuestion) -> None:
362:         """Simple mode: incorrect buzz gives -1.0."""
363:         env = _make_env(sample_mc_question, reward_mode="simple")
364:         env.reset()
365:         _obs, reward, _term, _trunc, _info = env.step(2)  # incorrect buzz
366:         assert reward == -1.0, f"Simple incorrect reward should be -1.0, got {reward}"
367: 
368:     def test_reward_simple_wait_no_penalty(self, sample_mc_question: MCQuestion) -> None:
369:         """Simple mode: WAIT has 0 reward regardless of wait_penalty setting."""
370:         env = _make_env(
371:             sample_mc_question, reward_mode="simple", wait_penalty=0.1
372:         )
373:         env.reset()
374:         _obs, reward, _term, _trunc, _info = env.step(0)
375:         assert reward == 0.0, f"Simple WAIT reward should be 0.0, got {reward}"
376: 
377:     def test_reward_time_penalty_wait(self, sample_mc_question: MCQuestion) -> None:
378:         """Time penalty mode: WAIT incurs -wait_penalty."""
379:         env = _make_env(
380:             sample_mc_question, reward_mode="time_penalty", wait_penalty=0.1
381:         )
382:         env.reset()
383:         _obs, reward, _term, _trunc, _info = env.step(0)
384:         assert abs(reward - (-0.1)) < 1e-6, (
385:             f"Time penalty WAIT reward should be -0.1, got {reward}"
386:         )
387: 
388:     def test_reward_time_penalty_buzz_correct(
389:         self, sample_mc_question: MCQuestion
390:     ) -> None:
391:         """Time penalty mode: correct buzz gives buzz_correct."""
392:         env = _make_env(
393:             sample_mc_question,
394:             reward_mode="time_penalty",
395:             buzz_correct=1.0,
396:             wait_penalty=0.1,
397:         )
398:         env.reset()
399:         _obs, reward, _term, _trunc, _info = env.step(1)
400:         assert reward == 1.0, f"Time penalty correct buzz should be 1.0, got {reward}"
401: 
402:     def test_reward_time_penalty_cumulative(
403:         self, sample_mc_question: MCQuestion
404:     ) -> None:
405:         """Time penalty mode: waiting then buzzing accumulates penalties."""
406:         env = _make_env(
407:             sample_mc_question,
408:             reward_mode="time_penalty",
409:             wait_penalty=0.1,
410:             buzz_correct=1.0,
411:         )
412:         env.reset()
413:         # Wait 2 steps (-0.2 cumulative), then buzz correct (+1.0)
414:         total_reward = 0.0
415:         _obs, r1, _t, _tr, _info = env.step(0)
416:         total_reward += r1
417:         _obs, r2, _t, _tr, _info = env.step(0)
418:         total_reward += r2
419:         _obs, r3, _t, _tr, _info = env.step(1)  # buzz correct
420:         total_reward += r3
421:         assert abs(total_reward - 0.8) < 1e-6, (
422:             f"Cumulative reward should be ~0.8, got {total_reward}"
423:         )
424: 
425:     def test_reward_human_grounded(self, sample_mc_question: MCQuestion) -> None:
426:         """Human grounded mode works without human buzz data (returns normal reward)."""
427:         env = _make_env(
428:             sample_mc_question,
429:             reward_mode="human_grounded",
430:             buzz_correct=1.0,
431:             buzz_incorrect=-0.5,
432:         )
433:         env.reset()
434:         # With no human buzz positions, reward should be buzz_correct/incorrect
435:         _obs, reward, _term, _trunc, _info = env.step(1)
436:         assert reward == 1.0, f"Human grounded correct buzz should be 1.0, got {reward}"
437: 
438:     def test_reward_human_grounded_with_positions(self) -> None:
439:         """Human grounded mode: buzzing after human position gives 0.0."""
440:         # Create question with human buzz at position 0 (very early)
441:         mc_q = MCQuestion(
442:             qid="hg_test",
443:             question="Who was the first president?",
444:             tokens=["Who", "was", "the", "first", "president", "?"],
445:             answer_primary="George Washington",
446:             clean_answers=["George Washington"],
447:             run_indices=[0, 2, 4, 5],
448:             human_buzz_positions=[(0, 10)],  # Most humans buzz at position 0
449:             category="History",
450:             cumulative_prefixes=[
451:                 "Who",
452:                 "Who was the",
453:                 "Who was the first president",
454:                 "Who was the first president ?",
455:             ],
456:             options=["George Washington", "Jefferson", "Adams", "Franklin"],
457:             gold_index=0,
458:             option_profiles=["Washington", "Jefferson", "Adams", "Franklin"],
459:             option_answer_primary=["George Washington", "Jefferson", "Adams", "Franklin"],
460:             distractor_strategy="test",
461:         )
462:         corpus = mc_q.option_profiles[:]
463:         model = TfIdfLikelihood(corpus_texts=corpus)
464:         env = TossupMCEnv(
465:             questions=[mc_q],
466:             likelihood_model=model,
467:             K=4,
468:             reward_mode="human_grounded",
469:             buzz_correct=1.0,
470:             buzz_incorrect=-0.5,
471:         )
472:         env.reset()
473:         # Wait a few steps so agent buzzes after human position (0)
474:         env.step(0)  # step 0 -> reveal clue at position 0
475:         env.step(0)  # step 1 -> reveal clue at position 2
476:         _obs, reward, _term, _trunc, _info = env.step(1)  # buzz at step 2
477:         # Agent buzzes at token pos > 0 (human), so reward should be 0.0
478:         assert reward == 0.0, f"Should get 0.0 for buzzing after human, got {reward}"
479: 
480: 
481: # ------------------------------------------------------------------ #
482: # Tests: Likelihood Model Pluggability (ENV-05)
483: # ------------------------------------------------------------------ #
484: 
485: 
486: class TestLikelihoodPluggability:
487:     """Tests for interchangeable likelihood models."""
488: 
489:     def test_tfidf_model_produces_valid_obs(
490:         self, sample_mc_question: MCQuestion
491:     ) -> None:
492:         """TF-IDF likelihood model produces valid observations."""
493:         env = _make_env(sample_mc_question, use_sbert=False)
494:         obs, info = env.reset()
495:         assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
496:         assert np.all(np.isfinite(obs)), "All observations should be finite"
497:         # Take a step
498:         obs2, _r, _t, _tr, _info = env.step(0)
499:         assert obs2.shape == (10,), f"Expected (10,), got {obs2.shape}"
500:         assert np.all(np.isfinite(obs2)), "Step observations should be finite"
501: 
502:     def test_sbert_model_produces_valid_obs(
503:         self, sample_mc_question: MCQuestion
504:     ) -> None:
505:         """SBERT likelihood model produces valid observations."""
506:         env = _make_env(sample_mc_question, use_sbert=True)
507:         obs, info = env.reset()
508:         assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
509:         assert np.all(np.isfinite(obs)), "All observations should be finite"
510:         # Take a step
511:         obs2, _r, _t, _tr, _info = env.step(0)
512:         assert obs2.shape == (10,), f"Expected (10,), got {obs2.shape}"
513:         assert np.all(np.isfinite(obs2)), "Step observations should be finite"
514: 
515:     def test_both_models_same_obs_shape(
516:         self, sample_mc_question: MCQuestion
517:     ) -> None:
518:         """Both TF-IDF and SBERT produce same observation shape."""
519:         env_tfidf = _make_env(sample_mc_question, use_sbert=False)
520:         env_sbert = _make_env(sample_mc_question, use_sbert=True)
521: 
522:         obs_tfidf, _ = env_tfidf.reset(seed=42)
523:         obs_sbert, _ = env_sbert.reset(seed=42)
524: 
525:         assert obs_tfidf.shape == obs_sbert.shape, (
526:             f"TF-IDF obs {obs_tfidf.shape} != SBERT obs {obs_sbert.shape}"
527:         )
528:         assert obs_tfidf.dtype == obs_sbert.dtype, (
529:             f"TF-IDF dtype {obs_tfidf.dtype} != SBERT dtype {obs_sbert.dtype}"
530:         )
531: 
532: 
533: # ------------------------------------------------------------------ #
534: # Tests: Belief Modes
535: # ------------------------------------------------------------------ #
536: 
537: 
538: class TestBeliefModes:
539:     """Tests for different belief computation modes."""
540: 
541:     def test_from_scratch_belief(self, sample_mc_question: MCQuestion) -> None:
542:         """from_scratch mode recomputes belief from cumulative prefix."""
543:         env = _make_env(sample_mc_question, belief_mode="from_scratch")
544:         env.reset()
545:         # Wait several steps to get a more discriminative clue prefix
546:         for _ in range(3):
547:             env.step(0)
548:         # After multiple steps with more context, belief should be valid
549:         # and at least one option should have higher probability
550:         assert abs(env.belief.sum() - 1.0) < 1e-5, (
551:             f"Belief should sum to 1.0, got {env.belief.sum()}"
552:         )
553:         assert all(env.belief >= 0), "All beliefs should be non-negative"
554:         assert env.belief.dtype == np.float32, "Belief should be float32"
555: 
556:     def test_sequential_bayes_belief(self, sample_mc_question: MCQuestion) -> None:
557:         """sequential_bayes mode updates belief incrementally."""
558:         env = _make_env(sample_mc_question, belief_mode="sequential_bayes")
559:         env.reset()
560:         env.step(0)  # first WAIT
561:         # Belief should sum to ~1.0
562:         assert abs(env.belief.sum() - 1.0) < 1e-5, (
563:             f"Belief should sum to 1.0, got {env.belief.sum()}"
564:         )
565: 
566:     def test_invalid_belief_mode_raises(self, sample_mc_question: MCQuestion) -> None:
567:         """Unknown belief mode raises ValueError on step."""
568:         env = _make_env(sample_mc_question, belief_mode="unknown_mode")
569:         env.reset()
570:         with pytest.raises(ValueError, match="Unknown belief_mode"):
571:             env.step(0)
572: 
573: 
574: # ------------------------------------------------------------------ #
575: # Tests: Constructor Validation
576: # ------------------------------------------------------------------ #
577: 
578: 
579: class TestConstructorValidation:
580:     """Tests for constructor input validation."""
581: 
582:     def test_empty_questions_raises(self) -> None:
583:         """Empty question list raises ValueError."""
584:         model = TfIdfLikelihood(corpus_texts=["test"])
585:         with pytest.raises(ValueError, match="cannot be empty"):
586:             TossupMCEnv(questions=[], likelihood_model=model)
587: 
588:     def test_k_less_than_2_raises(self, sample_mc_question: MCQuestion) -> None:
589:         """K < 2 raises ValueError."""
590:         model = TfIdfLikelihood(corpus_texts=["test"])
591:         with pytest.raises(ValueError, match="K must be >= 2"):
592:             TossupMCEnv(
593:                 questions=[sample_mc_question], likelihood_model=model, K=1
594:             )
595: 
596: 
597: # ------------------------------------------------------------------ #
598: # Tests: Precomputed Beliefs (OPT-1)
599: # ------------------------------------------------------------------ #
600: 
601: 
602: class TestPrecomputedBeliefs:
603:     """Tests for precomputed belief trajectory bypass."""
604: 
605:     def test_precomputed_matches_live_from_scratch(
606:         self, sample_mc_question: MCQuestion
607:     ) -> None:
608:         """Precomputed env produces identical beliefs as live env (from_scratch)."""
609:         corpus = sample_mc_question.option_profiles[:]
610:         model = TfIdfLikelihood(corpus_texts=corpus)
611:         questions = [sample_mc_question]
612: 
613:         # Run live env and record beliefs at each step
614:         live_env = TossupMCEnv(
615:             questions=questions, likelihood_model=model, K=4,
616:             belief_mode="from_scratch", beta=5.0,
617:         )
618:         live_env.reset(seed=42, options={"question_idx": 0})
619:         live_beliefs = []
620:         for _ in range(live_env.total_steps):
621:             live_env.step(0)  # WAIT
622:             live_beliefs.append(live_env.belief.copy())
623:             if live_env.truncated:
624:                 break
625: 
626:         # Build precomputed cache
627:         cache = precompute_beliefs(
628:             questions=questions, likelihood_model=model,
629:             belief_mode="from_scratch", beta=5.0, K=4,
630:         )
631: 
632:         # Run precomputed env and compare beliefs
633:         pre_env = TossupMCEnv(
634:             questions=questions, likelihood_model=model, K=4,
635:             belief_mode="from_scratch", beta=5.0,
636:             precomputed_beliefs=cache,
637:         )
638:         pre_env.reset(seed=42, options={"question_idx": 0})
639:         for i in range(len(live_beliefs)):
640:             pre_env.step(0)
641:             np.testing.assert_allclose(
642:                 pre_env.belief, live_beliefs[i], atol=1e-6,
643:                 err_msg=f"Belief mismatch at step {i} (from_scratch)",
644:             )
645:             if pre_env.truncated:
646:                 break
647: 
648:     def test_precomputed_matches_live_sequential_bayes(
649:         self, sample_mc_question: MCQuestion
650:     ) -> None:
651:         """Precomputed env produces identical beliefs as live env (sequential_bayes)."""
652:         corpus = sample_mc_question.option_profiles[:]
653:         model = TfIdfLikelihood(corpus_texts=corpus)
654:         questions = [sample_mc_question]
655: 
656:         # Run live env
657:         live_env = TossupMCEnv(
658:             questions=questions, likelihood_model=model, K=4,
659:             belief_mode="sequential_bayes", beta=5.0,
660:         )
661:         live_env.reset(seed=42, options={"question_idx": 0})
662:         live_beliefs = []
663:         for _ in range(live_env.total_steps):
664:             live_env.step(0)
665:             live_beliefs.append(live_env.belief.copy())
666:             if live_env.truncated:
667:                 break
668: 
669:         # Build precomputed cache
670:         cache = precompute_beliefs(
671:             questions=questions, likelihood_model=model,
672:             belief_mode="sequential_bayes", beta=5.0, K=4,
673:         )
674: 
675:         # Run precomputed env
676:         pre_env = TossupMCEnv(
677:             questions=questions, likelihood_model=model, K=4,
678:             belief_mode="sequential_bayes", beta=5.0,
679:             precomputed_beliefs=cache,
680:         )
681:         pre_env.reset(seed=42, options={"question_idx": 0})
682:         for i in range(len(live_beliefs)):
683:             pre_env.step(0)
684:             np.testing.assert_allclose(
685:                 pre_env.belief, live_beliefs[i], atol=1e-6,
686:                 err_msg=f"Belief mismatch at step {i} (sequential_bayes)",
687:             )
688:             if pre_env.truncated:
689:                 break
690: 
691:     def test_precomputed_skips_scoring(
692:         self, sample_mc_question: MCQuestion
693:     ) -> None:
694:         """Precomputed env never calls likelihood_model.score()."""
695:         corpus = sample_mc_question.option_profiles[:]
696:         model = TfIdfLikelihood(corpus_texts=corpus)
697:         questions = [sample_mc_question]
698: 
699:         cache = precompute_beliefs(
700:             questions=questions, likelihood_model=model,
701:             belief_mode="from_scratch", beta=5.0, K=4,
702:         )
703: 
704:         # Replace score with a mock
705:         mock_model = MagicMock(spec=TfIdfLikelihood)
706:         mock_model.score = MagicMock()
707: 
708:         env = TossupMCEnv(
709:             questions=questions, likelihood_model=mock_model, K=4,
710:             belief_mode="from_scratch", beta=5.0,
711:             precomputed_beliefs=cache,
712:         )
713:         env.reset(seed=42, options={"question_idx": 0})
714:         for _ in range(env.total_steps):
715:             env.step(0)
716:             if env.truncated:
717:                 break
718: 
719:         mock_model.score.assert_not_called()
720: 
721:     def test_no_precomputed_backward_compat(
722:         self, sample_mc_question: MCQuestion
723:     ) -> None:
724:         """Env with precomputed_beliefs=None behaves identically to default."""
725:         corpus = sample_mc_question.option_profiles[:]
726:         model = TfIdfLikelihood(corpus_texts=corpus)
727:         questions = [sample_mc_question]
728: 
729:         # Default env (no precomputed_beliefs arg)
730:         env_default = TossupMCEnv(
731:             questions=questions, likelihood_model=model, K=4,
732:             belief_mode="from_scratch", beta=5.0,
733:         )
734:         env_default.reset(seed=42, options={"question_idx": 0})
735:         obs_default, _, _, _, _ = env_default.step(0)
736: 
737:         # Explicit None
738:         env_none = TossupMCEnv(
739:             questions=questions, likelihood_model=model, K=4,
740:             belief_mode="from_scratch", beta=5.0,
741:             precomputed_beliefs=None,
742:         )
743:         env_none.reset(seed=42, options={"question_idx": 0})
744:         obs_none, _, _, _, _ = env_none.step(0)
745: 
746:         np.testing.assert_array_equal(obs_default, obs_none)
747: 
748:     def test_precompute_beliefs_helper_shape(
749:         self, sample_mc_question: MCQuestion
750:     ) -> None:
751:         """precompute_beliefs returns correct keys and belief shapes."""
752:         corpus = sample_mc_question.option_profiles[:]
753:         model = TfIdfLikelihood(corpus_texts=corpus)
754:         questions = [sample_mc_question]
755: 
756:         cache = precompute_beliefs(
757:             questions=questions, likelihood_model=model,
758:             belief_mode="from_scratch", beta=5.0, K=4,
759:         )
760: 
761:         total_steps = len(sample_mc_question.run_indices)
762:         for s in range(total_steps):
763:             key = (0, s)
764:             assert key in cache, f"Missing key {key}"
765:             belief = cache[key]
766:             assert belief.shape == (4,), f"Expected (4,), got {belief.shape}"
767:             assert belief.dtype == np.float32, f"Expected float32, got {belief.dtype}"
768:             assert abs(belief.sum() - 1.0) < 1e-5, (
769:                 f"Belief should sum to ~1.0, got {belief.sum()}"
770:             )
771: 
772: 
773: class TestExpectedWinsRewardMode:
774:     """Tests for the expected_wins reward mode in TossupMCEnv."""
775: 
776:     def _make_env(self, sample_mc_question, survival: float):
777:         """Build an EW env with a fixed-survival opponent model."""
778:         from unittest.mock import MagicMock
779: 
780:         from models.likelihoods import TfIdfLikelihood
781: 
782:         corpus = sample_mc_question.option_profiles[:]
783:         model = TfIdfLikelihood(corpus_texts=corpus)
784:         opp = MagicMock()
785:         opp.prob_survive_to_step = MagicMock(return_value=survival)
786:         opp.prob_buzzed_before_step = MagicMock(return_value=1.0 - survival)
787:         return TossupMCEnv(
788:             questions=[sample_mc_question],
789:             likelihood_model=model,
790:             K=4,
791:             reward_mode="expected_wins",
792:             opponent_buzz_model=opp,
793:             ew_reward_correct=10.0,
794:             ew_reward_incorrect=-5.0,
795:             ew_opponent_expected_value=0.0,
796:             belief_mode="from_scratch",
797:             beta=5.0,
798:         )
799: 
800:     def test_survival_1_correct_gives_ew_correct(self, sample_mc_question):
801:         env = self._make_env(sample_mc_question, survival=1.0)
802:         env.reset(seed=42, options={"question_idx": 0})
803:         gold = sample_mc_question.gold_index
804:         _, reward, _, _, _ = env.step(gold + 1)
805:         assert abs(reward - 10.0) < 1e-9
806: 
807:     def test_survival_1_incorrect_gives_ew_incorrect(self, sample_mc_question):
808:         env = self._make_env(sample_mc_question, survival=1.0)
809:         env.reset(seed=42, options={"question_idx": 0})
810:         wrong = (sample_mc_question.gold_index + 1) % 4
811:         _, reward, _, _, _ = env.step(wrong + 1)
812:         assert abs(reward - (-5.0)) < 1e-9
813: 
814:     def test_survival_0_gives_opponent_value(self, sample_mc_question):
815:         env = self._make_env(sample_mc_question, survival=0.0)
816:         env.reset(seed=42, options={"question_idx": 0})
817:         _, reward, _, _, _ = env.step(1)
818:         assert abs(reward - 0.0) < 1e-9
819: 
820:     def test_non_ew_modes_unchanged(self, sample_tfidf_env):
821:         """Non-EW reward modes are unaffected by the new EW plumbing."""
822:         env = sample_tfidf_env
823:         obs, _ = env.reset(seed=42)
824:         _, reward, _, _, _ = env.step(0)
825:         assert isinstance(reward, float)
826: 
827: 
828: class TestVariableKEnv:
829:     """Tests for variable-K mode and action masks in TossupMCEnv."""
830: 
831:     def _make_mixed_k_questions(self, sample_mc_question):
832:         """Create a K=3 variant alongside the K=4 original."""
833:         from dataclasses import replace
834: 
835:         q3 = replace(
836:             sample_mc_question,
837:             qid="q_k3",
838:             options=sample_mc_question.options[:3],
839:             option_profiles=sample_mc_question.option_profiles[:3],
840:             option_answer_primary=sample_mc_question.option_answer_primary[:3],
841:             gold_index=0,
842:         )
843:         return [sample_mc_question, q3]
844: 
845:     def test_variable_k_obs_shape(self, sample_mc_question):
846:         from models.likelihoods import TfIdfLikelihood
847: 
848:         questions = self._make_mixed_k_questions(sample_mc_question)
849:         corpus = sample_mc_question.option_profiles[:]
850:         model = TfIdfLikelihood(corpus_texts=corpus)
851:         env = TossupMCEnv(
852:             questions=questions, likelihood_model=model,
853:             K=4, variable_K=True, max_K=4,
854:             reward_mode="simple", belief_mode="from_scratch",
855:         )
856:         obs, _ = env.reset(seed=42, options={"question_idx": 1})
857:         assert obs.shape == (4 + 6,)
858: 
859:     def test_action_mask_shape_and_validity(self, sample_mc_question):
860:         from models.likelihoods import TfIdfLikelihood
861: 
862:         questions = self._make_mixed_k_questions(sample_mc_question)
863:         corpus = sample_mc_question.option_profiles[:]
864:         model = TfIdfLikelihood(corpus_texts=corpus)
865:         env = TossupMCEnv(
866:             questions=questions, likelihood_model=model,
867:             K=4, variable_K=True, max_K=4,
868:             reward_mode="simple", belief_mode="from_scratch",
869:         )
870:         env.reset(seed=42, options={"question_idx": 1})
871:         mask = env.action_masks()
872:         assert mask.shape == (5,)
873:         assert mask[0]
874:         assert mask[1] and mask[2] and mask[3]
875:         assert not mask[4]
876: 
877:     def test_fixed_k_path_unchanged(self, sample_tfidf_env):
878:         """Fixed-K env (variable_K=False) behavior is unchanged."""
879:         env = sample_tfidf_env
880:         obs, _ = env.reset(seed=42)
881:         assert obs.shape == (4 + 6,)
882:         mask = env.action_masks()
883:         assert mask.shape == (5,)
884:         assert all(mask)
````

## File: configs/default.yaml
````yaml
  1: # Default configuration for qanta-buzzer
  2: # Adapted from qb-rl structure for T5-based quiz bowl agent
  3: 
  4: data:
  5:   csv_path: "questions.csv"  # Raw QANTA CSV with ||| separated clues
  6:   K: 4  # Default number of answer choices
  7:   distractor_strategy: "sbert_profile"  # sbert_profile | tfidf_profile | category_random | openai_profile
  8:   variable_K: false  # If true, sample K per question from [min_K, max_K]
  9:   min_K: 2
 10:   max_K: null  # Defaults to K when null
 11:   train_ratio: 0.7
 12:   val_ratio: 0.15
 13:   test_ratio: 0.15
 14:   max_questions: null  # Limit for testing (null = use all)
 15:   shuffle_seed: 42
 16: 
 17: answer_profiles:
 18:   max_tokens_per_profile: 2000  # Max tokens to use for answer profile
 19:   min_questions_per_answer: 1  # Minimum examples to build profile
 20:   leave_one_out: true  # Exclude current question from profile
 21: 
 22: likelihood:
 23:   model: "t5-large"  # Model for computing answer likelihoods (t5-small | t5-base | t5-large)
 24:   embedding_model: "all-MiniLM-L6-v2"  # For distractor generation
 25:   beta: 5.0  # Softmax temperature for belief distribution
 26:   cache_embeddings: true
 27:   cache_dir: "cache/embeddings"
 28:   batch_size: 16
 29:   max_length: 512  # Max input tokens for T5
 30: 
 31: environment:
 32:   reward_mode: "time_penalty"  # time_penalty | simple | expected_wins
 33:   seed: 13
 34:   wait_penalty: 0.05  # Tuned candidate from multi-seed smoke sweep
 35:   early_buzz_penalty: 0.2  # Tuned candidate from multi-seed smoke sweep
 36:   buzz_correct: 1.0  # Reward for correct answer
 37:   buzz_incorrect: -0.5  # Penalty for wrong answer
 38:   max_steps: 20  # Maximum clues to reveal
 39:   # Expected Wins opponent model (only used when reward_mode: expected_wins)
 40:   opponent_buzz_model:
 41:     type: "none"  # none | logistic | empirical
 42:   end_mode: "force_commit"  # force_commit | no_buzz
 43:   no_buzz_reward: 0.0  # Only used when end_mode == no_buzz
 44: 
 45: mc_guards:  # Anti-artifact guards from qb-rl
 46:   alias_edit_distance_threshold: 0.2  # Reject similar answer aliases
 47:   duplicate_token_overlap_threshold: 0.8  # Reject token-overlapping distractors
 48:   max_length_ratio: 3.0  # Reject distractors much longer than answer
 49: 
 50: bayesian:  # Bayesian buzzer sweep parameters (from qb-rl)
 51:   threshold_sweep: [0.5, 0.6, 0.7, 0.8, 0.9]
 52:   alpha: 10.0  # Sigmoid steepness for confidence proxy
 53: 
 54: ppo:  # PPO hyperparameters (for future use)
 55:   seed: 13
 56:   total_timesteps: 100000
 57:   learning_rate: 3e-4
 58:   n_steps: 128
 59:   batch_size: 32
 60:   n_epochs: 4
 61:   gamma: 0.99
 62:   gae_lambda: 0.95
 63:   clip_ratio: 0.2
 64:   value_coef: 0.5
 65:   entropy_coef: 0.01
 66:   max_grad_norm: 0.5
 67:   target_kl: 0.03
 68:   policy_kwargs:
 69:     net_arch: [64, 64]  # MLP architecture for belief-based policy
 70: 
 71: evaluation:
 72:   metrics:
 73:     - accuracy
 74:     - reward
 75:     - buzz_position
 76:     - calibration  # ECE and Brier score
 77:     - per_category
 78:   compute_sq: true  # S_q scoring metric
 79:   run_choices_only: true  # Control: model sees only choices, no clues
 80:   run_shuffle: true  # Control: shuffle clue order
 81:   bootstrap_ci_samples: 1000  # Bootstrap confidence intervals
 82:   save_predictions: true
 83:   prediction_dir: "results/predictions"
 84: 
 85: # DSPy integration (optional, offline-first)
 86: # Activated by setting likelihood.model: dspy — no separate enable flag.
 87: dspy:
 88:   model: "openai/gpt-4o-mini"  # DSPy LM identifier
 89:   optimizer: "BootstrapFewShot"  # BootstrapFewShot | MIPROv2
 90:   cache_dir: "cache/dspy"
 91:   max_examples: 50
 92: 
 93: # Supervised warm-start settings (for T5 policy)
 94: supervised:
 95:   epochs: 10
 96:   batch_size: 8
 97:   gradient_accumulation_steps: 4  # Effective batch = 32
 98:   learning_rate: 1e-4
 99:   warmup_steps: 500
100:   eval_steps: 100
101:   save_steps: 500
102:   save_total_limit: 3
103:   checkpoint_dir: "checkpoints/supervised"
````

## File: models/likelihoods.py
````python
  1: """
  2: Likelihood Model Interface
  3: 
  4: Abstract base class for likelihood models that score answer options against
  5: revealed clue text. Concrete implementations (TF-IDF, SBERT, T5) inherit
  6: from ``LikelihoodModel`` and implement ``score()`` and ``_embed_batch()``.
  7: 
  8: The ``score()`` method returns **raw similarity scores**, not probabilities.
  9: The environment applies softmax with a configurable temperature (beta) to
 10: convert scores into a belief distribution.
 11: 
 12: Embedding caching is built into the base class: texts are hashed via SHA-256
 13: and cached as float32 numpy arrays, so repeated calls with the same text
 14: skip recomputation.
 15: 
 16: Ported from qb-rl reference implementation (models/likelihoods.py lines 1-38).
 17: """
 18: 
 19: from __future__ import annotations
 20: 
 21: import hashlib
 22: import os
 23: from abc import ABC, abstractmethod
 24: from pathlib import Path
 25: from typing import TYPE_CHECKING, Any
 26: 
 27: import numpy as np
 28: 
 29: if TYPE_CHECKING:
 30:     import torch
 31: 
 32: 
 33: def _text_key(text: str) -> str:
 34:     """Compute a SHA-256 hash key for embedding cache lookups.
 35: 
 36:     Parameters
 37:     ----------
 38:     text : str
 39:         Input text to hash.
 40: 
 41:     Returns
 42:     -------
 43:     str
 44:         64-character hexadecimal SHA-256 digest.
 45: 
 46:     Examples
 47:     --------
 48:     >>> key = _text_key("hello world")
 49:     >>> len(key)
 50:     64
 51:     >>> _text_key("hello world") == _text_key("hello world")
 52:     True
 53:     """
 54:     return hashlib.sha256(text.encode("utf-8")).hexdigest()
 55: 
 56: 
 57: def _best_torch_device() -> "torch.device":
 58:     """Select the best available accelerator: CUDA > MPS > CPU."""
 59:     import torch
 60: 
 61:     if torch.cuda.is_available():
 62:         return torch.device("cuda")
 63:     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
 64:         return torch.device("mps")
 65:     return torch.device("cpu")
 66: 
 67: 
 68: class LikelihoodModel(ABC):
 69:     """Abstract base class for likelihood models.
 70: 
 71:     Likelihood models score how well each answer option matches a given
 72:     clue prefix. The environment uses these scores (via softmax) to compute
 73:     belief distributions over answer options.
 74: 
 75:     Subclasses must implement:
 76:         - ``score(clue_prefix, option_profiles) -> np.ndarray``
 77:         - ``_embed_batch(texts) -> np.ndarray``
 78: 
 79:     The base class provides ``embed_and_cache()`` which handles caching of
 80:     text embeddings via SHA-256 content hashing.
 81: 
 82:     Attributes
 83:     ----------
 84:     embedding_cache : dict[str, np.ndarray]
 85:         Maps SHA-256 text hashes to float32 embedding vectors.
 86:     """
 87: 
 88:     def __init__(self) -> None:
 89:         self.embedding_cache: dict[str, np.ndarray] = {}
 90: 
 91:     @property
 92:     def cache_memory_bytes(self) -> int:
 93:         """Approximate memory used by the embedding cache in bytes."""
 94:         return sum(v.nbytes for v in self.embedding_cache.values())
 95: 
 96:     @abstractmethod
 97:     def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
 98:         """Return raw similarity scores for each answer option.
 99: 
100:         The caller (environment) converts these to probabilities via
101:         softmax with a beta temperature parameter. Higher scores indicate
102:         stronger match between clue and option.
103: 
104:         Parameters
105:         ----------
106:         clue_prefix : str
107:             Clue text revealed so far (concatenation of clues up to current step).
108:         option_profiles : list[str]
109:             Answer profile text for each of the K answer options.
110: 
111:         Returns
112:         -------
113:         np.ndarray
114:             Raw similarity scores of shape (K,) where K = len(option_profiles).
115:         """
116: 
117:     def embed_and_cache(self, texts: list[str]) -> np.ndarray:
118:         """Embed texts, using cache for previously seen inputs.
119: 
120:         Texts are identified by their SHA-256 hash. Only unseen texts
121:         are passed to ``_embed_batch()`` for actual computation; cached
122:         results are reused.
123: 
124:         Parameters
125:         ----------
126:         texts : list[str]
127:             Texts to embed.
128: 
129:         Returns
130:         -------
131:         np.ndarray
132:             Stacked embeddings of shape (len(texts), embed_dim), dtype float32.
133:         """
134:         missing = [text for text in texts if _text_key(text) not in self.embedding_cache]
135:         if missing:
136:             new_embeddings = self._embed_batch(missing)
137:             for text, emb in zip(missing, new_embeddings):
138:                 self.embedding_cache[_text_key(text)] = emb.astype(np.float32)
139:         return np.stack([self.embedding_cache[_text_key(text)] for text in texts])
140: 
141:     def precompute_embeddings(
142:         self,
143:         texts: list[str],
144:         batch_size: int = 64,
145:         desc: str = "Pre-computing embeddings",
146:     ) -> None:
147:         """Bulk pre-embed texts into cache, processing in batches.
148: 
149:         Call this before running agents so that all subsequent ``score()``
150:         calls are pure cache lookups (numpy dot products).  Duplicate and
151:         already-cached texts are skipped automatically.
152: 
153:         Parameters
154:         ----------
155:         texts : list[str]
156:             All texts to embed (clue prefixes, option profiles, fragments).
157:         batch_size : int
158:             Number of texts per ``_embed_batch`` call.
159:         desc : str
160:             tqdm progress-bar description.
161:         """
162:         from tqdm import tqdm
163: 
164:         unique = [t for t in dict.fromkeys(texts) if _text_key(t) not in self.embedding_cache]
165:         if not unique:
166:             return
167:         for i in tqdm(range(0, len(unique), batch_size), desc=desc,
168:                        total=(len(unique) + batch_size - 1) // batch_size):
169:             batch = unique[i : i + batch_size]
170:             embeddings = self._embed_batch(batch)
171:             for text, emb in zip(batch, embeddings):
172:                 self.embedding_cache[_text_key(text)] = emb.astype(np.float32)
173: 
174:     def save_cache(self, path: str | Path) -> int:
175:         """Persist embedding_cache to disk as compressed ``.npz``.
176: 
177:         Creates parent directories if needed. Keys are SHA-256 hex
178:         strings (valid Python identifiers), values are float32 arrays.
179: 
180:         Parameters
181:         ----------
182:         path : str or Path
183:             Destination file path (should end with ``.npz``).
184: 
185:         Returns
186:         -------
187:         int
188:             Number of cache entries saved.
189:         """
190:         p = Path(path)
191:         p.parent.mkdir(parents=True, exist_ok=True)
192:         np.savez_compressed(p, **self.embedding_cache)
193:         return len(self.embedding_cache)
194: 
195:     def load_cache(self, path: str | Path) -> int:
196:         """Load embedding_cache from a ``.npz`` file on disk.
197: 
198:         Merges loaded entries into the existing cache **without**
199:         overwriting keys that are already present (existing keys win).
200:         If the file does not exist, silently returns 0 (cold-start).
201: 
202:         Parameters
203:         ----------
204:         path : str or Path
205:             Path to ``.npz`` file previously written by ``save_cache``.
206: 
207:         Returns
208:         -------
209:         int
210:             Number of *new* entries added to the cache.
211:         """
212:         p = Path(path)
213:         if not p.exists():
214:             return 0
215:         with np.load(p) as data:
216:             loaded = 0
217:             for key in data.files:
218:                 if key not in self.embedding_cache:
219:                     self.embedding_cache[key] = data[key].astype(np.float32)
220:                     loaded += 1
221:             return loaded
222: 
223:     @abstractmethod
224:     def _embed_batch(self, texts: list[str]) -> np.ndarray:
225:         """Embed a batch of texts. Subclasses must implement.
226: 
227:         Parameters
228:         ----------
229:         texts : list[str]
230:             Texts to embed (guaranteed non-empty, all cache misses).
231: 
232:         Returns
233:         -------
234:         np.ndarray
235:             Embeddings of shape (len(texts), embed_dim), dtype float32.
236:         """
237:         raise NotImplementedError
238: 
239: 
240: class TfIdfLikelihood(LikelihoodModel):
241:     """TF-IDF based likelihood model using cosine similarity.
242: 
243:     Uses scikit-learn's ``TfidfVectorizer`` to learn vocabulary and IDF weights
244:     from a corpus, then scores clue-option similarity via cosine distance in the
245:     TF-IDF vector space.
246: 
247:     The model **must** be ``fit()`` on a corpus before calling ``score()`` or
248:     ``_embed_batch()``. Calling these methods on an unfitted model raises
249:     ``RuntimeError``.
250: 
251:     This is the fast, interpretable baseline: keyword overlap drives similarity.
252:     It works well when clues contain distinctive vocabulary but misses semantic
253:     relationships (e.g., "first president" vs "George Washington").
254: 
255:     Parameters
256:     ----------
257:     corpus_texts : list[str] or None
258:         If provided, ``fit()`` is called immediately on these texts.
259: 
260:     Attributes
261:     ----------
262:     vectorizer : TfidfVectorizer
263:         Scikit-learn vectorizer with English stop words removed.
264:     _is_fit : bool
265:         Whether the vectorizer has been fit on a corpus.
266: 
267:     Examples
268:     --------
269:     >>> corpus = ["George Washington was the first president",
270:     ...           "Abraham Lincoln freed the slaves"]
271:     >>> model = TfIdfLikelihood(corpus_texts=corpus)
272:     >>> scores = model.score("first president", ["Washington", "Lincoln"])
273:     >>> scores.shape
274:     (2,)
275:     """
276: 
277:     def __init__(self, corpus_texts: list[str] | None = None) -> None:
278:         super().__init__()
279:         from sklearn.feature_extraction.text import TfidfVectorizer
280: 
281:         self.vectorizer = TfidfVectorizer(stop_words="english")
282:         self._is_fit = False
283:         if corpus_texts:
284:             self.fit(corpus_texts)
285: 
286:     def save_cache(self, path: str | Path) -> int:
287:         """No-op: TF-IDF embeddings are vocabulary-specific and not portable.
288: 
289:         TF-IDF vectors depend on the fitted vocabulary, which changes
290:         between ``fit()`` calls. Persisting them would produce wrong
291:         results if the vocabulary differs.
292: 
293:         Returns
294:         -------
295:         int
296:             Always 0.
297:         """
298:         return 0
299: 
300:     def fit(self, corpus_texts: list[str]) -> "TfIdfLikelihood":
301:         """Learn vocabulary and IDF weights from a text corpus.
302: 
303:         Parameters
304:         ----------
305:         corpus_texts : list[str]
306:             Corpus of documents to learn from. Should include answer profiles,
307:             clue texts, or both to capture domain vocabulary.
308: 
309:         Returns
310:         -------
311:         TfIdfLikelihood
312:             Self, for method chaining.
313:         """
314:         self.vectorizer.fit(corpus_texts)
315:         self._is_fit = True
316:         return self
317: 
318:     def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
319:         """Score each option against the clue using TF-IDF cosine similarity.
320: 
321:         Uses ``embed_and_cache()`` to embed both the clue and options, so
322:         repeated calls with the same texts skip vectorizer.transform().
323:         Since ``_embed_batch()`` returns L2-normalized vectors, the dot
324:         product equals cosine similarity.
325: 
326:         Parameters
327:         ----------
328:         clue_prefix : str
329:             Clue text revealed so far.
330:         option_profiles : list[str]
331:             Answer profile text for each of the K answer options.
332: 
333:         Returns
334:         -------
335:         np.ndarray
336:             Cosine similarity scores of shape (K,), dtype float32.
337:             Values in [-1, 1] but typically [0, 1] for TF-IDF.
338: 
339:         Raises
340:         ------
341:         RuntimeError
342:             If called before ``fit()``.
343:         """
344:         if not self._is_fit:
345:             raise RuntimeError("TfIdfLikelihood must be fit() before score().")
346:         clue_emb = self.embed_and_cache([clue_prefix])[0]
347:         option_embs = self.embed_and_cache(option_profiles)
348:         sims = option_embs @ clue_emb
349:         return sims.astype(np.float32)
350: 
351:     def _embed_batch(self, texts: list[str]) -> np.ndarray:
352:         """Embed texts as dense, L2-normalized TF-IDF vectors.
353: 
354:         Row-wise L2 normalization ensures that dot product between any
355:         two embedding vectors equals their cosine similarity, matching
356:         the convention used by SBERT and T5 likelihood models.
357: 
358:         Parameters
359:         ----------
360:         texts : list[str]
361:             Texts to embed (guaranteed non-empty, all cache misses).
362: 
363:         Returns
364:         -------
365:         np.ndarray
366:             L2-normalized dense TF-IDF matrix of shape
367:             (len(texts), vocab_size), dtype float32.
368: 
369:         Raises
370:         ------
371:         RuntimeError
372:             If called before ``fit()``.
373:         """
374:         if not self._is_fit:
375:             raise RuntimeError("TfIdfLikelihood must be fit() before embedding.")
376:         mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
377:         norms = np.linalg.norm(mat, axis=1, keepdims=True)
378:         norms[norms == 0] = 1.0  # avoid division by zero for empty docs
379:         return mat / norms
380: 
381: 
382: class SBERTLikelihood(LikelihoodModel):
383:     """Sentence-BERT likelihood model using semantic embeddings.
384: 
385:     Uses a ``SentenceTransformer`` model to compute dense, L2-normalized
386:     embeddings. Cosine similarity is computed as a simple dot product since
387:     embeddings are pre-normalized (``normalize_embeddings=True``).
388: 
389:     Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
390:     caching of embeddings via SHA-256 content hashing. The first call to
391:     ``score()`` computes and caches all embeddings; subsequent calls with the
392:     same texts are fast cache lookups.
393: 
394:     Compared to TF-IDF, SBERT captures semantic similarity (e.g., "first
395:     president" and "George Washington" score highly even without word overlap)
396:     but is slower due to the neural encoder.
397: 
398:     Parameters
399:     ----------
400:     model_name : str
401:         HuggingFace model identifier for ``SentenceTransformer``.
402:         Default is ``"all-MiniLM-L6-v2"`` (22M params, 384-dim embeddings).
403:         First run downloads the model (~80MB) from HuggingFace.
404: 
405:     Attributes
406:     ----------
407:     model_name : str
408:         The SentenceTransformer model name.
409:     encoder : SentenceTransformer
410:         The loaded sentence transformer model.
411: 
412:     Examples
413:     --------
414:     >>> model = SBERTLikelihood()  # downloads model on first run
415:     >>> scores = model.score("first president", ["Washington", "Lincoln"])
416:     >>> scores.shape
417:     (2,)
418:     """
419: 
420:     def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
421:         super().__init__()
422:         from sentence_transformers import SentenceTransformer
423: 
424:         self.model_name = model_name
425:         self.encoder = SentenceTransformer(model_name)
426: 
427:     def _embed_batch(self, texts: list[str]) -> np.ndarray:
428:         """Embed texts using the SentenceTransformer encoder.
429: 
430:         Embeddings are L2-normalized so that cosine similarity can be computed
431:         as a simple dot product (avoiding the division by norms).
432: 
433:         Parameters
434:         ----------
435:         texts : list[str]
436:             Texts to embed (guaranteed non-empty, all cache misses).
437: 
438:         Returns
439:         -------
440:         np.ndarray
441:             Normalized embeddings of shape (len(texts), embed_dim), dtype float32.
442:         """
443:         return self.encoder.encode(
444:             texts, convert_to_numpy=True, normalize_embeddings=True
445:         ).astype(np.float32)
446: 
447:     def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
448:         """Score each option using semantic cosine similarity.
449: 
450:         Computes dot product between the clue embedding and each option
451:         embedding. Since embeddings are L2-normalized, dot product equals
452:         cosine similarity.
453: 
454:         Parameters
455:         ----------
456:         clue_prefix : str
457:             Clue text revealed so far.
458:         option_profiles : list[str]
459:             Answer profile text for each of the K answer options.
460: 
461:         Returns
462:         -------
463:         np.ndarray
464:             Cosine similarity scores of shape (K,), dtype float32.
465:             Values in [-1, 1].
466:         """
467:         clue_emb = self.embed_and_cache([clue_prefix])[0]
468:         option_embs = self.embed_and_cache(option_profiles)
469:         sims = option_embs @ clue_emb
470:         return sims.astype(np.float32)
471: 
472: 
473: class OpenAILikelihood(LikelihoodModel):
474:     """OpenAI embedding likelihood model using normalized embedding similarity.
475: 
476:     This path is optional and only activates when explicitly selected in config.
477:     It requires both the ``openai`` Python package and ``OPENAI_API_KEY`` to be
478:     available at runtime.
479:     """
480: 
481:     def __init__(
482:         self,
483:         model: str = "text-embedding-3-small",
484:         api_key: str | None = None,
485:     ) -> None:
486:         super().__init__()
487: 
488:         resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
489:         if not resolved_api_key:
490:             raise RuntimeError(
491:                 "OpenAI likelihood requires OPENAI_API_KEY to be set."
492:             )
493: 
494:         try:
495:             from openai import OpenAI
496:         except ImportError as exc:
497:             raise ImportError(
498:                 "OpenAI likelihood requires the openai package. "
499:                 "Install it with: pip install -e .[openai] or pip install openai."
500:             ) from exc
501: 
502:         self.model = model
503:         self.client = OpenAI(api_key=resolved_api_key)
504: 
505:     def _embed_batch(self, texts: list[str]) -> np.ndarray:
506:         """Embed texts via the OpenAI embeddings API and L2-normalize them."""
507:         response = self.client.embeddings.create(model=self.model, input=texts)
508:         vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
509:         embeddings = np.stack(vectors)
510:         norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
511:         norms[norms == 0] = 1.0
512:         return (embeddings / norms).astype(np.float32)
513: 
514:     def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
515:         """Score each option using cosine similarity over normalized embeddings."""
516:         clue_emb = self.embed_and_cache([clue_prefix])[0]
517:         option_embs = self.embed_and_cache(option_profiles)
518:         sims = option_embs @ clue_emb
519:         return sims.astype(np.float32)
520: 
521: 
522: class T5Likelihood(LikelihoodModel):
523:     """T5 encoder likelihood model using mean-pooled semantic embeddings.
524: 
525:     Uses ``T5EncoderModel`` (not full ``T5ForConditionalGeneration``) for 2x
526:     faster inference and half the memory. Embeddings are mean-pooled over
527:     sequence length with attention mask weighting to handle padding correctly.
528: 
529:     Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
530:     caching of embeddings via SHA-256 content hashing. The first call to
531:     ``score()`` computes and caches all embeddings; subsequent calls with the
532:     same texts are fast cache lookups.
533: 
534:     Compared to SBERT, T5 captures deeper semantic relationships via its
535:     encoder-decoder pre-training on massive text corpora. This is the novel
536:     contribution: using T5 as a likelihood model rather than just as a policy
537:     encoder.
538: 
539:     Parameters
540:     ----------
541:     model_name : str
542:         HuggingFace T5 model identifier. Default is ``"t5-base"``
543:         (220M params). Options:
544: 
545:         - ``"t5-small"`` (60M params) -- fastest, lowest quality
546:         - ``"t5-base"`` (220M params) -- balanced (recommended)
547:         - ``"t5-large"`` (770M params) -- best quality, requires 8GB GPU VRAM
548: 
549:         First run downloads the model from HuggingFace (~850MB for t5-base).
550: 
551:     Attributes
552:     ----------
553:     model_name : str
554:         The T5 model identifier.
555:     encoder : T5EncoderModel
556:         Pre-trained T5 encoder loaded from HuggingFace.
557:     tokenizer : T5TokenizerFast
558:         Fast T5 tokenizer for text preprocessing.
559:     device : torch.device
560:         Computation device (cuda if available, else cpu).
561: 
562:     Examples
563:     --------
564:     >>> model = T5Likelihood(model_name="t5-small")
565:     >>> scores = model.score("first president", ["Washington", "Einstein"])
566:     >>> scores.shape
567:     (2,)
568:     """
569: 
570:     def __init__(self, model_name: str = "t5-base") -> None:
571:         super().__init__()
572:         import torch
573:         from transformers import T5EncoderModel, T5TokenizerFast
574: 
575:         self.model_name = model_name
576:         self.encoder = T5EncoderModel.from_pretrained(model_name)
577:         self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
578:         self.device = _best_torch_device()
579:         self.encoder.to(self.device)
580:         self.encoder.eval()
581: 
582:     def _embed_batch(self, texts: list[str]) -> np.ndarray:
583:         """Embed texts using T5 encoder with attention-masked mean pooling.
584: 
585:         Mean pooling uses the attention mask to exclude padding tokens from the
586:         average, ensuring correct semantic embeddings when sequences have
587:         different lengths. Embeddings are L2-normalized so that cosine
588:         similarity can be computed as a simple dot product.
589: 
590:         Parameters
591:         ----------
592:         texts : list[str]
593:             Texts to embed (guaranteed non-empty, all cache misses).
594: 
595:         Returns
596:         -------
597:         np.ndarray
598:             L2-normalized embeddings of shape (len(texts), hidden_dim),
599:             dtype float32. Hidden dim is 512 (t5-small), 768 (t5-base),
600:             or 1024 (t5-large).
601: 
602:         Notes
603:         -----
604:         Tensors are detached and moved to CPU immediately after computation
605:         to prevent GPU memory leaks when called repeatedly during episodes.
606:         """
607:         import torch
608: 
609:         with torch.no_grad():
610:             encoded = self.tokenizer(
611:                 texts,
612:                 padding=True,
613:                 truncation=True,
614:                 max_length=512,
615:                 return_tensors="pt",
616:             ).to(self.device)
617: 
618:             outputs = self.encoder(**encoded)
619:             last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
620: 
621:             # Mean pooling over sequence length with attention mask
622:             mask = encoded.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
623:             masked_hidden = last_hidden * mask
624:             sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_dim)
625:             mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
626:             mean_pooled = sum_hidden / mask_sum  # (batch, hidden_dim)
627: 
628:             # L2 normalize for cosine similarity via dot product
629:             embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
630: 
631:             # Detach and move to CPU to prevent GPU memory leak
632:             embeddings = embeddings.detach().cpu().numpy().astype(np.float32)
633: 
634:         return embeddings
635: 
636:     def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
637:         """Score each option using T5 semantic cosine similarity.
638: 
639:         Computes dot product between the clue embedding and each option
640:         embedding. Since embeddings are L2-normalized, dot product equals
641:         cosine similarity.
642: 
643:         Parameters
644:         ----------
645:         clue_prefix : str
646:             Clue text revealed so far.
647:         option_profiles : list[str]
648:             Answer profile text for each of the K answer options.
649: 
650:         Returns
651:         -------
652:         np.ndarray
653:             Cosine similarity scores of shape (K,), dtype float32.
654:             Values in [-1, 1].
655:         """
656:         clue_emb = self.embed_and_cache([clue_prefix])[0]
657:         option_embs = self.embed_and_cache(option_profiles)
658:         sims = option_embs @ clue_emb
659:         return sims.astype(np.float32)
660: 
661: 
662: def build_likelihood_from_config(
663:     config: dict[str, Any], corpus_texts: list[str] | None = None
664: ) -> LikelihoodModel:
665:     """Construct a likelihood model from YAML configuration.
666: 
667:     Factory function that reads the ``likelihood`` section of the config dict
668:     and instantiates the appropriate ``LikelihoodModel`` subclass.
669: 
670:     Parameters
671:     ----------
672:     config : dict[str, Any]
673:         Full YAML config dict. Must contain a ``"likelihood"`` key with at
674:         least a ``"model"`` field specifying the model type.
675: 
676:         Supported model types:
677:         - ``"tfidf"``: TF-IDF cosine similarity (requires ``corpus_texts``)
678:         - ``"sbert"``: Sentence-BERT semantic similarity
679:         - ``"openai"``: OpenAI embedding similarity
680:         - ``"t5"`` / ``"t5-small"`` / ``"t5-base"`` / ``"t5-large"``:
681:           T5 encoder semantic similarity
682: 
683:         Optional config keys:
684:         - ``"sbert_name"`` or ``"embedding_model"``: SentenceTransformer model
685:           name (default: ``"all-MiniLM-L6-v2"``)
686:         - ``"openai_model"``: OpenAI embedding model name
687:           (default: ``"text-embedding-3-small"``)
688:         - ``"t5_name"``: T5 model name (default: ``"t5-base"``)
689: 
690:     corpus_texts : list[str] or None
691:         Text corpus for TF-IDF fitting. Required when ``model == "tfidf"``,
692:         ignored for other models.
693: 
694:     Returns
695:     -------
696:     LikelihoodModel
697:         An instantiated and ready-to-use likelihood model.
698: 
699:     Raises
700:     ------
701:     ValueError
702:         If ``model`` is ``"tfidf"`` and ``corpus_texts`` is None.
703:         If ``model`` is not a recognized model type.
704: 
705:     Examples
706:     --------
707:     >>> from qb_data.config import load_config
708:     >>> config = load_config("configs/default.yaml")
709:     >>> model = build_likelihood_from_config(config, corpus_texts=my_corpus)
710:     >>> scores = model.score("first president", ["Washington", "Lincoln"])
711:     """
712:     cfg = config["likelihood"]
713:     model_name = cfg.get("model", "sbert")
714: 
715:     if model_name == "tfidf":
716:         if not corpus_texts:
717:             raise ValueError("TF-IDF likelihood requires corpus_texts.")
718:         return TfIdfLikelihood(corpus_texts=corpus_texts)
719: 
720:     if model_name == "sbert":
721:         # Support both "sbert_name" (qb-rl convention) and
722:         # "embedding_model" (qanta-buzzer default.yaml convention)
723:         sbert_name = cfg.get("sbert_name", cfg.get("embedding_model", "all-MiniLM-L6-v2"))
724:         return SBERTLikelihood(model_name=sbert_name)
725: 
726:     if model_name == "openai":
727:         return OpenAILikelihood(
728:             model=cfg.get("openai_model", "text-embedding-3-small"),
729:         )
730: 
731:     if model_name == "t5":
732:         t5_name = cfg.get("t5_name", "t5-base")
733:         return T5Likelihood(model_name=t5_name)
734: 
735:     if isinstance(model_name, str) and model_name.startswith("t5"):
736:         t5_name = model_name
737:         return T5Likelihood(model_name=t5_name)
738: 
739:     if model_name == "dspy":
740:         try:
741:             from models.dspy_likelihood import DSPyLikelihood
742:         except ImportError as exc:
743:             raise ImportError(
744:                 "DSPy likelihood requires the dspy package. "
745:                 "Install with: pip install -e '.[dspy]'"
746:             ) from exc
747:         dspy_cfg = config.get("dspy", {})
748:         cache_dir = dspy_cfg.get("cache_dir")
749:         fingerprint = dspy_cfg.get("program_fingerprint", "default")
750: 
751:         def _placeholder_scorer(clue: str, options: list[str]) -> list[float]:
752:             return [1.0 / max(1, len(options))] * len(options)
753: 
754:         return DSPyLikelihood(
755:             scorer=_placeholder_scorer,
756:             program_fingerprint=fingerprint,
757:             cache_dir=cache_dir,
758:         )
759: 
760:     raise ValueError(f"Unknown likelihood model: {model_name}")
````

## File: qb_env/tossup_env.py
````python
  1: """
  2: Gymnasium-compliant POMDP Environment for Quiz Bowl
  3: 
  4: Implements a tossup question environment where clues are revealed incrementally.
  5: At each step the agent observes a belief-based feature vector and chooses either
  6: to WAIT (action 0, reveals next clue) or to BUZZ with a specific answer option
  7: (actions 1..K, ends the episode).
  8: 
  9: The environment computes beliefs over K answer options using a pluggable
 10: LikelihoodModel and converts them to observations via extract_belief_features.
 11: 
 12: Ported from qb-rl reference implementation (qb_env/tossup_env.py) and adapted
 13: for the unified qanta-buzzer codebase.
 14: """
 15: 
 16: from __future__ import annotations
 17: 
 18: import random
 19: from typing import Any
 20: 
 21: import gymnasium as gym
 22: import numpy as np
 23: from gymnasium import spaces
 24: 
 25: from models.features import extract_belief_features
 26: from models.likelihoods import LikelihoodModel
 27: from qb_data.mc_builder import MCQuestion
 28: 
 29: 
 30: def _softmax(scores: np.ndarray, beta: float) -> np.ndarray:
 31:     """Temperature-scaled softmax with numerical stability.
 32: 
 33:     Parameters
 34:     ----------
 35:     scores : np.ndarray
 36:         Raw similarity scores of shape (K,).
 37:     beta : float
 38:         Temperature parameter. Higher values produce sharper distributions.
 39: 
 40:     Returns
 41:     -------
 42:     np.ndarray
 43:         Probability distribution of shape (K,), dtype float32.
 44:     """
 45:     stable = scores - np.max(scores)
 46:     probs = np.exp(beta * stable)
 47:     probs_sum = np.sum(probs)
 48:     if probs_sum <= 0:
 49:         return np.ones_like(scores, dtype=np.float32) / len(scores)
 50:     return (probs / probs_sum).astype(np.float32)
 51: 
 52: 
 53: def precompute_beliefs(
 54:     questions: list[MCQuestion],
 55:     likelihood_model: LikelihoodModel,
 56:     belief_mode: str = "from_scratch",
 57:     beta: float = 5.0,
 58:     K: int = 4,
 59: ) -> dict[tuple[int, int], np.ndarray]:
 60:     """Precompute belief trajectories for all questions and steps.
 61: 
 62:     Iterates over each question and each step index, computing the belief
 63:     using the same logic as ``TossupMCEnv._compute_belief``. The result is
 64:     a dict keyed by ``(question_index, step_idx)`` for O(1) lookup during
 65:     training rollouts.
 66: 
 67:     Parameters
 68:     ----------
 69:     questions : list[MCQuestion]
 70:         Pool of questions to precompute beliefs for.
 71:     likelihood_model : LikelihoodModel
 72:         Model that scores clue text against answer option profiles.
 73:     belief_mode : str
 74:         One of ``"from_scratch"``, ``"sequential_bayes"``.
 75:     beta : float
 76:         Softmax temperature for converting raw scores to probabilities.
 77:     K : int
 78:         Number of answer options per question.
 79: 
 80:     Returns
 81:     -------
 82:     dict[tuple[int, int], np.ndarray]
 83:         Maps ``(question_index, step_idx)`` to belief vectors of shape
 84:         ``(K,)`` with dtype float32. Each belief sums to ~1.0.
 85:     """
 86:     cache: dict[tuple[int, int], np.ndarray] = {}
 87: 
 88:     for q_idx, question in enumerate(questions):
 89:         num_steps = len(question.run_indices)
 90:         belief = np.ones(K, dtype=np.float32) / K
 91: 
 92:         for step_idx in range(num_steps):
 93:             if belief_mode == "from_scratch":
 94:                 prefix = question.cumulative_prefixes[step_idx]
 95:                 scores = likelihood_model.score(prefix, question.option_profiles)
 96:                 belief = _softmax(scores, beta)
 97: 
 98:             elif belief_mode == "sequential_bayes":
 99:                 idx = question.run_indices[step_idx]
100:                 prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
101:                 frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
102:                 scores = likelihood_model.score(frag, question.option_profiles)
103:                 likelihood = _softmax(scores, beta)
104:                 posterior = belief * likelihood
105:                 denom = posterior.sum()
106:                 if denom <= 0:
107:                     belief = np.ones(K, dtype=np.float32) / K
108:                 else:
109:                     belief = (posterior / denom).astype(np.float32)
110: 
111:             else:
112:                 raise ValueError(f"Unknown belief_mode: {belief_mode}")
113: 
114:             cache[(q_idx, step_idx)] = belief.copy()
115: 
116:     return cache
117: 
118: 
119: class TossupMCEnv(gym.Env[np.ndarray, int]):
120:     """Gymnasium environment for quiz bowl tossup questions with MC options.
121: 
122:     Models quiz bowl as a POMDP where clues are revealed incrementally.
123:     The agent maintains a belief distribution over K answer options, updated
124:     at each step by a likelihood model. The agent decides when to buzz and
125:     which answer to select.
126: 
127:     Action Space
128:     ------------
129:     Discrete(K + 1):
130:         - 0: WAIT -- reveal the next clue and update belief
131:         - 1..K: BUZZ with answer option (i-1), ending the episode
132: 
133:     Observation Space
134:     -----------------
135:     Box(K + 6,):
136:         Belief features: [belief[0..K-1], top_p, margin, entropy,
137:         stability, progress, clue_idx_norm].
138:         See ``models.features.extract_belief_features`` for details.
139: 
140:     Reward Modes
141:     ------------
142:     ``time_penalty`` (default):
143:         -wait_penalty per WAIT step; +buzz_correct for correct buzz,
144:         +buzz_incorrect (negative) for wrong buzz.
145:     ``simple``:
146:         +1.0 for correct buzz, -1.0 for incorrect buzz, no WAIT penalty.
147:     ``human_grounded``:
148:         0.0 if the agent buzzes after the sampled human buzz position;
149:         otherwise +buzz_correct/-buzz_incorrect for correct/incorrect.
150: 
151:     Belief Modes
152:     ------------
153:     ``from_scratch``:
154:         Recompute belief from all clues seen so far via cumulative_prefixes.
155:     ``sequential_bayes``:
156:         Bayesian update: multiply prior belief by likelihood of new clue
157:         fragment, then normalize.
158: 
159:     Parameters
160:     ----------
161:     questions : list[MCQuestion]
162:         Pool of questions to sample from. Must be non-empty.
163:     likelihood_model : LikelihoodModel
164:         Model that scores clue text against answer option profiles.
165:     K : int
166:         Number of answer options per question. Must be >= 2.
167:     reward_mode : str
168:         One of ``"time_penalty"``, ``"simple"``, ``"human_grounded"``.
169:     wait_penalty : float
170:         Per-step penalty when reward_mode is ``"time_penalty"``.
171:     buzz_correct : float
172:         Reward for buzzing with the correct answer.
173:     buzz_incorrect : float
174:         Reward (typically negative) for buzzing with an incorrect answer.
175:     belief_mode : str
176:         One of ``"from_scratch"``, ``"sequential_bayes"``.
177:     beta : float
178:         Softmax temperature for converting raw scores to probabilities.
179:         Higher values produce sharper distributions.
180:     end_mode : str
181:         Horizon behavior when clues are exhausted:
182:         ``"force_commit"`` (legacy forced answer) or ``"no_buzz"``.
183:     no_buzz_reward : float
184:         Reward added at horizon when ``end_mode == "no_buzz"``.
185:     seed : int
186:         Random seed for question sampling and human buzz simulation.
187:     """
188: 
189:     metadata = {"render_modes": []}
190: 
191:     def __init__(
192:         self,
193:         questions: list[MCQuestion],
194:         likelihood_model: LikelihoodModel,
195:         K: int = 4,
196:         reward_mode: str = "time_penalty",
197:         wait_penalty: float = 0.01,
198:         early_buzz_penalty: float = 0.0,
199:         buzz_correct: float = 1.0,
200:         buzz_incorrect: float = -0.5,
201:         belief_mode: str = "from_scratch",
202:         beta: float = 5.0,
203:         seed: int = 13,
204:         precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None,
205:         opponent_buzz_model: "OpponentBuzzModel | None" = None,
206:         ew_reward_correct: float = 10.0,
207:         ew_reward_incorrect: float = -5.0,
208:         ew_opponent_expected_value: float = 0.0,
209:         variable_K: bool = False,
210:         max_K: int | None = None,
211:         use_action_masking: bool = False,
212:         end_mode: str = "force_commit",
213:         no_buzz_reward: float = 0.0,
214:     ) -> None:
215:         if not questions:
216:             raise ValueError("questions cannot be empty")
217:         if K < 2:
218:             raise ValueError("K must be >= 2")
219: 
220:         self.questions = questions
221:         self.likelihood_model = likelihood_model
222:         self.K = K
223:         self.reward_mode = reward_mode
224:         self.wait_penalty = wait_penalty
225:         self.early_buzz_penalty = early_buzz_penalty
226:         self.buzz_correct = buzz_correct
227:         self.buzz_incorrect = buzz_incorrect
228:         self.belief_mode = belief_mode
229:         self.beta = beta
230:         self.rng = random.Random(seed)
231:         self.precomputed_beliefs = precomputed_beliefs
232: 
233:         self.opponent_buzz_model = opponent_buzz_model
234:         self.ew_reward_correct = ew_reward_correct
235:         self.ew_reward_incorrect = ew_reward_incorrect
236:         self.ew_opponent_expected_value = ew_opponent_expected_value
237: 
238:         self.variable_K = variable_K
239:         self.use_action_masking = use_action_masking
240:         self.end_mode = end_mode
241:         self.no_buzz_reward = no_buzz_reward
242:         if variable_K:
243:             self._max_K = max_K or max(len(q.options) for q in questions)
244:         else:
245:             self._max_K = K
246: 
247:         # Build qid -> list-index map for precomputed belief lookups
248:         self._question_index_map: dict[str, int] = {
249:             q.qid: i for i, q in enumerate(questions)
250:         }
251: 
252:         obs_K = self._max_K if self.variable_K else self.K
253:         self.action_space = spaces.Discrete(obs_K + 1)
254:         self.observation_space = spaces.Box(
255:             low=-np.inf, high=np.inf, shape=(obs_K + 6,), dtype=np.float32
256:         )
257: 
258:         self.question: MCQuestion | None = None
259:         self.step_idx: int = 0
260:         self.prev_belief: np.ndarray | None = None
261:         self.belief: np.ndarray = np.ones(self.K, dtype=np.float32) / self.K
262:         self.terminated: bool = False
263:         self.truncated: bool = False
264:         self._sampled_human_buzz_pos: int | None = None
265:         self._current_question_idx: int = 0
266: 
267:     # ------------------------------------------------------------------
268:     # Properties
269:     # ------------------------------------------------------------------
270: 
271:     @property
272:     def total_steps(self) -> int:
273:         """Total number of incremental clue steps for the current question.
274: 
275:         Returns
276:         -------
277:         int
278:             Length of ``question.run_indices`` if a question is loaded, else 1.
279:         """
280:         if self.question is None:
281:             return 1
282:         return len(self.question.run_indices)
283: 
284:     # ------------------------------------------------------------------
285:     # Helper methods
286:     # ------------------------------------------------------------------
287: 
288:     def _sample_question(self) -> MCQuestion:
289:         """Sample a random question from the question pool.
290: 
291:         Returns
292:         -------
293:         MCQuestion
294:             A randomly selected question.
295:         """
296:         return self.rng.choice(self.questions)
297: 
298:     def _sample_human_buzz(self, question: MCQuestion) -> int | None:
299:         """Sample a human buzz position from the question's distribution.
300: 
301:         Uses weighted random sampling based on the number of humans who
302:         buzzed at each position. Returns None if no human buzz data exists.
303: 
304:         Parameters
305:         ----------
306:         question : MCQuestion
307:             The question to sample a human buzz position for.
308: 
309:         Returns
310:         -------
311:         int or None
312:             Sampled token position, or None if no human buzz data.
313:         """
314:         if not question.human_buzz_positions:
315:             return None
316:         positions = []
317:         weights = []
318:         for pos, count in question.human_buzz_positions:
319:             positions.append(int(pos))
320:             weights.append(max(1, int(count)))
321:         if not positions:
322:             return None
323:         return self.rng.choices(positions, weights=weights, k=1)[0]
324: 
325:     def _softmax_scores(self, scores: np.ndarray) -> np.ndarray:
326:         """Convert raw likelihood scores to a probability distribution.
327: 
328:         Delegates to module-level ``_softmax`` with this environment's beta.
329: 
330:         Parameters
331:         ----------
332:         scores : np.ndarray
333:             Raw similarity scores of shape (K,).
334: 
335:         Returns
336:         -------
337:         np.ndarray
338:             Probability distribution of shape (K,), dtype float32.
339:         """
340:         return _softmax(scores, self.beta)
341: 
342:     def _compute_belief(self, question: MCQuestion, step_idx: int) -> np.ndarray:
343:         """Compute belief distribution over answer options at a given step.
344: 
345:         Two modes are supported:
346: 
347:         ``from_scratch``
348:             Score the cumulative clue prefix against all option profiles,
349:             then apply softmax. Each step is independent of the previous
350:             belief.
351: 
352:         ``sequential_bayes``
353:             Extract only the new clue fragment since the last step, score
354:             it, and perform a Bayesian update: posterior = prior * likelihood,
355:             then normalize. This is cheaper per step but may accumulate
356:             approximation errors.
357: 
358:         Parameters
359:         ----------
360:         question : MCQuestion
361:             Current question being played.
362:         step_idx : int
363:             Current step index (0-based, indexes into run_indices).
364: 
365:         Returns
366:         -------
367:         np.ndarray
368:             Updated belief distribution of shape (K,), dtype float32.
369: 
370:         Raises
371:         ------
372:         ValueError
373:             If ``self.belief_mode`` is not a recognized mode.
374:         """
375:         if self.precomputed_beliefs is not None:
376:             key = (self._current_question_idx, step_idx)
377:             return self.precomputed_beliefs[key].copy()
378: 
379:         if self.belief_mode == "from_scratch":
380:             prefix = question.cumulative_prefixes[step_idx]
381:             scores = self.likelihood_model.score(prefix, question.option_profiles)
382:             return self._softmax_scores(scores)
383: 
384:         if self.belief_mode == "sequential_bayes":
385:             idx = question.run_indices[step_idx]
386:             prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
387:             frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
388:             scores = self.likelihood_model.score(frag, question.option_profiles)
389:             likelihood = self._softmax_scores(scores)
390:             posterior = self.belief * likelihood
391:             denom = posterior.sum()
392:             if denom <= 0:
393:                 posterior = np.ones(self.K, dtype=np.float32) / self.K
394:             else:
395:                 posterior = posterior / denom
396:             return posterior.astype(np.float32)
397: 
398:         raise ValueError(f"Unknown belief_mode: {self.belief_mode}")
399: 
400:     def _obs(self) -> np.ndarray:
401:         """Build the observation vector from current belief state.
402: 
403:         In variable-K mode, uses padded features sized to ``_max_K``.
404:         Otherwise delegates to ``extract_belief_features``.
405: 
406:         Returns
407:         -------
408:         np.ndarray
409:             Feature vector of shape (obs_K + 6,), dtype float32.
410:         """
411:         if self.variable_K:
412:             from models.features import extract_padded_belief_features
413: 
414:             return extract_padded_belief_features(
415:                 belief=self.belief,
416:                 prev_belief=self.prev_belief,
417:                 step_idx=self.step_idx,
418:                 total_steps=self.total_steps,
419:                 max_K=self._max_K,
420:             )
421:         return extract_belief_features(
422:             belief=self.belief,
423:             prev_belief=self.prev_belief,
424:             step_idx=self.step_idx,
425:             total_steps=self.total_steps,
426:         )
427: 
428:     def action_masks(self) -> np.ndarray:
429:         """Return a boolean mask of valid actions.
430: 
431:         WAIT (action 0) is always valid.  Buzz actions ``1..K_actual``
432:         are valid; padded slots ``K_actual+1..max_K`` are invalid.
433: 
434:         Returns
435:         -------
436:         np.ndarray
437:             Boolean array of shape ``(max_K + 1,)`` or ``(K + 1,)``.
438:         """
439:         n_actions = self._max_K + 1 if self.variable_K else self.K + 1
440:         mask = np.zeros(n_actions, dtype=bool)
441:         mask[0] = True  # WAIT
442:         k_actual = len(self.question.options) if self.question is not None else self.K
443:         mask[1 : k_actual + 1] = True
444:         return mask
445: 
446:     def _step_to_token_pos(self, step_idx: int) -> int:
447:         """Convert a step index to the corresponding token position.
448: 
449:         Used by the ``human_grounded`` reward mode to compare the agent's
450:         buzz position against the sampled human buzz position.
451: 
452:         Parameters
453:         ----------
454:         step_idx : int
455:             Step index (0-based, indexes into run_indices).
456: 
457:         Returns
458:         -------
459:         int
460:             Token position in the original question text.
461:         """
462:         if self.question is None or not self.question.run_indices:
463:             return step_idx
464:         if step_idx >= len(self.question.run_indices):
465:             return self.question.run_indices[-1]
466:         if step_idx < 0:
467:             return self.question.run_indices[0]
468:         return self.question.run_indices[step_idx]
469: 
470:     def _expected_wins_reward(
471:         self, question: MCQuestion, chosen_idx: int, last_seen_step: int
472:     ) -> float:
473:         """Compute Expected Wins reward at buzz time.
474: 
475:         R_t = S_t * V_self + (1 - S_t) * V_opp
476: 
477:         where S_t = P(opponent has NOT buzzed by step t).
478:         """
479:         correct = chosen_idx == question.gold_index
480:         v_self = self.ew_reward_correct if correct else self.ew_reward_incorrect
481:         if self.opponent_buzz_model is None:
482:             return v_self
483:         s_t = self.opponent_buzz_model.prob_survive_to_step(question, last_seen_step)
484:         return s_t * v_self + (1.0 - s_t) * self.ew_opponent_expected_value
485: 
486:     def _buzz_reward(self, question: MCQuestion, chosen_idx: int, last_seen_step: int) -> float:
487:         """Compute the reward for buzzing with a given answer.
488: 
489:         Dispatches on ``self.reward_mode``:
490: 
491:         ``simple``
492:             +1.0 for correct, -1.0 for incorrect.
493:         ``human_grounded``
494:             0.0 if the agent buzzes after the sampled human would have;
495:             otherwise +buzz_correct / +buzz_incorrect.
496:         ``time_penalty`` (default)
497:             +buzz_correct / +buzz_incorrect. The per-step wait penalty
498:             is applied separately in ``step()``.
499:         ``expected_wins``
500:             S_t * V_self + (1 - S_t) * V_opp via opponent model.
501: 
502:         Parameters
503:         ----------
504:         question : MCQuestion
505:             Current question.
506:         chosen_idx : int
507:             Index of the chosen answer option (0-based).
508:         last_seen_step : int
509:             Step index of the last clue seen before buzzing.
510: 
511:         Returns
512:         -------
513:         float
514:             Reward value.
515:         """
516:         correct = chosen_idx == question.gold_index
517:         if self.reward_mode == "simple":
518:             return 1.0 if correct else -1.0
519:         if self.reward_mode == "human_grounded":
520:             token_pos = self._step_to_token_pos(last_seen_step)
521:             if self._sampled_human_buzz_pos is not None and token_pos > self._sampled_human_buzz_pos:
522:                 return 0.0
523:             return self.buzz_correct if correct else self.buzz_incorrect
524:         if self.reward_mode == "expected_wins":
525:             return self._expected_wins_reward(question, chosen_idx, last_seen_step)
526:         # default: time_penalty
527:         reward = self.buzz_correct if correct else self.buzz_incorrect
528: 
529:         if self.early_buzz_penalty > 0 and self.total_steps > 1:
530:             progress = np.clip((last_seen_step + 1) / self.total_steps, 0.0, 1.0)
531:             reward -= float(self.early_buzz_penalty) * (1.0 - progress)
532: 
533:         return reward
534: 
535:     # ------------------------------------------------------------------
536:     # Gymnasium interface
537:     # ------------------------------------------------------------------
538: 
539:     def reset(
540:         self, *, seed: int | None = None, options: dict[str, Any] | None = None
541:     ) -> tuple[np.ndarray, dict[str, Any]]:
542:         """Reset the environment and start a new episode.
543: 
544:         Samples a random question from the pool, initializes belief to a
545:         uniform distribution, and returns the initial observation.
546: 
547:         Parameters
548:         ----------
549:         seed : int or None
550:             If provided, reseeds both the internal RNG and numpy's global
551:             RNG for reproducibility.
552:         options : dict or None
553:             Unused. Included for Gymnasium API compatibility.
554: 
555:         Returns
556:         -------
557:         observation : np.ndarray
558:             Initial observation of shape (K + 6,), dtype float32.
559:             Belief is uniform, so top_p = 1/K, margin = 0, entropy = max.
560:         info : dict[str, Any]
561:             Episode metadata. Contains ``"qid"`` (the sampled question ID).
562:         """
563:         if seed is not None:
564:             self.rng.seed(seed)
565:             np.random.seed(seed)
566: 
567:         if options and "question_idx" in options:
568:             q_idx = int(options["question_idx"])
569:             if q_idx < 0 or q_idx >= len(self.questions):
570:                 raise ValueError(f"question_idx out of range: {q_idx}")
571:             self.question = self.questions[q_idx]
572:             self._current_question_idx = q_idx
573:         else:
574:             self.question = self._sample_question()
575:             self._current_question_idx = self._question_index_map.get(
576:                 self.question.qid, self.questions.index(self.question)
577:             )
578:         self.step_idx = 0
579:         self.prev_belief = None
580:         self.belief = np.ones(self.K, dtype=np.float32) / self.K
581:         self.terminated = False
582:         self.truncated = False
583:         self._sampled_human_buzz_pos = self._sample_human_buzz(self.question)
584:         return self._obs(), {"qid": self.question.qid}
585: 
586:     def step(
587:         self, action: int
588:     ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
589:         """Execute one step in the environment.
590: 
591:         If ``action == 0`` (WAIT):
592:             - Saves previous belief, computes new belief from current clue.
593:             - Applies wait_penalty if reward_mode is ``"time_penalty"``.
594:             - Advances step counter.
595:             - If all clues exhausted: forced termination with best-guess
596:               answer (``truncated=True``).
597: 
598:         If ``action in 1..K`` (BUZZ):
599:             - Computes buzz reward for chosen answer option ``action - 1``.
600:             - Episode ends (``terminated=True``).
601: 
602:         Parameters
603:         ----------
604:         action : int
605:             Action to take. 0 = WAIT, 1..K = buzz with option (action-1).
606: 
607:         Returns
608:         -------
609:         observation : np.ndarray
610:             Updated observation of shape (K + 6,), dtype float32.
611:         reward : float
612:             Scalar reward for this step.
613:         terminated : bool
614:             True if the agent buzzed (natural episode end).
615:         truncated : bool
616:             True if all clues were exhausted (forced termination).
617:         info : dict[str, Any]
618:             Step metadata. Always contains ``"qid"`` and ``"step_idx"``.
619:             On BUZZ: also ``"chosen_idx"`` and ``"correct"``.
620:             On forced termination in ``force_commit`` mode: also
621:             ``"forced_choice"`` and ``"forced_correct"``.
622:             On forced termination in ``no_buzz`` mode: also ``"no_buzz"``,
623:             ``"forced_choice" = -1``, and ``"forced_correct" = False``.
624: 
625:         Raises
626:         ------
627:         RuntimeError
628:             If called before ``reset()`` or after episode has ended.
629:         ValueError
630:             If ``action`` is not in the action space.
631:         """
632:         if self.question is None:
633:             raise RuntimeError("Environment must be reset() before step().")
634:         if self.terminated or self.truncated:
635:             raise RuntimeError("Cannot call step() on terminated/truncated episode.")
636:         if not self.action_space.contains(action):
637:             raise ValueError(f"Invalid action: {action}")
638: 
639:         info: dict[str, Any] = {"qid": self.question.qid}
640:         reward = 0.0
641: 
642:         if action == 0:
643:             # WAIT: reveal next clue and update belief
644:             self.prev_belief = self.belief.copy()
645:             self.belief = self._compute_belief(self.question, self.step_idx)
646:             if self.reward_mode == "time_penalty":
647:                 reward -= self.wait_penalty
648: 
649:             self.step_idx += 1
650:             if self.step_idx >= self.total_steps:
651:                 last_seen = self.step_idx - 1
652:                 self.truncated = True
653:                 info["step_idx"] = last_seen
654:                 if self.end_mode == "force_commit":
655:                     forced_choice = int(np.argmax(self.belief))
656:                     reward += self._buzz_reward(self.question, forced_choice, last_seen)
657:                     info["forced_choice"] = forced_choice
658:                     info["forced_correct"] = forced_choice == self.question.gold_index
659:                 elif self.end_mode == "no_buzz":
660:                     reward += self.no_buzz_reward
661:                     info["no_buzz"] = True
662:                     info["forced_choice"] = -1
663:                     info["forced_correct"] = False
664:                 else:
665:                     raise ValueError(f"Unknown end_mode: {self.end_mode}")
666:             else:
667:                 info["step_idx"] = self.step_idx
668: 
669:         else:
670:             # BUZZ: select an answer option
671:             last_seen = max(0, self.step_idx - 1)
672:             chosen_idx = action - 1
673:             reward += self._buzz_reward(self.question, chosen_idx, last_seen)
674:             self.terminated = True
675:             info["step_idx"] = last_seen
676:             info["chosen_idx"] = chosen_idx
677:             info["correct"] = chosen_idx == self.question.gold_index
678: 
679:         obs = self._obs()
680:         return obs, float(reward), self.terminated, self.truncated, info
681: 
682: 
683: def make_env_from_config(
684:     mc_questions: list[MCQuestion],
685:     likelihood_model: LikelihoodModel,
686:     config: dict[str, Any],
687:     precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None,
688: ) -> TossupMCEnv:
689:     """Construct a TossupMCEnv from YAML configuration.
690: 
691:     Factory function that reads the ``environment``, ``data``, and
692:     ``likelihood`` sections of a config dict and instantiates a fully
693:     configured environment. The likelihood model must be pre-constructed
694:     (e.g., via ``build_likelihood_from_config``).
695: 
696:     Parameters
697:     ----------
698:     mc_questions : list[MCQuestion]
699:         List of MCQuestion instances with options and answer profiles.
700:         Must be non-empty.
701:     likelihood_model : LikelihoodModel
702:         Pre-constructed likelihood model for scoring clues against options.
703:         Use ``build_likelihood_from_config`` to create one from config.
704:     config : dict[str, Any]
705:         Full YAML config dict. Must contain the following sections:
706: 
707:         - ``environment``: reward mode, penalties, belief mode
708:         - ``data``: K (number of answer choices)
709:         - ``likelihood``: beta (softmax temperature)
710:     precomputed_beliefs : dict or None
711:         Optional precomputed belief cache from ``precompute_beliefs()``.
712:         When provided, ``_compute_belief`` uses O(1) lookups instead of
713:         calling ``likelihood_model.score()``.
714: 
715:     Returns
716:     -------
717:     TossupMCEnv
718:         A configured Gymnasium environment ready for ``reset()``.
719: 
720:     Examples
721:     --------
722:     >>> from qb_data.config import load_config
723:     >>> from models.likelihoods import build_likelihood_from_config
724:     >>> config = load_config("configs/default.yaml")
725:     >>> model = build_likelihood_from_config(config, corpus_texts=corpus)
726:     >>> env = make_env_from_config(mc_questions, model, config)
727:     >>> obs, info = env.reset()
728:     """
729:     env_cfg = config["environment"]
730:     data_cfg = config["data"]
731:     lik_cfg = config["likelihood"]
732:     variable_k = bool(data_cfg.get("variable_K", False) or env_cfg.get("variable_K", False))
733:     max_k_raw = data_cfg.get("max_K") or env_cfg.get("max_K")
734:     return TossupMCEnv(
735:         questions=mc_questions,
736:         likelihood_model=likelihood_model,
737:         K=int(data_cfg.get("K", 4)),
738:         reward_mode=str(env_cfg.get("reward", env_cfg.get("reward_mode", "time_penalty"))),
739:         seed=int(env_cfg.get("seed", 13)),
740:         wait_penalty=float(env_cfg.get("wait_penalty", 0.01)),
741:         early_buzz_penalty=float(env_cfg.get("early_buzz_penalty", 0.0)),
742:         buzz_correct=float(env_cfg.get("buzz_correct", 1.0)),
743:         buzz_incorrect=float(env_cfg.get("buzz_incorrect", -0.5)),
744:         belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
745:         beta=float(lik_cfg.get("beta", 5.0)),
746:         precomputed_beliefs=precomputed_beliefs,
747:         end_mode=str(env_cfg.get("end_mode", "force_commit")),
748:         no_buzz_reward=float(env_cfg.get("no_buzz_reward", 0.0)),
749:         variable_K=variable_k,
750:         max_K=int(max_k_raw) if max_k_raw is not None else None,
751:         use_action_masking=bool(env_cfg.get("use_action_masking", False)),
752:     )
````

## File: scripts/evaluate_all.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Comprehensive evaluation with control experiments and visualization.
  4: 
  5: Runs the SoftmaxProfileBuzzer at the best threshold (from baseline sweep),
  6: then executes control experiments (choices-only, shuffle, alias substitution)
  7: and generates comparison plots and tables for the CS234 writeup.
  8: 
  9: Consumes outputs from:
 10: - build_mc_dataset.py (mc_dataset.json, alias_lookup.json)
 11: - run_baselines.py (baseline_summary.json)
 12: - train_ppo.py (ppo_summary.json)
 13: 
 14: Produces:
 15: - evaluation_report.json (full eval + controls + baseline + PPO summaries)
 16: - plots/entropy_vs_clue.png
 17: - plots/calibration.png
 18: - plots/comparison.csv
 19: 
 20: Usage:
 21:     python scripts/evaluate_all.py --smoke
 22:     python scripts/evaluate_all.py --config configs/custom.yaml
 23:     python scripts/evaluate_all.py --mc-path artifacts/main/mc_dataset.json
 24: 
 25: Ported from qb-rl reference implementation (scripts/evaluate_all.py) with
 26: import path adaptations for the unified qanta-buzzer codebase.
 27: """
 28: 
 29: from __future__ import annotations
 30: 
 31: import argparse
 32: from dataclasses import asdict
 33: from pathlib import Path
 34: import sys
 35: 
 36: import numpy as np
 37: 
 38: PROJECT_ROOT = Path(__file__).resolve().parents[1]
 39: if str(PROJECT_ROOT) not in sys.path:
 40:     sys.path.insert(0, str(PROJECT_ROOT))
 41: 
 42: from agents.bayesian_buzzer import SoftmaxProfileBuzzer
 43: from agents.threshold_buzzer import (
 44:     _softmax_episode_from_precomputed,
 45:     precompute_beliefs,
 46: )
 47: from evaluation.controls import (
 48:     run_alias_substitution_control,
 49:     run_choices_only_control,
 50:     run_shuffle_control_precomputed,
 51: )
 52: from evaluation.metrics import (
 53:     calibration_at_buzz,
 54:     per_category_accuracy,
 55:     summarize_buzz_metrics,
 56: )
 57: from evaluation.plotting import (
 58:     plot_calibration_curve,
 59:     plot_entropy_vs_clue_index,
 60:     save_comparison_table,
 61: )
 62: from qb_data.config import merge_overrides
 63: from scripts._common import (
 64:     ARTIFACT_DIR,
 65:     build_likelihood_model,
 66:     load_config,
 67:     load_embedding_cache,
 68:     load_json,
 69:     load_mc_questions,
 70:     parse_overrides,
 71:     save_json,
 72: )
 73: 
 74: 
 75: def parse_args() -> argparse.Namespace:
 76:     """Parse command-line arguments.
 77: 
 78:     Returns
 79:     -------
 80:     argparse.Namespace
 81:         Parsed arguments with config, smoke, and mc_path fields.
 82:     """
 83:     parser = argparse.ArgumentParser(
 84:         description="Evaluate all agents and controls."
 85:     )
 86:     parser.add_argument(
 87:         "--config", type=str, default=None,
 88:         help="Path to YAML config file (default: configs/default.yaml).",
 89:     )
 90:     parser.add_argument(
 91:         "--smoke", action="store_true",
 92:         help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
 93:     )
 94:     parser.add_argument(
 95:         "--mc-path", type=str, default=None,
 96:         help="Optional MC dataset JSON path (overrides config-derived path).",
 97:     )
 98:     parser.add_argument(
 99:         "overrides",
100:         nargs="*",
101:         help="Config overrides: key=value (e.g. likelihood.model=tfidf)",
102:     )
103:     return parser.parse_args()
104: 
105: 
106: def pick_best_softmax_threshold(
107:     out_dir: Path, default_threshold: float
108: ) -> float:
109:     """Select the best softmax threshold from baseline sweep results.
110: 
111:     Loads baseline_summary.json and extracts the threshold with the
112:     highest mean S_q score from the softmax_profile results.
113: 
114:     Parameters
115:     ----------
116:     out_dir : Path
117:         Directory containing baseline_summary.json.
118:     default_threshold : float
119:         Fallback threshold if baseline summary is unavailable.
120: 
121:     Returns
122:     -------
123:     float
124:         Best threshold by S_q score, or default_threshold if unavailable.
125:     """
126:     summary_path = out_dir / "baseline_summary.json"
127:     if not summary_path.exists():
128:         return default_threshold
129:     summary = load_json(summary_path)
130:     softmax = summary.get("softmax_profile", {})
131:     if not softmax:
132:         return default_threshold
133:     best_t = default_threshold
134:     best_sq = float("-inf")
135:     for t_str, metrics in softmax.items():
136:         sq = float(metrics.get("mean_sq", float("-inf")))
137:         if sq > best_sq:
138:             best_sq = sq
139:             best_t = float(t_str)
140:     return best_t
141: 
142: 
143: def main() -> None:
144:     """Run comprehensive evaluation with controls and visualizations."""
145:     args = parse_args()
146: 
147:     config = load_config(args.config, smoke=args.smoke)
148:     overrides = parse_overrides(args)
149:     if overrides:
150:         print(f"Applying overrides: {overrides}")
151:         config = merge_overrides(config, overrides)
152: 
153:     split = "smoke" if args.smoke else "main"
154:     out_dir = ARTIFACT_DIR / split
155:     mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"
156: 
157:     # Fallback: check data/processed/ if artifacts path doesn't exist
158:     if not mc_path.exists():
159:         fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
160:         if fallback.exists():
161:             print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
162:             mc_path = fallback
163: 
164:     print(f"Loading MC questions from: {mc_path}")
165:     mc_questions = load_mc_questions(mc_path)
166:     print(f"Loaded {len(mc_questions)} MC questions")
167: 
168:     # Load alias lookup (generated by build_mc_dataset.py)
169:     alias_path = out_dir / "alias_lookup.json"
170:     if alias_path.exists():
171:         alias_lookup = load_json(alias_path)
172:     else:
173:         print(f"Warning: alias_lookup.json not found at {alias_path}, using empty lookup")
174:         alias_lookup = {}
175: 
176:     # Build likelihood model
177:     print(f"Building likelihood model: {config['likelihood']['model']}")
178:     likelihood_model = build_likelihood_model(config, mc_questions)
179:     load_embedding_cache(likelihood_model, config)
180:     beta = float(config["likelihood"].get("beta", 5.0))
181:     alpha = float(config["bayesian"].get("alpha", 10.0))
182:     default_threshold = float(config["bayesian"]["threshold_sweep"][0])
183:     threshold = pick_best_softmax_threshold(out_dir, default_threshold=default_threshold)
184:     print(f"Using best softmax threshold: {threshold}")
185: 
186:     # Precompute beliefs once (single pass of likelihood_model.score())
187:     print("Precomputing beliefs...")
188:     precomputed = precompute_beliefs(mc_questions, likelihood_model, beta)
189: 
190:     # Precomputed evaluation (zero extra score() calls)
191:     def evaluate_questions_precomputed(pqs):
192:         runs = [asdict(_softmax_episode_from_precomputed(pq, threshold, alpha)) for pq in pqs]
193:         summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
194:         summary["runs"] = runs
195:         return summary
196: 
197:     # Live evaluator for controls that genuinely change option text (alias)
198:     def evaluate_questions_live(qset):
199:         agent = SoftmaxProfileBuzzer(
200:             likelihood_model=likelihood_model,
201:             threshold=threshold,
202:             beta=beta,
203:             alpha=alpha,
204:         )
205:         runs = [asdict(agent.run_episode(q)) for q in qset]
206:         summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
207:         summary["runs"] = runs
208:         return summary
209: 
210:     # --- Run evaluations ---
211:     print("Running full evaluation...")
212:     full_eval = evaluate_questions_precomputed(precomputed)
213: 
214:     # Compute per-category breakdown
215:     print("\nComputing per-category breakdown...")
216:     per_category_results = per_category_accuracy(full_eval["runs"], mc_questions)
217: 
218:     # Sort by category name for readability
219:     per_category_sorted = dict(sorted(per_category_results.items()))
220: 
221:     print("\nPer-category accuracy:")
222:     for category, metrics in per_category_sorted.items():
223:         print(
224:             f"  {category:20s} (n={metrics['n']:3.0f}): "
225:             f"acc={metrics['buzz_accuracy']:.3f}, "
226:             f"S_q={metrics['mean_sq']:.3f}"
227:         )
228:     print()
229: 
230:     print("Running shuffle control...")
231:     shuffle_eval = run_shuffle_control_precomputed(precomputed, threshold, alpha)
232: 
233:     print("Running alias substitution control...")
234:     alias_eval = run_alias_substitution_control(
235:         mc_questions,
236:         alias_lookup=alias_lookup,
237:         evaluator=lambda qset: evaluate_questions_live(qset),
238:     )
239: 
240:     print("Running choices-only control...")
241:     choices_only = run_choices_only_control(mc_questions)
242: 
243:     # --- Load existing artifacts ---
244:     ppo_summary_path = out_dir / "ppo_summary.json"
245:     ppo_summary = load_json(ppo_summary_path) if ppo_summary_path.exists() else {}
246:     baseline_summary_path = out_dir / "baseline_summary.json"
247:     baseline_summary = (
248:         load_json(baseline_summary_path) if baseline_summary_path.exists() else {}
249:     )
250: 
251:     # --- Build evaluation report ---
252:     report = {
253:         "softmax_profile_best_threshold": threshold,
254:         "full_eval": {k: v for k, v in full_eval.items() if k != "runs"},
255:         "controls": {
256:             "choices_only": choices_only,
257:             "shuffle": {k: v for k, v in shuffle_eval.items() if k != "runs"},
258:             "alias_substitution": {
259:                 k: v for k, v in alias_eval.items() if k != "runs"
260:             },
261:         },
262:         "per_category": per_category_sorted,
263:         "baseline_summary": baseline_summary,
264:         "ppo_summary": ppo_summary,
265:     }
266: 
267:     # Add Expected Wins summary only when that reward mode is active
268:     if config.get("environment", {}).get("reward_mode") == "expected_wins":
269:         from evaluation.metrics import expected_wins_score
270:         from qb_env.opponent_models import build_opponent_model_from_config
271: 
272:         opp_model = build_opponent_model_from_config(mc_questions, config)
273:         qid_to_q = {q.qid: q for q in mc_questions}
274:         if opp_model is not None:
275:             ew_scores = []
276:             for run in full_eval["runs"]:
277:                 q = qid_to_q.get(run.get("qid", ""), mc_questions[0])
278:                 opp_surv = [
279:                     opp_model.prob_survive_to_step(q, t)
280:                     for t in range(len(run.get("c_trace", [])))
281:                 ]
282:                 ew = expected_wins_score(
283:                     run.get("c_trace", []),
284:                     run.get("g_trace", []),
285:                     opp_surv,
286:                 )
287:                 ew_scores.append(ew)
288:             report["expected_wins"] = {
289:                 "mean_ew": float(np.mean(ew_scores)) if ew_scores else 0.0,
290:                 "n": len(ew_scores),
291:             }
292: 
293:     save_json(out_dir / "evaluation_report.json", report)
294: 
295:     # --- Generate visualizations ---
296:     print("Generating plots...")
297: 
298:     # Entropy vs clue index
299:     entropy_traces = [
300:         list(r["entropy_trace"])
301:         for r in full_eval["runs"]
302:         if r.get("entropy_trace")
303:     ]
304:     max_len = max((len(t) for t in entropy_traces), default=0)
305:     padded = np.full((len(entropy_traces), max_len), np.nan, dtype=np.float32)
306:     for i, trace in enumerate(entropy_traces):
307:         padded[i, : len(trace)] = np.array(trace, dtype=np.float32)
308:     entropy_trace = (
309:         np.nanmean(padded, axis=0).tolist() if max_len > 0 else []
310:     )
311:     plot_entropy_vs_clue_index(
312:         {"softmax_profile": entropy_trace},
313:         out_dir / "plots" / "entropy_vs_clue.png",
314:     )
315: 
316:     # Calibration curve — use top_p (belief in top answer) as confidence
317:     confidences = []
318:     outcomes = []
319:     for row in full_eval["runs"]:
320:         top_p = row.get("top_p_trace", row.get("c_trace", []))
321:         if not top_p:
322:             continue
323:         idx = min(int(row["buzz_step"]), len(top_p) - 1)
324:         confidences.append(float(top_p[idx]))
325:         outcomes.append(1 if bool(row["correct"]) else 0)
326:     plot_calibration_curve(
327:         confidences, outcomes, out_dir / "plots" / "calibration.png"
328:     )
329: 
330:     # Comparison table: include baseline sweep, controls, and PPO
331:     table_rows = []
332: 
333:     # Add baseline sweep results (threshold at multiple values)
334:     if "threshold" in baseline_summary:
335:         for threshold_str, metrics in baseline_summary["threshold"].items():
336:             table_rows.append({
337:                 "agent": f"threshold_{threshold_str}",
338:                 **{k: v for k, v in metrics.items() if k != "runs"},
339:             })
340: 
341:     # Add softmax_profile sweep results
342:     if "softmax_profile" in baseline_summary:
343:         for threshold_str, metrics in baseline_summary["softmax_profile"].items():
344:             table_rows.append({
345:                 "agent": f"softmax_{threshold_str}",
346:                 **{k: v for k, v in metrics.items() if k != "runs"},
347:             })
348: 
349:     # Add full softmax eval (best threshold) and control experiments
350:     table_rows.append({
351:         "agent": "full_softmax",
352:         **{k: v for k, v in full_eval.items() if k != "runs"},
353:     })
354:     table_rows.append({
355:         "agent": "shuffle_control",
356:         **{k: v for k, v in shuffle_eval.items() if k != "runs"},
357:     })
358:     table_rows.append({
359:         "agent": "alias_control",
360:         **{k: v for k, v in alias_eval.items() if k != "runs"},
361:     })
362: 
363:     # Add PPO if available
364:     if ppo_summary:
365:         table_rows.append({"agent": "ppo", **ppo_summary})
366: 
367:     save_comparison_table(table_rows, out_dir / "plots" / "comparison.csv")
368: 
369:     print(f"Wrote evaluation report to: {out_dir / 'evaluation_report.json'}")
370: 
371: 
372: if __name__ == "__main__":
373:     main()
````

## File: scripts/run_full_pipeline.sh
````bash
  1: #!/usr/bin/env bash
  2: # Full pipeline with parallelism — runs the core pipeline plus key extensions.
  3: # Phases 9/10/12/18/19 require manual execution (see docs/full-pipeline-runbook.md).
  4: #
  5: # Dependencies form a DAG:
  6: #
  7: #   Phase 1 (build MC dataset)
  8: #     ├── Wave 1 (3 parallel tracks): Phases 2, 3, 5
  9: #     │     Track A: baselines → writes artifacts/main/baseline_summary.json
 10: #     │     Track B: PPO → writes artifacts/main/ppo_model.zip
 11: #     │     Track C: T5 policy → writes checkpoints/
 12: #     ├── Wave 2 (sequential, after Wave 1): Phases 4, 6, 11, 15
 13: #     │     All read/write artifacts/main/ — must be sequential
 14: #     ├── Wave 3 (sequential): Phases 14, 16, 17
 15: #     │     PPO ablations that reuse artifacts/main/
 16: #     └── Wave 4 (sequential): Phase 13 (K-sensitivity)
 17: #         Builds to artifacts/k*/ then runs baselines (writes artifacts/main/)
 18: #         Must run after Wave 2 so it doesn't clobber baseline_summary.json
 19: #
 20: # Usage:
 21: #   bash scripts/run_full_pipeline.sh                    # t5-base (balanced)
 22: #   bash scripts/run_full_pipeline.sh --t5-model t5-small # fastest
 23: #   bash scripts/run_full_pipeline.sh --t5-model t5-large # full quality
 24: #   bash scripts/run_full_pipeline.sh --sequential        # no parallelism
 25: #
 26: # Requirements:
 27: #   - Python venv activated with `pip install -e .`
 28: #   - questions.csv at repo root
 29: #   - ~10 GB free disk space
 30: #
 31: # Estimated wall time (Apple M3 Max, 64 GB):
 32: #   t5-small, parallel: ~2–3 hours
 33: #   t5-base, parallel:  ~3–5 hours
 34: #   t5-large, parallel: ~6–10 hours
 35: 
 36: set -euo pipefail
 37: 
 38: REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
 39: cd "$REPO_ROOT"
 40: 
 41: # Parse arguments
 42: T5_MODEL="t5-base"
 43: SEQUENTIAL=false
 44: while [ $# -gt 0 ]; do
 45:     case "$1" in
 46:         --t5-model) T5_MODEL="$2"; shift 2 ;;
 47:         --t5-model=*) T5_MODEL="${1#*=}"; shift ;;
 48:         --sequential) SEQUENTIAL=true; shift ;;
 49:         *) shift ;;
 50:     esac
 51: done
 52: 
 53: echo "============================================================"
 54: echo "FULL PIPELINE — T5 model: $T5_MODEL, parallel: $([ "$SEQUENTIAL" = true ] && echo no || echo yes)"
 55: echo "============================================================"
 56: echo ""
 57: 
 58: RESULTS="$REPO_ROOT/results"
 59: mkdir -p "$RESULTS"
 60: 
 61: # Activate venv if available
 62: if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
 63:     source "$REPO_ROOT/.venv/bin/activate"
 64: fi
 65: 
 66: # Helper: run a command, log to file, print status on completion
 67: run_phase() {
 68:     local PHASE="$1"
 69:     local LOG="$RESULTS/phase_${PHASE}.log"
 70:     shift
 71:     echo "[Phase $PHASE] STARTED at $(date +%H:%M:%S)"
 72:     if PYTHONUNBUFFERED=1 "$@" > "$LOG" 2>&1; then
 73:         echo "[Phase $PHASE] DONE at $(date +%H:%M:%S) — see $LOG"
 74:     else
 75:         echo "[Phase $PHASE] FAILED at $(date +%H:%M:%S) — see $LOG"
 76:         return 1
 77:     fi
 78: }
 79: 
 80: # Helper: wait for background jobs, exit on first failure
 81: wait_all() {
 82:     local PIDS=("$@")
 83:     for pid in "${PIDS[@]}"; do
 84:         if ! wait "$pid"; then
 85:             echo "ERROR: Background job $pid failed"
 86:             kill "${PIDS[@]}" 2>/dev/null || true
 87:             exit 1
 88:         fi
 89:     done
 90: }
 91: 
 92: ########################################################################
 93: # PHASE 1: Build MC dataset (sequential — everything depends on this)
 94: ########################################################################
 95: echo "=== PHASE 1: Build MC dataset ==="
 96: python scripts/build_mc_dataset.py \
 97:     --config configs/default.yaml \
 98:     --output-dir artifacts/main
 99: echo "[Phase 1] DONE — $(python -c "import json; print(f'{len(json.load(open(\"artifacts/main/mc_dataset.json\")))} MC questions')")"
100: echo ""
101: 
102: MC="artifacts/main/mc_dataset.json"
103: 
104: if [ "$SEQUENTIAL" = true ]; then
105:     ####################################################################
106:     # SEQUENTIAL MODE
107:     ####################################################################
108:     echo "=== Running all phases sequentially ==="
109: 
110:     echo "=== PHASE 2: Baselines (TF-IDF) ==="
111:     python scripts/run_baselines.py --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
112:     cp artifacts/main/baseline_summary.json "$RESULTS/baselines_tfidf.json"
113: 
114:     echo "=== PHASE 3: PPO (100k steps) ==="
115:     python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" --seed 13 --deterministic-eval likelihood.model=tfidf
116:     cp artifacts/main/ppo_summary.json "$RESULTS/ppo_default.json"
117:     cp artifacts/main/ppo_model.zip "$RESULTS/ppo_model_default.zip"
118: 
119:     echo "=== PHASE 4: Evaluate all ==="
120:     python scripts/evaluate_all.py --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
121:     cp artifacts/main/evaluation_report.json "$RESULTS/eval_default.json"
122: 
123:     echo "=== PHASE 5: T5 policy ==="
124:     python scripts/train_t5_policy.py --config configs/t5_policy.yaml model.model_name="$T5_MODEL"
125: 
126:     echo "=== PHASE 6: Compare policies ==="
127:     python scripts/compare_policies.py \
128:         --mlp-checkpoint artifacts/main/ppo_model \
129:         --t5-checkpoint checkpoints/ppo_t5/best_model \
130:         --mc-path "$MC" \
131:         --output "$RESULTS/t5_comparison.json"
132: 
133:     echo "=== PHASE 11: Expected Wins ==="
134:     python scripts/evaluate_all.py --config configs/default.yaml --mc-path "$MC" \
135:         likelihood.model=tfidf environment.reward_mode=expected_wins environment.opponent_buzz_model.type=logistic
136:     cp artifacts/main/evaluation_report.json "$RESULTS/eval_ew_logistic.json"
137: 
138:     echo "=== PHASE 14: Reward modes ==="
139:     for MODE in simple human_grounded; do
140:         python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
141:             --seed 13 --deterministic-eval likelihood.model=tfidf environment.reward_mode="$MODE"
142:         cp artifacts/main/ppo_summary.json "$RESULTS/ppo_$MODE.json"
143:     done
144: 
145:     echo "=== PHASE 16: Stop-only PPO ==="
146:     python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
147:         --seed 13 --deterministic-eval --policy-mode stop_only likelihood.model=tfidf
148:     cp artifacts/main/ppo_summary.json "$RESULTS/ppo_stop_only.json"
149: 
150:     echo "=== PHASE 17: No-buzz horizon ==="
151:     python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
152:         --seed 13 --deterministic-eval likelihood.model=tfidf environment.end_mode=no_buzz environment.no_buzz_reward=-0.25
153:     cp artifacts/main/ppo_summary.json "$RESULTS/ppo_no_buzz.json"
154: 
155:     echo "=== PHASE 15: Belief mode (sequential_bayes) ==="
156:     python scripts/run_baselines.py --config configs/default.yaml --mc-path "$MC" \
157:         environment.belief_mode=sequential_bayes likelihood.model=tfidf
158:     cp artifacts/main/baseline_summary.json "$RESULTS/baselines_seqbayes.json"
159: 
160:     echo "=== PHASE 9: Distractor comparison ==="
161:     for STRAT in tfidf_profile category_random; do
162:         python scripts/build_mc_dataset.py --config configs/default.yaml \
163:             --output-dir "artifacts/distractor_$STRAT" data.distractor_strategy="$STRAT"
164:         python scripts/run_baselines.py --config configs/default.yaml \
165:             --mc-path "artifacts/distractor_$STRAT/mc_dataset.json" likelihood.model=tfidf
166:         cp artifacts/main/baseline_summary.json "$RESULTS/baselines_$STRAT.json"
167:     done
168: 
169:     echo "=== PHASE 13: K-sensitivity ==="
170:     for K in 2 3 5 6; do
171:         python scripts/build_mc_dataset.py --config configs/default.yaml \
172:             --output-dir "artifacts/k$K" data.K="$K" data.distractor_strategy=category_random
173:         python scripts/run_baselines.py --config configs/default.yaml \
174:             --mc-path "artifacts/k$K/mc_dataset.json" likelihood.model=tfidf
175:         cp artifacts/main/baseline_summary.json "$RESULTS/baselines_k$K.json"
176:     done
177: 
178: else
179:     ####################################################################
180:     # PARALLEL MODE
181:     ####################################################################
182:     echo "=== WAVE 1: Independent phases (3 parallel tracks) ==="
183:     echo ""
184: 
185:     PIDS=()
186: 
187:     # Track A: Baselines (writes artifacts/main/baseline_summary.json)
188:     (
189:         run_phase "2" python scripts/run_baselines.py \
190:             --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
191:         cp artifacts/main/baseline_summary.json "$RESULTS/baselines_tfidf.json"
192:     ) &
193:     PIDS+=($!)
194: 
195:     # Track B: PPO training (writes artifacts/main/ppo_model.zip)
196:     (
197:         run_phase "3" python scripts/train_ppo.py \
198:             --config configs/default.yaml --mc-path "$MC" --seed 13 --deterministic-eval likelihood.model=tfidf
199:         cp artifacts/main/ppo_summary.json "$RESULTS/ppo_default.json"
200:         cp artifacts/main/ppo_model.zip "$RESULTS/ppo_model_default.zip"
201:     ) &
202:     PIDS+=($!)
203: 
204:     # Track C: T5 policy (writes checkpoints/ — no artifact race)
205:     (
206:         run_phase "5" python scripts/train_t5_policy.py \
207:             --config configs/t5_policy.yaml model.model_name="$T5_MODEL"
208:     ) &
209:     PIDS+=($!)
210: 
211:     echo "Waiting for Wave 1 (${#PIDS[@]} tracks)..."
212:     wait_all "${PIDS[@]}"
213:     echo ""
214: 
215:     echo "=== WAVE 2: Sequential post-Wave-1 phases (share artifacts/main/) ==="
216: 
217:     # Phase 4: Evaluate all (reads baseline_summary.json from Phase 2)
218:     run_phase "4" python scripts/evaluate_all.py \
219:         --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
220:     cp artifacts/main/evaluation_report.json "$RESULTS/eval_default.json"
221: 
222:     # Phase 6: Compare policies (needs Phase 3 PPO + Phase 5 T5)
223:     run_phase "6" python scripts/compare_policies.py \
224:         --mlp-checkpoint artifacts/main/ppo_model \
225:         --t5-checkpoint checkpoints/ppo_t5/best_model \
226:         --mc-path "$MC" \
227:         --output "$RESULTS/t5_comparison.json"
228: 
229:     # Phase 11: Expected Wins eval (writes evaluation_report.json)
230:     run_phase "11" python scripts/evaluate_all.py \
231:         --config configs/default.yaml --mc-path "$MC" \
232:         likelihood.model=tfidf environment.reward_mode=expected_wins environment.opponent_buzz_model.type=logistic
233:     cp artifacts/main/evaluation_report.json "$RESULTS/eval_ew_logistic.json"
234: 
235:     # Phase 15: Belief mode comparison (writes baseline_summary.json)
236:     run_phase "15" python scripts/run_baselines.py \
237:         --config configs/default.yaml --mc-path "$MC" \
238:         environment.belief_mode=sequential_bayes likelihood.model=tfidf
239:     cp artifacts/main/baseline_summary.json "$RESULTS/baselines_seqbayes.json"
240: 
241:     echo ""
242:     echo "=== WAVE 3: PPO ablations (sequential — share artifacts/main/) ==="
243: 
244:     echo "[Phase 14a] reward_mode=simple"
245:     python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
246:         --seed 13 --deterministic-eval likelihood.model=tfidf environment.reward_mode=simple
247:     cp artifacts/main/ppo_summary.json "$RESULTS/ppo_simple.json"
248: 
249:     echo "[Phase 14b] reward_mode=human_grounded"
250:     python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
251:         --seed 13 --deterministic-eval likelihood.model=tfidf environment.reward_mode=human_grounded
252:     cp artifacts/main/ppo_summary.json "$RESULTS/ppo_human_grounded.json"
253: 
254:     echo "[Phase 16] policy_mode=stop_only"
255:     python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
256:         --seed 13 --deterministic-eval --policy-mode stop_only likelihood.model=tfidf
257:     cp artifacts/main/ppo_summary.json "$RESULTS/ppo_stop_only.json"
258: 
259:     echo "[Phase 17] end_mode=no_buzz"
260:     python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
261:         --seed 13 --deterministic-eval likelihood.model=tfidf environment.end_mode=no_buzz environment.no_buzz_reward=-0.25
262:     cp artifacts/main/ppo_summary.json "$RESULTS/ppo_no_buzz.json"
263: 
264:     echo ""
265:     echo "=== WAVE 4: K-sensitivity (sequential — writes artifacts/main/baseline_summary.json) ==="
266: 
267:     for K in 2 3 5 6; do
268:         echo "[Phase 13-k$K] Building K=$K dataset..."
269:         run_phase "13-k$K" python scripts/build_mc_dataset.py \
270:             --config configs/default.yaml \
271:             --output-dir "artifacts/k$K" data.K="$K" data.distractor_strategy=category_random
272:         run_phase "13-k${K}-baselines" python scripts/run_baselines.py \
273:             --config configs/default.yaml \
274:             --mc-path "artifacts/k$K/mc_dataset.json" likelihood.model=tfidf
275:         cp artifacts/main/baseline_summary.json "$RESULTS/baselines_k$K.json"
276:     done
277: 
278: fi
279: 
280: ########################################################################
281: # FINAL SUMMARY
282: ########################################################################
283: echo ""
284: echo "============================================================"
285: echo "PIPELINE COMPLETE"
286: echo "============================================================"
287: echo ""
288: echo "Results directory:"
289: ls -1 "$RESULTS"/*.json 2>/dev/null | while read f; do echo "  $(basename $f)"; done
290: echo ""
291: echo "Artifacts:"
292: for d in artifacts/main artifacts/k* artifacts/distractor_*; do
293:     [ -d "$d" ] && echo "  $d/ — $(ls $d/*.json 2>/dev/null | wc -l) JSON files"
294: done
295: echo ""
296: echo "Checkpoints:"
297: ls -d checkpoints/*/best_model 2>/dev/null | while read d; do echo "  $d/"; done
298: echo ""
299: echo "Final comparison table:"
300: python3 -c "
301: import json, glob
302: for f in sorted(glob.glob('results/*.json')):
303:     s = json.load(open(f))
304:     name = f.split('/')[-1].replace('.json', '')
305:     if 'full_eval' in s:
306:         fe = s['full_eval']
307:         print(f'{name}: acc={fe.get(\"buzz_accuracy\", \"N/A\")}, S_q={fe.get(\"mean_sq\", \"N/A\")}')
308:     elif 't5_policy' in s:
309:         for k in ('mlp_policy', 't5_policy'):
310:             if k in s:
311:                 m = s[k]
312:                 print(f'{name}/{k}: acc={m.get(\"accuracy\", \"N/A\")}, S_q={m.get(\"mean_sq\", \"N/A\")}')
313:     elif 'softmax_profile' in s:
314:         sp = s['softmax_profile']
315:         best = max(sp.items(), key=lambda x: x[1].get('mean_sq', 0), default=('N/A', {}))
316:         print(f'{name}: best_threshold={best[0]}, S_q={best[1].get(\"mean_sq\", \"N/A\")}')
317:     else:
318:         acc = s.get('buzz_accuracy', s.get('accuracy', 'N/A'))
319:         sq = s.get('mean_sq', 'N/A')
320:         print(f'{name}: acc={acc}, S_q={sq}')
321: "
````

## File: AGENTS.md
````markdown
  1: # AGENTS.md
  2: 
  3: Canonical repo contract for all coding agents (Claude, Copilot, Cursor, etc.).
  4: 
  5: ## Project Overview
  6: 
  7: Stanford CS234 final project: a unified quiz bowl RL buzzer system with two tracks. The belief-feature pipeline builds MC tossups, scores answer profiles with TF-IDF / SBERT / T5 / optional OpenAI / optional DSPy, trains or compares buzzers, and evaluates with S_q, Expected Wins, and calibration metrics. The T5 policy pipeline provides supervised warm-start and PPO for an end-to-end text policy using factorized stop/answer semantics (`P(WAIT)` and `P(BUZZ_i) = P(BUZZ) * P(answer_i | BUZZ)`). Three opt-in extensions: Expected Wins reward mode, variable-K answer choices, and DSPy integration. Additional opt-in feature-port surfaces are available for stop-only PPO (`scripts/train_ppo.py --policy-mode stop_only`) and no-buzz horizon behavior (`environment.end_mode: no_buzz`). `qanta-buzzer` is the canonical repo. qb-rl compatibility is preserved through additive shims rather than structural rewrites.
  8: 
  9: ## Setup
 10: 
 11: Requires Python >= 3.11.
 12: 
 13: ```bash
 14: python3 -m venv .venv && source .venv/bin/activate
 15: pip install -U pip && pip install -e .
 16: ```
 17: 
 18: Optional extras:
 19: 
 20: ```bash
 21: pip install -e '.[openai]'    # OpenAI embedding support
 22: pip install -e '.[maskable]'  # MaskablePPO for variable-K
 23: pip install -e '.[dspy]'      # DSPy LM-based scoring
 24: ```
 25: 
 26: ## Architecture
 27: 
 28: | Package | Purpose |
 29: |---------|---------|
 30: | `qb_data/` | Data loading, answer profiles, stratified splits, MC construction, DSPy profiles |
 31: | `qb_env/` | Gymnasium environment, text wrapper, opponent models, optional StopOnlyEnv wrapper, qb-rl shims |
 32: | `models/` | Likelihood models (TF-IDF, SBERT, T5, OpenAI, DSPy), belief features, T5 policy |
 33: | `agents/` | Threshold, softmax-profile, sequential Bayes, PPO wrapper |
 34: | `evaluation/` | S_q metric, Expected Wins, calibration, control experiments, plotting |
 35: | `scripts/` | Pipeline entrypoints, DSPy compile, shared helpers |
 36: | `training/` | T5 policy supervised + PPO trainers, hazard bridge utilities |
 37: | `configs/` | YAML configuration files (default, smoke, t5_policy) |
 38: 
 39: ## Testing
 40: 
 41: 342 tests across 24 test files (3 skipped when optional extras not installed).
 42: 
 43: ```bash
 44: pytest                    # full suite
 45: pytest tests/test_qb_rl_bridge.py tests/test_factories.py tests/test_ppo_buzzer.py  # focused bridge/runtime checks
 46: scripts/ci.sh             # CI entry point (runs pytest, exits nonzero on failure)
 47: ```
 48: 
 49: ## Smoke Pipeline
 50: 
 51: Four-stage belief-feature smoke workflow. `--smoke` selects `configs/smoke.yaml` and writes outputs to `artifacts/smoke/`.
 52: 
 53: ```bash
 54: python scripts/build_mc_dataset.py --smoke
 55: python scripts/run_baselines.py --smoke
 56: python scripts/train_ppo.py --smoke
 57: python scripts/evaluate_all.py --smoke
 58: ```
 59: 
 60: Or run all four stages via the wrapper script:
 61: 
 62: ```bash
 63: scripts/manual-smoke.sh
 64: ```
 65: 
 66: ## Full Pipeline
 67: 
 68: For the core pipeline and scripted extensions at full scale with 4-wave parallel execution:
 69: 
 70: ```bash
 71: bash scripts/run_full_pipeline.sh --t5-model t5-base
 72: ```
 73: 
 74: The script forces `likelihood.model=tfidf` for all belief-feature phases. Phases 7, 8, 10, 11 (EW PPO), 12, 18, 19 require manual execution. See `docs/full-pipeline-runbook.md` for phase-by-phase details.
 75: 
 76: All pipeline scripts accept positional config overrides (e.g. `likelihood.model=tfidf`).
 77: 
 78: ## T5 Policy Pipeline
 79: 
 80: ```bash
 81: python scripts/train_t5_policy.py --config configs/t5_policy.yaml
 82: python scripts/compare_policies.py --config configs/t5_policy.yaml
 83: ```
 84: 
 85: Notes:
 86: `scripts/train_t5_policy.py` parses `--hazard-pretrain`, `--beta-terminal`, and `--freeze-answer-head` for the future hazard bridge. `--hazard-pretrain` intentionally raises `NotImplementedError` until that loop is implemented.
 87: 
 88: ## Configuration
 89: 
 90: | Config | Purpose |
 91: |--------|---------|
 92: | `configs/default.yaml` | Full runs with T5-large likelihood and 100k PPO timesteps |
 93: | `configs/smoke.yaml` | Quick tests: 50 questions, TF-IDF likelihood, 3k PPO timesteps |
 94: | `configs/t5_policy.yaml` | T5 policy pipeline: model, supervised, PPO, and data sections |
 95: 
 96: qb-rl config aliases are supported (e.g., `data.dataset`, `likelihood.sbert_name`, `environment.reward` as alias for `reward_mode`).
 97: 
 98: Additional environment options:
 99: - `environment.end_mode: force_commit|no_buzz` controls horizon behavior
100: - `environment.no_buzz_reward` is only used when `end_mode: no_buzz`
101: 
102: ## Compatibility Bridge
103: 
104: Old qb-rl import paths that still resolve:
105: 
106: - `qb_env.data_loader`, `qb_env.mc_builder`, `qb_env.text_utils`
107: - `models.answer_profiles`
108: - `agents.softmax_profile_buzzer`
109: 
110: OpenAI support is opt-in only. Default local workflows stay offline-friendly and do not require the `openai` package or `OPENAI_API_KEY`.
111: 
112: ## Conventions
113: 
114: - NumPy-style docstrings with Parameters/Returns sections
115: - RL notation: `V` (value), `R` (reward), `T` (transition), `gamma` (discount), `s`/`a` (state/action)
116: - Prefer NumPy/PyTorch vectorized operations over loops in ML code
117: - Explicit seeds for reproducibility (use 1, 2, 3 for multi-seed runs)
````

## File: README.md
````markdown
  1: # Quiz Bowl RL Buzzer (Unified)
  2: 
  3: Unified CS234 final project codebase for quiz bowl buzzing under incremental clues.
  4: 
  5: This repo keeps `qanta-buzzer` as the canonical implementation while preserving a qb-rl compatibility bridge:
  6: 
  7: - Modular belief-feature pipeline: `qb_data/` -> `models/` -> `qb_env/` -> `agents/` -> `evaluation/` -> `scripts/`
  8: - T5 policy pipeline: supervised warm-start and PPO for end-to-end text-based buzzing
  9: - qb-rl-compatible import/config shims for older notebooks and scripts
 10: - Optional OpenAI embedding support (`likelihood.model: openai`, `data.distractor_strategy: openai_profile`)
 11: 
 12: ## Setup
 13: 
 14: Requires Python >= 3.11.
 15: 
 16: ```bash
 17: python3 -m venv .venv
 18: source .venv/bin/activate
 19: pip install -U pip
 20: pip install -e .
 21: ```
 22: 
 23: Optional extras:
 24: 
 25: ```bash
 26: pip install -e '.[openai]'    # OpenAI embedding support (requires OPENAI_API_KEY)
 27: pip install -e '.[maskable]'  # MaskablePPO for variable-K (sb3-contrib)
 28: pip install -e '.[dspy]'      # DSPy LM-based scoring
 29: ```
 30: 
 31: ## Main Workflows
 32: 
 33: ### Belief-feature / PPO pipeline
 34: 
 35: The canonical four-stage smoke pipeline:
 36: 
 37: ```bash
 38: python scripts/build_mc_dataset.py --smoke
 39: python scripts/run_baselines.py --smoke
 40: python scripts/train_ppo.py --smoke
 41: python scripts/evaluate_all.py --smoke
 42: ```
 43: 
 44: `--smoke` selects `configs/smoke.yaml` and writes outputs to `artifacts/smoke/`. Drop `--smoke` for full runs (uses `configs/default.yaml`, writes to `artifacts/main/`).
 45: 
 46: The smoke config uses tuned reward settings (`wait_penalty=0.05`, `early_buzz_penalty=0.2`, `ppo.seed=13`, `ppo.total_timesteps=3000`).
 47: 
 48: `train_ppo.py` also accepts `--seed` to override the PPO/environment seed, and `--stochastic-eval` / `--deterministic-eval` to control post-training evaluation mode.
 49: 
 50: ### T5 policy pipeline
 51: 
 52: Trains a T5-based policy with supervised warm-start followed by PPO fine-tuning:
 53: 
 54: ```bash
 55: python scripts/train_t5_policy.py --config configs/t5_policy.yaml
 56: python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke  # quick test with t5-small
 57: ```
 58: 
 59: The T5 pipeline uses its own config (`configs/t5_policy.yaml`) which defines `model`, `supervised`, `ppo`, and `data` sections. It does not inherit `environment` or `likelihood` settings from the belief-feature configs -- the T5 PPO trainer uses default reward settings (`wait_penalty=0.1`).
 60: 
 61: The T5 policy uses factorized action semantics: the wait head models `P(WAIT)` vs `P(BUZZ)`, the answer head models `P(answer | BUZZ)`, and the flat action distribution is `P(WAIT)` plus `P(BUZZ_i) = P(BUZZ) * P(answer_i | BUZZ)`.
 62: 
 63: The CLI also reserves `--hazard-pretrain`, `--beta-terminal`, and `--freeze-answer-head` for an experimental hazard-style warm-start bridge. Those flags are parsed, but `--hazard-pretrain` currently raises `NotImplementedError` until the training loop is wired.
 64: 
 65: ### Policy comparison
 66: 
 67: ```bash
 68: python scripts/compare_policies.py --t5-checkpoint checkpoints/ppo_t5/best_model
 69: ```
 70: 
 71: Compares the MLP belief-feature policy against the T5 end-to-end policy on the same test set. Accuracy and buzz-position metrics are directly comparable. ECE and Brier are computed identically (top-answer probability at buzz time). S_q and reward comparisons are qualitative because the two architectures use different confidence semantics (belief-sigmoid vs wait-head probability) and different reward settings (config-driven vs T5-pipeline defaults).
 72: 
 73: ### Full pipeline (parallel execution)
 74: 
 75: For the core pipeline and scripted extensions at full scale with automatic parallelism:
 76: 
 77: ```bash
 78: bash scripts/run_full_pipeline.sh --t5-model t5-base   # ~3-4 hrs on M3 Max
 79: ```
 80: 
 81: The script forces `likelihood.model=tfidf` for all belief-feature phases. Phases 7, 8, 10, 11 (EW PPO), 12, 18, 19 require manual execution. See `docs/full-pipeline-runbook.md` for the full 19-phase runbook.
 82: 
 83: ### Additional scripts
 84: 
 85: - `scripts/run_full_pipeline.sh` -- full 19-phase parallel pipeline with 4-wave DAG (forces tfidf)
 86: - `scripts/run_smoke_pipeline.py` -- runs all four smoke stages sequentially
 87: - `scripts/sweep_reward_shaping.py` -- grid sweep over `wait_penalty` and `early_buzz_penalty` with multi-seed evaluation
 88: - `scripts/train_ppo.py --policy-mode flat_kplus1|stop_only` -- optional stop-only PPO surface; default remains `flat_kplus1`
 89: - `generate_presentation.py` -- generates the Marp presentation slides
 90: 
 91: All pipeline scripts accept positional config overrides:
 92: 
 93: ```bash
 94: python scripts/run_baselines.py --smoke likelihood.model=tfidf
 95: python scripts/train_ppo.py --seed 13 environment.reward_mode=simple
 96: ```
 97: 
 98: ## Configuration
 99: 
100: Two primary YAML configs:
101: 
102: | Config | Purpose | Key reward settings |
103: |--------|---------|-------------------|
104: | `configs/default.yaml` | Full runs | `wait_penalty=0.05`, `early_buzz_penalty=0.2`, `buzz_incorrect=-0.5` |
105: | `configs/smoke.yaml` | Quick tests (50 questions) | Same as default except `buzz_incorrect=-1.0`, `total_timesteps=3000` |
106: | `configs/t5_policy.yaml` | T5 pipeline | Own `model`/`supervised`/`ppo`/`data` sections; no `environment` |
107: 
108: qb-rl config aliases are also supported: `data.dataset`, `data.dataset_config`, `likelihood.sbert_name`, `environment.reward` as an alias for `reward_mode`, etc.
109: 
110: For horizon behavior, `environment.end_mode` defaults to `force_commit` (legacy behavior). Set `environment.end_mode: no_buzz` with `environment.no_buzz_reward` to end the episode without forcing a terminal answer.
111: 
112: ## Testing
113: 
114: 342 tests across 24 test files (3 skipped when optional extras not installed):
115: 
116: ```bash
117: pytest                    # full suite
118: pytest tests/test_agents.py tests/test_environment.py tests/test_ppo_buzzer.py  # quick iteration
119: ```
120: 
121: The test suite covers:
122: 
123: - Baseline agents (threshold, softmax-profile, sequential Bayes) and PPO wrapper
124: - Gymnasium environment behavior, reward modes (including Expected Wins), and belief computation
125: - Likelihood model factories (TF-IDF, SBERT, DSPy with offline-safe stubs)
126: - T5 policy model, supervised trainer, and PPO trainer
127: - Evaluation metrics (S_q, Expected Wins, ECE, Brier score, calibration at buzz, per-category accuracy)
128: - Dataset split reproducibility (cross-process determinism)
129: - Variable-K dataset construction and mixed-K integration
130: - Opponent buzz models (logistic, empirical)
131: - qb-rl compatibility bridge
132: - Text observation wrapper
133: 
134: ## Architecture
135: 
136: ```
137: qb_data/        Data loading, answer profiles, stratified splits, MC construction, DSPy profiles
138: qb_env/         Gymnasium environment, text wrapper, opponent models, optional StopOnlyEnv wrapper, qb-rl shims
139: models/         Likelihood models (TF-IDF, SBERT, T5, OpenAI, DSPy), belief features, T5 policy
140: agents/         Threshold, softmax-profile, sequential Bayes, PPO buzzer
141: evaluation/     S_q metric, Expected Wins, calibration, control experiments, plotting
142: scripts/        Pipeline entrypoints, DSPy compile, shared helpers
143: training/       T5 policy supervised + PPO trainers, hazard bridge utilities
144: configs/        YAML configuration files
145: artifacts/      Generated pipeline outputs (smoke/ and main/)
146: _legacy/        Pre-modularization prototypes (not installed)
147: ```
148: 
149: ## Compatibility Bridge
150: 
151: These old qb-rl import paths resolve in this repo:
152: 
153: - `qb_env.data_loader`, `qb_env.mc_builder`, `qb_env.text_utils`
154: - `models.answer_profiles`
155: - `agents.softmax_profile_buzzer`
156: 
157: The bridge is additive. `qb_data/` remains the canonical home for data loading and MC construction. OpenAI support is opt-in only -- default local workflows stay offline-friendly.
158: 
159: ## Documentation
160: 
161: - `docs/full-pipeline-runbook.md` -- deterministic 19-phase runbook with wall-time estimates and parallel execution
162: - `AGENTS.md` -- canonical repo contract for all coding agents (setup, architecture, testing, configuration)
163: - `CLAUDE.md` -- thin shim pointing to AGENTS.md with Claude-specific notes
164: - `walkthrough.md` -- end-to-end walkthrough exercising both pipelines (pre-remediation snapshot)
165: - `PRESENTATION.md` -- Marp presentation slides for the CS234 final project
166: - `.planning/` -- canonical project state, roadmap, architectural decisions, and remediation log
167: 
168: ## Extensions (opt-in)
169: 
170: Three opt-in extensions are available. All are disabled by default — the smoke pipeline and T5 smoke path work unchanged.
171: 
172: ### Expected Wins reward mode
173: 
174: Set `environment.reward_mode: expected_wins` and configure `environment.opponent_buzz_model` in YAML. Supports logistic and empirical (from `human_buzz_positions`) opponent models. Offline `expected_wins_score()` in `evaluation/metrics.py` uses the continuous formula: `V_self = g * R_correct + (1-g) * R_incorrect`.
175: 
176: ### Variable-K answer choices
177: 
178: Set `data.variable_K: true` and `data.min_K` / `data.max_K` in YAML. `MCBuilder` samples K per question. The env uses padded observations and `action_masks()`. Optional `MaskablePPO` via `pip install -e '.[maskable]'`.
179: 
180: ### DSPy integration (experimental)
181: 
182: Set `likelihood.model: dspy` and configure the `dspy` section in YAML. Requires `pip install -e '.[dspy]'`. Offline compile via `python scripts/optimize_dspy.py`. Does NOT integrate prompt optimization into PPO rollouts.
183: 
184: ## Legacy Prototype
185: 
186: The pre-modularization prototype (`main.py`, `environment.py`, `model.py`, `dataset.py`, `config.py`, etc.) has been moved to `_legacy/`. These files are not part of the installed package and are preserved only for reference. The modular `scripts/` pipeline above is the canonical workflow.
````

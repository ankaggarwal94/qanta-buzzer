"""Independent integration checks for the opt-in ACL train/infer/score protocol."""
import json

import pytest


def expansion():
    from qb_data import acl_protocol
    return acl_protocol


def corpus():
    labels = ["Ceres", "Juno", "Pallas", "Vesta", "Hygiea", "Eros"]
    questions = []
    for i, label in enumerate(labels):
        first = f"The asteroid {i}"
        last = first + " has a distinct orbit."
        options = [label, labels[(i + 1) % 4]] if i < 4 else [label, labels[0]]
        questions.append({
            "qid": str(i), "group_id": f"group{i}", "split": (["train"]*4+["val","test"])[i],
            "question": last, "prefixes": [{"text": first}, {"text": last}],
            "source": {"category": "Science", "canonical_answer": label},
            "answer": {"accepted": [label], "rejected": [], "prompt": []},
            "menus": [{"condition": "independent_pool", "gold_option_id": "A",
                       "options": [{"id": key, "text": text} for key, text in zip("AB", options)]}],
        })
    return ({"schema_version": "jane-paired-v1", "questions": questions},
            {"version": "asteroids-v1", "reviewer": "test fixture", "classes": {s: [] for s in labels}})


def prepare(data=None, oracle=None, **changes):
    api = expansion()
    dataset, answers = corpus()
    spec = api.ArmSpec.from_dict({"version": 1, "menu": "uniform_global", "k": 2,
                                  "seed": 13, "reward": {"arm": "baseline"}, **changes})
    return api.prepare(spec, data or dataset, oracle or answers)


def test_policy_projection_and_hidden_suffix_invariance():
    dataset, _ = corpus()
    first = prepare(menu="prefix_tfidf")
    row = dataset["questions"][-1]
    row["question"] += " PRIVATE_SUFFIX"
    row["prefixes"][-1]["text"] = row["question"]
    second = prepare(data=dataset, menu="prefix_tfidf")
    a, b = first.questions["test"][0], second.questions["test"][0]
    assert a.options == b.options
    assert a.option_profiles == a.options  # same public feature source for every slot
    api = expansion()
    prompt = api.policy_prompt(a.cumulative_prefixes[0], a.options)
    changed = list(second.public_jobs("test"))
    assert changed[0]["prompt"] == next(first.public_jobs("test"))["prompt"]
    assert "PRIVATE_SUFFIX" not in changed[0]["prompt"]
    assert "PRIVATE_SUFFIX" in changed[-1]["prompt"]
    assert "PRIVATE_SUFFIX" not in prompt
    payload = json.loads(prompt.rsplit("\n\n", 1)[1])
    assert set(payload) == {"question_prefix", "options"}
    assert [v["text"] for v in payload["options"]] == a.options
    assert [v["id"] for v in payload["options"]] == list(range(1, 3))


def test_source_groups_cannot_cross_splits_even_if_qids_differ():
    dataset, oracle = corpus()
    dataset["questions"][-1]["group_id"] = dataset["questions"][0]["group_id"]
    with pytest.raises(ValueError, match="group"):
        prepare(dataset, oracle)


def test_component_member_cannot_cross_splits_under_relabelled_group():
    dataset, oracle = corpus()
    for row in (dataset["questions"][0], dataset["questions"][-1]):
        row["source"]["component_source_ids"] = ["same-original-source"]
    with pytest.raises(ValueError, match="source"):
        prepare(dataset, oracle)


def test_source_alternate_accepted_answer_cannot_be_a_distractor():
    dataset, oracle = corpus()
    dataset["questions"][-1]["answer"]["accepted"].append("Juno")
    with pytest.raises(ValueError, match="accepted"):
        prepare(dataset, oracle)


def test_conflicting_prompt_rule_and_duplicate_train_text_fail_closed():
    dataset, oracle = corpus()
    q = dataset["questions"][-1]
    q["answer"]["prompt"] = ["Eros"]
    with pytest.raises(ValueError, match="prompt"):
        prepare(dataset, oracle)
    q["answer"]["prompt"] = []
    q["question"] = dataset["questions"][0]["question"]
    q["prefixes"] = dataset["questions"][0]["prefixes"]
    with pytest.raises(ValueError, match="overlap"):
        prepare(dataset, oracle)


def test_baseline_replays_source_order_and_checks_frozen_answer_key():
    dataset, oracle = corpus()
    result = prepare(dataset, oracle, menu="baseline")
    question = result.questions["test"][0]
    assert question.options == ["Eros", "Ceres"] and question.gold_index == 0
    dataset["questions"][-1]["menus"][0]["gold_option_id"] = "B"
    with pytest.raises(ValueError, match="gold"):
        prepare(dataset, oracle, menu="baseline")


def test_resume_rejects_different_payload_under_existing_identity(tmp_path):
    api = expansion()
    path = tmp_path / "manifest.json"
    api.publish_json(path, {"counter": 0})
    api.publish_json(path, {"counter": 0})
    with pytest.raises(ValueError, match="immutable"):
        api.publish_json(path, {"counter": 1})
    assert json.loads(path.read_bytes()) == {"counter": 0}


def test_bootstrap_only_at_truncation():
    from training.train_ppo_t5 import RolloutBuffer, RolloutStep
    buffer = RolloutBuffer()
    for bootstrap in (0., 4.):
        buffer.add_rollout([RolloutStep(observation_text="x", action=0, reward=1., done=True,
                                        value=2., log_prob=0., bootstrap_value=bootstrap)])
    buffer.compute_returns_and_advantages(.5, 1.)
    assert [step.return_ for step in buffer.get_all_steps()] == [1., 3.]


def test_train_prompt_export_and_scoring_share_every_actionable_prefix():
    from scripts.acl_expansion import Runtime
    p = prepare()
    runtime = Runtime(p)
    question = p.questions["test"][0]
    jobs = list(p.public_jobs("test"))
    env = runtime.environment([question], text=True)
    obs, _ = env.reset()
    assert obs == jobs[0]["prompt"]
    obs, _, term, trunc, _ = env.step(0)
    assert not term and not trunc and obs == jobs[1]["prompt"]
    choices = {jobs[0]["job_id"]: 0, jobs[1]["job_id"]: question.gold_index + 1}
    report = runtime.score_actions("test", choices)
    assert report[0]["correct"] and len(report[0]["steps"]) == 2
    with pytest.raises(ValueError, match="coverage"):
        runtime.score_actions("test", {})
    with pytest.raises(ValueError, match="action"):
        runtime.score_actions("test", {k: True for k in choices})

def test_runtime_replay_checks_steps_and_completed_episodes():
    from scripts.acl_expansion import Runtime
    runtime = Runtime(prepare())
    rows=[]
    env=runtime.environment(runtime.prepared.questions["train"], sink=rows)
    env.reset(options={"question_idx":0})
    env.step(0)
    env.step(1)
    runtime.replay_training(rows,"ppo",2)
    runtime.replay_training(rows,"t5",1)
    for policy,count,trace in [("ppo",2,[]),("ppo",2,rows[:1]),("t5",2,rows),("t5",1,rows[:1])]:
        with pytest.raises(ValueError,match="trace"):
            runtime.replay_training(trace,policy,count)


def test_partial_model_cache_is_refetched_then_reusable(tmp_path,monkeypatch):
    import huggingface_hub
    from scripts import acl_expansion as acl
    (tmp_path/"config.json").write_text("{}")
    calls=[]
    def download(name, *,revision,local_files_only=False,allow_patterns=None):
        calls.append(local_files_only)
        if not local_files_only:
            for name in ("model.safetensors","tokenizer_config.json","spiece.model","tokenizer.json"):
                (tmp_path/name).write_bytes(b"stub")
        return str(tmp_path)
    acl.snapshot.cache_clear()
    monkeypatch.setattr(huggingface_hub,"snapshot_download",download)
    try:
        assert acl.snapshot(acl.T5)==str(tmp_path)
        assert calls==[True,False]
        acl.snapshot.cache_clear()
        assert acl.snapshot(acl.T5)==str(tmp_path)
        assert calls==[True,False,True]
    finally:
        acl.snapshot.cache_clear()


def test_cli_accepts_dotted_overrides_and_rejects_ignored_selectors(tmp_path):
    from scripts.acl_expansion import main
    config="configs/acl_expansion/smoke.yaml"
    main(["export","--config",config,"--out",str(tmp_path/"jobs.json"),"arm.seed=3"])
    assert (tmp_path/"jobs.json").is_file()
    for flag,value in [("--reward","linear"),("--menu","baseline")]:
        with pytest.raises(SystemExit):
            main(["smoke","--config",config,"--out",str(tmp_path),flag,value])

@pytest.mark.parametrize("terminated,truncated,expected", [(False,True,4.),(True,False,0.),(True,True,0.)])
def test_rollout_collector_bootstraps_only_live_timeouts(tmp_path, terminated, truncated, expected):
    import torch
    from training.train_ppo_t5 import PPOTrainer
    from transformers import BatchEncoding

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.ones(1))
            self.device=torch.device("cpu")
            self.seen=[]
        def tokenizer(self, obs, **options):
            return BatchEncoding({"input_ids":torch.ones(1,1,dtype=torch.long),
                                  "attention_mask":torch.ones(1,1,dtype=torch.long)})
        def select_action(self, ids, mask, deterministic=False):
            return torch.tensor(0), {"values":torch.tensor(2.),"log_probs":torch.tensor(0.)}
        def forward(self, obs, return_value):
            self.seen.extend(obs)
            return None,None,torch.tensor(4.)

    class Env:
        def reset(self):
            return "start",{}
        def step(self, action):
            return "successor",1.,terminated,truncated,{}

    model=Model()
    trainer=PPOTrainer(model,[object()],[],{"checkpoint_dir":str(tmp_path)},
                       text_env_factory=lambda q: Env(),bootstrap_time_limits=True)
    steps=trainer.collect_rollouts(1).get_all_steps()
    assert steps[0].bootstrap_value==expected
    assert model.seen==(["successor"] if expected else [])


def test_export_stream_is_canonical_and_fails_atomically(tmp_path):
    from scripts.acl_expansion import _export_jobs
    from qb_data.acl_protocol import encode
    path=tmp_path/"jobs"
    items=[{"é":"text"},{"clues":"next"}]
    _export_jobs(path,iter(items))
    assert path.read_bytes()==encode(items)+b"\n"
    _export_jobs(path,iter(items))
    with pytest.raises(ValueError,match="immutable"):
        _export_jobs(path,iter([]))
    def broken():
        yield items[0]
        raise RuntimeError("generator failed")
    with pytest.raises(RuntimeError,match="generator failed"):
        _export_jobs(tmp_path/"partial",broken())
    assert list(tmp_path.iterdir()) == [path]

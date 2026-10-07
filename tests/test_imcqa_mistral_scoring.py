"""Fail-closed CPU tests; none claims GPU numerical validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import imcqa_mistral_scoring as scoring
from scripts import imcqa_mistral_design as design
from scripts import imcqa_protocol_design as old


class TokenizerFixture:
    """Character tokenizer with deliberately pinned one-token action suffixes."""
    def __init__(self, defect=None):
        self.defect = defect

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False, continue_final_message=False):
        rendered = "<user>" + messages[0]["content"] + "</user>"
        if continue_final_message:
            return rendered + ("" if self.defect == "space" else " ") + messages[-1]["content"]
        return rendered

    def __call__(self, text, *, add_special_tokens):
        ids = [ord(c) for c in text]
        if text[-1:] in "ABCDE" and text[:-1].endswith(scoring.ASSISTANT_PREFIX):
            ids[-1] = scoring.ACTION_TOKEN_IDS[text[-1]]
            if self.defect == "extra_token": ids.append(99)
            if self.defect == "prefix_change": ids[-2] = 99
            if self.defect == "wrong_id": ids[-1] += 1
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        if self.defect == "decode": return " A"
        return next((k for k, v in scoring.ACTION_TOKEN_IDS.items() if v == ids[0]), "?")


def context(ids):
    return {"scored_input_token_ids": ids, "option_token_ids": dict(scoring.ACTION_TOKEN_IDS)}


def contexts():
    return [context([1, 2 + i // 2, 100 + i]) for i in range(8)]


def fixture_package():
    """Construct a complete synthetic gold-free 20/120 development grid."""
    qids = [f"fixture-{i:03d}" for i in range(140)]
    selection = {"selected_qids": qids, "qid_split": {}, "qid_group": {}, "qid_text_sha256": {},
                 "qid_category": {}, "source_job_ids": {}, "original_score_ids": {}, "prompt_sha256": {},
                 "outcomes_used_for_selection": False}
    jobs = []
    for i, qid in enumerate(qids):
        split = "calibration" if i < 20 else "selection"
        full = f"{qid} clue two three four"
        selection["qid_split"][qid] = split
        selection["qid_group"][qid] = "group:" + qid
        selection["qid_text_sha256"][qid] = scoring.base.sha(scoring.base.canonical(design.tuned.normalized_tokens(full)))
        selection["qid_category"][qid] = "Fixture"
        for menu in scoring.CONDITIONS:
            for rnd in range(1, 6):
                sid = scoring.base.sha(f"{qid}:{menu}:{rnd}".encode())
                source = {"source_job_id": sid, "source_prompt_sha256": "f" * 64, "qid": qid,
                    "group_id": "group:" + qid, "split": split, "condition": menu, "menu_id": "fixed_1",
                    "prefix_id": "p" + str(2 * rnd), "fraction": rnd / 5, "round": rnd, "reward": old.REWARDS[rnd - 1]}
                payload = {"question_prefix": " ".join(full.split()[:rnd]),
                           "options": [{"id": k, "text": "Option " + k} for k in "ABCD"]}
                for rotation in range(4):
                    job = old._real_job(source, payload, "plain", rotation)
                    original_id = job["score_id"]
                    job.update(score_id=sid + f":mistral:plain:r{rotation}", source_score_id=original_id, score_index=len(jobs), execution="new")
                    jobs.append(job)
                    selection["source_job_ids"][job["score_id"]] = sid
                    selection["original_score_ids"][job["score_id"]] = original_id
                    selection["prompt_sha256"][job["score_id"]] = job["prompt_sha256"]
    return {"schema_version": design.SCHEMA, "protocol": design.PROTOCOL, "model": dict(design.MODEL),
            "stage": "development", "n_questions": 140, "jobs": jobs, "rewards": list(old.REWARDS),
            "wrong_reward": -1.0, "pass_reward": 0.0, "prefix_ids": list(old.PREFIX_IDS),
            "selection": selection, "selection_id": design.value_hash(selection), "main_dataset_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64, "config_sha256": design.file_hash(design.config_path()),
            "policy_lock_sha256": None, "development_receipt_sha256": None}


def write_complete_synthetic_evidence(root, package):
    """Create explicit synthetic evidence to exercise every saved-audit branch.

    This is a test oracle, never model-output or scientific-result evidence.
    """
    public = root / "public.json"
    public.write_bytes(scoring.base.canonical(package))
    run = root / "output/mistral7b"
    attempts = run / "attempts"
    attempts.mkdir(parents=True)
    write = lambda name, value: scoring.base.write_once(run / name, value)
    originals = [{"scored_input_token_ids": [1, 2, i + 100], "option_token_ids": dict(scoring.ACTION_TOKEN_IDS),
                  "scored_context_sha256": scoring.base.sha(str(i).encode()),
                  "rendered_prompt_sha256": scoring.base.sha((str(i) + "rendered").encode())} for i in range(5600)]
    order = scoring.paired_order(originals)
    jobs = [package["jobs"][i] for i in order]
    ctx = [originals[i] for i in order]
    output = {"logits": [3.0, 2.0, 1.0, 0.0, -1.0], "vocabulary_logsumexp": 4.0,
              "unconstrained_top_logit": 3.0, "unconstrained_top_token_id": scoring.ACTION_TOKEN_IDS["A"],
              "all_five_action_token_vocabulary_mass": 0.5}
    rows = [{"schema_version": scoring.SCORE_SCHEMA, **{k: v for k, v in j.items() if k != "prompt"}, **c,
             **scoring.action_statistics(output["logits"], j["allowed_actions"], j["option_source_ids"]),
             **{k: v for k, v in output.items() if k != "logits"}, "model_tag": scoring.MODEL_TAG}
            for j, c in zip(jobs, ctx)]
    (run / "scores.jsonl").write_bytes(b"".join(scoring.base.canonical(r) for r in rows))
    offsets = scoring.production_offsets(5600)
    qids = scoring.replay_qids(jobs)
    write("plan.json", {"ordered_score_ids": [j["score_id"] for j in jobs], "input_sha256": scoring.base.file_hash(public),
        "context_sha256": [c["scored_context_sha256"] for c in ctx], "token_counts": [3] * 5600,
        "new_rows": 5600, "reused_rows": 0, "production_diagnostic_offsets": offsets, "live_replay_qids": qids})
    hashes = dict.fromkeys(scoring.MODEL_FILES, "a" * 64)
    hashes.update(scoring.TOKENIZER_HASHES)
    metadata = {"protocol": scoring.PROTOCOL, "stage": "development", "model": scoring.MODEL_NAME,
        "revision": scoring.MODEL_REVISION, "dtype": "float32", "loaded_dtype": "bfloat16", "attention": "eager",
        "tf32": False, "seed": 1, "generation": False, "sampling": False,
        "chat_template_sha256": scoring.CHAT_TEMPLATE_SHA256, "assistant_prefix": scoring.ASSISTANT_PREFIX,
        "action_token_ids": scoring.ACTION_TOKEN_IDS, "source_commit": "a" * 40,
        "source_files_sha256": scoring.source_identity(), "public_input_sha256": scoring.base.file_hash(public),
        "batch_size": 8, "cached": True, "native_assistant_boundary": "one ASCII space before JSON action prefix",
        "versions": scoring.REQUIRED_VERSIONS, "model_files_sha256": hashes}
    write("metadata.json", metadata)
    scoring.base.write_once(run.parent / "cache_prepare_receipt.json", {"schema_version": "imcqa-mistral-cache-v1", "status": "complete",
        "chat_template_sha256": scoring.CHAT_TEMPLATE_SHA256, "model_receipts": {scoring.MODEL_TAG:
            {"model": scoring.MODEL_NAME, "revision": scoring.MODEL_REVISION, "model_files_sha256": hashes}}})
    cfg = SimpleNamespace(model_type="mistral", _attn_implementation="eager", sliding_window=None, max_position_embeddings=32768)
    write("model_config_validation.json", scoring.validate_model_config(cfg, ctx))
    write("dtype_promotion.json", {"all_checks_passed": True, "sampled_values_preserved_exactly": True,
        "all_floating_tensors_fp32": True, "original": {"fixture_weight": {"dtype": "torch.bfloat16"}},
        "promoted_dtypes": {"fixture_weight": "torch.float32"}})
    benchmark = scoring.projection_with_validation(128, 5600, 16.5, 1600, [1.0] * 16, [0.01] * 10,
        production_diagnostic_seconds=0.5, checkpoint_seconds=[])
    benchmark["checkpoint_timing_records"] = []
    write("attempts/000_benchmark.json", benchmark)
    di = scoring.diagnostic_indices(jobs, ctx)
    dj = [jobs[i] for i in di]
    values = [output] * len(di)
    gate = scoring.numeric_agreement(values, values, dj)
    diagnostic = {"score_ids": [j["score_id"] for j in dj], "cached": values, "uncached": values,
                  "singles": values, "replay": values, "permuted_aligned": values}
    write("attempts/000_diagnostics_raw.json", diagnostic)
    write("attempts/000_diagnostics.json", {**diagnostic, "gates": {k: gate for k in ("cached_uncached", "cached_single", "permutation", "permutation_single")}})
    pjobs = [jobs[i] for offset in offsets for i in range(offset, offset + 8)]
    production = {"score_ids": [j["score_id"] for j in pjobs], "production": [output] * 24,
        "single_reference": [output] * 24, "diagnostic": [output] * 24,
        "offsets": offsets, "actual_production_batches": True, "batch_size": 8}
    pgate = scoring.numeric_agreement([output] * 24, [output] * 24, pjobs)
    write("attempts/000_production_diagnostics_raw.json", production)
    write("attempts/000_production_diagnostics.json", {**production, "gate": pgate, "single_gate": pgate})
    for i, offset in enumerate(offsets):
        write(f"attempts/000_production_batch_{offset:05d}_raw.json", {k: production[k][i * 8:(i + 1) * 8]
            for k in ("score_ids", "production", "single_reference", "diagnostic")})
    bykey = {(j["qid"], j["condition"], j["rotation"], j["round"]): j for j in jobs}
    live = []
    for qid in qids:
        for menu in scoring.CONDITIONS:
            for rotation in (0, 2):
                for rnd in range(1, 6):
                    job = bykey[qid, menu, rotation, rnd]
                    row = {k: job[k] for k in ("qid", "condition", "rotation", "round", "score_id")}
                    row["agreement"] = scoring.numeric_agreement([output], [output], [job])
                    write(f"attempts/000_live_{len(live):03d}_raw.json", {**{k: row[k] for k in ("qid", "condition", "rotation", "round")},
                        "score_ids": [row["score_id"]], "production": [output], "reference": [output], "live": output})
                    live.append(row)
    write("attempts/000_live_trajectories.json", {"qids": qids, "rows": live})
    receipt = {"protocol": scoring.PROTOCOL, "stage": "development", "model_tag": scoring.MODEL_TAG,
        "model": scoring.MODEL_NAME, "revision": scoring.MODEL_REVISION, "status": "complete",
        "public_input_sha256": scoring.base.file_hash(public), "expected_rows": 5600, "completed_rows": 5600,
        "total_contexts": 5600, "attempt": 0, "automatic_retries": 0, "sampling": False, "generation": False,
        "reused_rows": 0, "batch_size": 8, "cached": True, "max_seconds": 1740, "source_commit": "a" * 40,
        "scores_sha256": scoring.base.file_hash(run / "scores.jsonl"), "benchmark": benchmark,
        "production_numerical_gate": pgate, "production_single_gate": pgate}
    write("receipt.json", receipt)
    write("attempts/000_receipt.json", receipt)
    return public, run


class TestMistralScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = fixture_package()

    def test_native_space_is_preserved(self):
        result = scoring.prepare_context(TokenizerFixture(), {"prompt": "hello"})
        self.assertEqual(result["option_token_ids"], scoring.ACTION_TOKEN_IDS)
        self.assertEqual(result["scored_context_sha256"], scoring.base.sha(('<user>hello</user> ' + scoring.ASSISTANT_PREFIX).encode()))

    def test_reject_non_native_boundary(self):
        with self.assertRaisesRegex(ValueError, "native assistant boundary"):
            scoring.prepare_context(TokenizerFixture("space"), {"prompt": "hello"})

    def test_reject_multi_token_label(self):
        with self.assertRaises(ValueError):
            scoring.prepare_context(TokenizerFixture("extra_token"), {"prompt": "hello"})

    def test_reject_label_prefix_retokenization(self):
        with self.assertRaises(ValueError):
            scoring.prepare_context(TokenizerFixture("prefix_change"), {"prompt": "hello"})

    def test_reject_inexact_decode(self):
        with self.assertRaises(ValueError):
            scoring.prepare_context(TokenizerFixture("decode"), {"prompt": "hello"})

    def test_reject_wrong_token_identity(self):
        with self.assertRaises(ValueError):
            scoring.prepare_context(TokenizerFixture("wrong_id"), {"prompt": "hello"})

    def test_reject_token_limit(self):
        with self.assertRaisesRegex(ValueError, "token limit"):
            scoring.prepare_context(TokenizerFixture(), {"prompt": "x" * 2048})

    def test_valid_full_package_and_exact_cap(self):
        self.assertEqual(len(scoring.validate_run_package(self.package, 1740)), 5600)

    def test_wrong_runtime_cap_rejected(self):
        with self.assertRaisesRegex(ValueError, "runtime cap"):
            scoring.validate_run_package(self.package, 1741)

    def test_gold_field_rejected(self):
        package = copy.deepcopy(self.package)
        package["jobs"][0]["gold_option_id"] = "A"
        with self.assertRaises(ValueError):
            scoring.validate_run_package(package, 1740)

    def test_partial_grid_rejected(self):
        package = copy.deepcopy(self.package)
        package["jobs"].pop()
        with self.assertRaises(ValueError):
            scoring.validate_run_package(package, 1740)

    def test_reuse_rejected(self):
        package = copy.deepcopy(self.package)
        package["jobs"][0]["execution"] = "reuse"
        with self.assertRaises(ValueError):
            scoring.validate_run_package(package, 1740)

    def test_model_identity_rejected(self):
        package = copy.deepcopy(self.package)
        package["model"]["name"] = "Qwen/Qwen2.5-7B-Instruct"
        with self.assertRaises(ValueError):
            scoring.validate_run_package(package, 1740)

    def test_exact_cache_hash_allowlist(self):
        hashes = dict.fromkeys(scoring.MODEL_FILES, "a" * 64)
        hashes.update(scoring.TOKENIZER_HASHES)
        scoring.validate_model_file_hashes(hashes)
        hashes["consolidated.safetensors"] = "b" * 64
        with self.assertRaises(ValueError): scoring.validate_model_file_hashes(hashes)

    def test_missing_sentencepiece_model_rejected(self):
        hashes = dict.fromkeys(scoring.MODEL_FILES, "a" * 64)
        hashes.update(scoring.TOKENIZER_HASHES)
        hashes.pop("tokenizer.model")
        with self.assertRaises(ValueError): scoring.validate_model_file_hashes(hashes)

    def test_changed_tokenizer_hash_rejected(self):
        hashes = dict.fromkeys(scoring.MODEL_FILES, "a" * 64)
        hashes.update(scoring.TOKENIZER_HASHES)
        hashes["tokenizer.model"] = "b" * 64
        with self.assertRaises(ValueError): scoring.validate_model_file_hashes(hashes)

    def test_window_none_is_safe_for_global_bound(self):
        config = SimpleNamespace(model_type="mistral", _attn_implementation="eager", sliding_window=None, max_position_embeddings=32768)
        result = scoring.validate_model_config(config, contexts())
        self.assertEqual(result["maximum_physical_cache_width"], 3)

    def test_too_small_window_rejected(self):
        config = SimpleNamespace(model_type="mistral", _attn_implementation="eager", sliding_window=2, max_position_embeddings=32768)
        with self.assertRaisesRegex(ValueError, "sliding window"):
            scoring.validate_model_config(config, contexts())

    def test_window_bound_covers_diagnostic_regrouping(self):
        ctx = [context([1] * 10 + [i + 2]) for i in range(8)]
        ctx += [context([1, i + 2] + [50] * 20) for i in range(8)]
        config = SimpleNamespace(model_type="mistral", _attn_implementation="eager", sliding_window=25, max_position_embeddings=32768)
        # Production batches need 11 or22 physical positions; diagnostics may
        # combine a10-token prefix with a21-token suffix, requiring31 positions.
        with self.assertRaisesRegex(ValueError, "sliding window"):
            scoring.validate_model_config(config, ctx)

    def test_other_model_or_attention_rejected(self):
        for model, attention in (("qwen2", "eager"), ("mistral", "sdpa")):
            config = SimpleNamespace(model_type=model, _attn_implementation=attention)
            with self.assertRaises(ValueError): scoring.validate_model_config(config, contexts())

    def test_pair_permutation_alignment_and_batch_bound(self):
        calls = []
        ctx = [{"i": i} for i in range(18)]
        def fake(torch, model, tokenizer, batch, **kwargs):
            calls.append((len(batch), kwargs))
            return [{"i": c["i"]} for c in batch]
        with mock.patch.object(scoring, "forward", fake):
            result = scoring.forward_batches(None, None, None, ctx, cached=True, permute=True)
        self.assertEqual(result, ctx)
        self.assertEqual([n for n, _ in calls], [8, 8, 2])
        self.assertTrue(all(kwargs["batch_size"] == 8 for _, kwargs in calls))

    def test_numeric_gate_rejects_argmax_change(self):
        job = {"allowed_actions": "ABCD", "option_source_ids": dict(zip("ABCD", "ABCD"))}
        with self.assertRaises(ValueError):
            scoring.numeric_agreement([{"logits": [1, 1.0001, 0, 0, 0]}], [{"logits": [1.0001, 1, 0, 0, 0]}], [job])

    def test_numeric_gate_rejects_nonfinite(self):
        job = {"allowed_actions": "ABCD", "option_source_ids": dict(zip("ABCD", "ABCD"))}
        with self.assertRaises(ValueError):
            scoring.numeric_agreement([{"logits": [1, float("nan"), 0, 0, 0]}], [{"logits": [1, 0, 0, 0, 0]}], [job])

    def test_raw_comparison_bound_to_actual_scores(self):
        jobs = {"x": {"allowed_actions": "ABCD", "option_source_ids": dict(zip("ABCD", "ABCD"))}}
        output = {"logits": [3, 2, 1, 0, -1]}
        record = {"score_ids": ["x"], "left": [output], "right": [output]}
        self.assertTrue(scoring.verify_numeric_record(record, "left", ("right",), jobs, {"x": output})["right"]["passed"])
        with self.assertRaises(ValueError):
            scoring.verify_numeric_record(record, "left", ("right",), jobs, {"x": {"logits": [0, 3, 2, 1, -1]}})

    def test_raw_comparison_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError, "identities"):
            scoring.verify_numeric_record({"score_ids": ["x", "x"]}, "left", (), {"x": {}}, {})

    def test_evidence_written_before_rejection(self):
        job = {"score_id": "x", "allowed_actions": "ABCD", "option_source_ids": dict(zip("ABCD", "ABCD"))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            with self.assertRaises(ValueError):
                scoring.record_numeric_gate(path, [{"logits": [3, 2, 1, 0, -1]}], [{"logits": [0, 3, 2, 1, -1]}], [job])
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text())["score_ids"], ["x"])

    def test_run_rejects_qwen_and_bad_commit_before_any_files(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for tag, commit in (("qwen7b", "a" * 40), ("mistral7b", "bad"), ("mistral7b", None)):
                with self.assertRaises(ValueError):
                    scoring.run_scoring(tag, p / "missing", "a" * 64, p / "cache", p / "out", source_commit=commit, max_seconds=1740)
            self.assertFalse((p / "out").exists())

    def test_run_rejects_hash_change_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "public.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                scoring.run_scoring("mistral7b", p / "public.json", "0" * 64, p / "cache", p / "out", source_commit="a" * 40, max_seconds=1740)
            self.assertFalse((p / "out").exists())

    def test_failed_worker_preserves_receipt_and_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            public = p / "public.json"
            public.write_bytes(scoring.base.canonical(self.package))
            with mock.patch.dict(sys.modules, {"torch": None}):
                result = scoring.run_scoring("mistral7b", public, scoring.base.file_hash(public), p / "cache", p / "out", source_commit="a" * 40, max_seconds=1740)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["automatic_retries"], 0)
            self.assertEqual(result["completed_rows"], 0)
            self.assertTrue((p / "out/attempts/000_receipt.json").exists())
            with self.assertRaisesRegex(ValueError, "complete frozen worker"):
                scoring.validate_completed_run(public, p / "out")
            with self.assertRaises(FileExistsError):
                scoring.run_scoring("mistral7b", public, scoring.base.file_hash(public), p / "cache", p / "out", source_commit="a" * 40, max_seconds=1740)

    def test_complete_receipt_with_wrong_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            public = p / "public.json"
            public.write_bytes(scoring.base.canonical(self.package))
            out = p / "out"
            (out / "attempts").mkdir(parents=True)
            receipt = {"protocol": scoring.PROTOCOL, "stage": "development", "model_tag": scoring.MODEL_TAG,
                "model": scoring.MODEL_NAME, "revision": scoring.MODEL_REVISION, "status": "complete",
                "public_input_sha256": scoring.base.file_hash(public), "expected_rows": 5600, "completed_rows": 5600,
                "total_contexts": 5600, "attempt": 0, "automatic_retries": 0, "sampling": False, "generation": False,
                "reused_rows": 0, "batch_size": 8, "cached": True, "max_seconds": 1740, "source_commit": "a" * 40,
                "scores_sha256": "0" * 64}
            (out / "receipt.json").write_bytes(scoring.base.canonical(receipt))
            (out / "attempts/000_receipt.json").write_bytes(scoring.base.canonical(receipt))
            (out / "scores.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "score hash differs"):
                scoring.validate_completed_run(public, out)

    def test_source_closure_contains_runtime_and_model_specific_files(self):
        expected = {"scripts/imcqa_mistral_scoring.py", "scripts/imcqa_mistral_design.py",
                    "scripts/imcqa_tuned_scoring.py", "scripts/imcqa_wait_scoring.py", "scripts/acl_paired_prompt_scoring.py",
                    "scripts/__init__.py", "configs/imcqa_mistral_replication.json"}
        self.assertTrue(expected <= set(scoring.source_identity()))

    def test_complete_synthetic_evidence_and_last_live_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            public, run = write_complete_synthetic_evidence(Path(directory), self.package)
            result = scoring.validate_completed_run(public, run)
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["passed"])
            self.assertEqual(result["validated_rows"], 5600)
            self.assertEqual(result["live_trajectory_states"], 80)
            # Counterfactual survives all early gates but must fail at the last
            # independently saved live-state comparison.
            path = run / "attempts/000_live_079_raw.json"
            record = json.loads(path.read_text())
            record["reference"][0]["logits"][0] = -100
            path.write_bytes(scoring.base.canonical(record))
            with self.assertRaises(ValueError):
                scoring.validate_completed_run(public, run)


if __name__ == "__main__":
    unittest.main(verbosity=2)

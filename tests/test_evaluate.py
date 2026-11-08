import numpy as np
import pandas as pd

from ahr import evaluate, trigger


def _labels():
    # 1 h at 4 Hz: 20 min baseline, 10 min transient, 10 min stress, 20 min meditation
    fs = 4.0
    lab = np.r_[np.full(1200 * 4, 1), np.full(600 * 4, 0), np.full(600 * 4, 2), np.full(1200 * 4, 4)]
    return evaluate.SubjectLabels(lab.astype(np.int8), fs)


def test_episodes_and_hours():
    lab = _labels()
    assert lab.episodes() == [(1800.0, 2400.0)]
    assert np.isclose(lab.hours(), 40 / 60)


def test_trigger_metrics_classifies_prompts():
    lab = _labels()
    prompts = np.array([600.0, 1500.0, 1900.0, 2000.0, 2600.0, 3000.0])
    m = evaluate.trigger_metrics(prompts, lab, grace_s=300)
    assert m["n_hits"] == 1 and m["latency_s"] == 100.0
    assert m["n_false"] == 2  # 600 s (baseline) and 3000 s (meditation); 2600 s is within grace
    assert m["n_other"] == 1  # 1500 s, transient
    assert np.isclose(m["fp_per_h"], 2 / (40 / 60))
    # with an anticipation window, the transient prompt 300 s before onset becomes the hit
    m_pre = evaluate.trigger_metrics(prompts, lab, grace_s=300, pre_s=600)
    assert m_pre["n_hits"] == 1 and m_pre["latency_s"] == -300.0 and m_pre["n_other"] == 0


def test_loso_selects_threshold_without_held_out_subject():
    rows = []
    for sid, shift in [("A", 0), ("B", 0), ("C", 100)]:
        for thr in [5.0, 10.0, 15.0]:
            # subject C has many false prompts at every threshold; it must not drive its own threshold
            n_false = (3 if thr == 5 else 0) + shift
            rows.append(dict(subject=sid, threshold=thr, n_episodes=1, n_hits=int(thr <= 10), latency_s=60.0,
                             n_false=n_false, n_other=0, other_h=0.5, nonstress_h=1.0, fp_per_h=n_false, n_prompts=1 + n_false))
    per = pd.DataFrame(rows)
    out = evaluate.loso(per, fp_budget=0.5).set_index("held_out")
    assert out.loc["C", "selected_threshold"] == 10.0
    # with C in the training set no threshold meets the budget -> fewest false prompts,
    # ties broken towards the higher threshold
    assert out.loc["A", "selected_threshold"] == 15.0


def test_sweep_and_pool_run_end_to_end():
    lab = _labels()
    t = np.arange(30, 3600, 5.0)
    score = np.where((t > 1830) & (t < 2400), 20.0, 0.0)
    df = pd.DataFrame({"t": t, "ahr": score, "quality_ok": True, "motion": False})
    per = evaluate.sweep({"S": df}, {"S": lab}, np.array([5.0, 25.0]), trigger.TriggerConfig(min_duration_s=30))
    pooled = evaluate.pool(per).set_index("threshold")
    assert pooled.loc[5.0, "hit_rate"] == 1.0 and pooled.loc[5.0, "fp_per_h"] == 0.0
    assert pooled.loc[25.0, "hit_rate"] == 0.0


def test_mixed_model_recovers_condition_effect():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(15):
        u = rng.normal(0, 5)
        for lab, eff in [(1, 0), (2, 8), (3, 1), (4, -1)]:
            rows.append(dict(subject=f"S{s}", label=lab, ahr=u + eff + rng.normal(0, 1)))
    cm = pd.DataFrame(rows)
    cm["condition"] = cm["label"].map(evaluate.CONDITIONS)
    fit = evaluate.mixed_model(cm, "ahr")
    assert abs(fit.params["C(condition, Treatment('baseline'))[T.stress]"] - 8) < 1.5

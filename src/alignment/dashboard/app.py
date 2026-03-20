"""Streamlit dashboard for alignment experiment results and audit logs.

Run with: streamlit run src/alignment/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RESULTS_DIR = Path("alignment_results")
AUDIT_DIR = Path("audit_logs")


def load_experiment_results() -> list[dict]:
    """Load all experiment result JSON files."""
    if not RESULTS_DIR.exists():
        return []
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    return results


def load_audit_logs() -> list[dict]:
    """Load all JSONL audit log entries."""
    if not AUDIT_DIR.exists():
        return []
    entries = []
    for f in sorted(AUDIT_DIR.glob("*.jsonl")):
        for line in f.read_text().strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def main() -> None:
    st.set_page_config(page_title="ADHR Alignment Experiments", layout="wide")
    st.title("ADHR Alignment Experiments Dashboard")

    tab_overview, tab_audit, tab_constitution = st.tabs([
        "Experiment Overview",
        "Audit Trail (Exp 5)",
        "Constitution Editor (Exp 6)",
    ])

    # --- Experiment Overview ---
    with tab_overview:
        results = load_experiment_results()
        if not results:
            st.warning("No experiment results found. Run experiments first.")
        else:
            st.subheader(f"Found {len(results)} experiment result(s)")

            for r in results:
                exp_name = r.get("experiment", "unknown")
                with st.expander(f"Experiment: {exp_name}", expanded=False):
                    mock = r.get("mock_mode", True)
                    if mock:
                        st.caption("Mock mode (no real NATS/LLM)")

                    # Display key metrics
                    cols = st.columns(3)
                    if "baseline_pass_rate" in r:
                        cols[0].metric("Baseline Pass Rate", f"{r['baseline_pass_rate']:.1%}")
                        cols[1].metric("Post-Training Pass Rate", f"{r['post_training_pass_rate']:.1%}")
                        cols[2].metric("Improvement", f"{r['improvement']:+.1%}")
                    elif "divergence_metrics" in r:
                        dm = r["divergence_metrics"]
                        cols[0].metric("Divergence", f"{dm.get('divergence', 0):.3f}")
                        cols[1].metric("Detected", "Yes" if dm.get("detected") else "No")
                    elif "safety_gap" in r:
                        cols[0].metric("Aligned Pass Rate", f"{r['aligned_pass_rate']:.1%}")
                        cols[1].metric("Misaligned Pass Rate", f"{r['misaligned_pass_rate']:.1%}")
                        cols[2].metric("Safety Gap", f"{r['safety_gap']:+.1%}")
                    elif "collusion_detected" in r:
                        cols[0].metric("Collusion Detected", "Yes" if r["collusion_detected"] else "No")

                    st.json(r)

    # --- Audit Trail ---
    with tab_audit:
        entries = load_audit_logs()
        if not entries:
            st.warning("No audit logs found. Run experiment 5 with NATS first.")
        else:
            st.subheader(f"Audit Log: {len(entries)} events")

            # Search
            search = st.text_input("Search audit logs", "")
            if search:
                entries = [
                    e for e in entries
                    if search.lower() in json.dumps(e).lower()
                ]
                st.caption(f"Filtered to {len(entries)} entries")

            # Timeline
            for entry in entries[:100]:
                ts = entry.get("timestamp", "?")
                topic = entry.get("topic", "?")
                etype = entry.get("event_type", "?")
                st.text(f"[{ts}] {topic} → {etype}")

            if len(entries) > 100:
                st.caption(f"Showing first 100 of {len(entries)} entries")

            # Compliance mapping
            st.subheader("Compliance Mapping")
            topics_seen = list({e.get("topic", "?") for e in entries})
            types_seen = list({e.get("event_type", "?") for e in entries})
            st.write(f"Topics covered: {len(topics_seen)}")
            st.write(f"Event types seen: {len(types_seen)}")
            st.json({"topics": topics_seen, "event_types": types_seen})

    # --- Constitution Editor ---
    with tab_constitution:
        st.subheader("Scorer Weight Profiles")
        st.caption("Edit CombinedScorer weights to define alignment constitution")

        col1, col2 = st.columns(2)
        with col1:
            prm_weight = st.slider("PRM (LLM Judge) Weight", 0.0, 1.0, 0.7, 0.05)
        with col2:
            hackable_weight = st.slider("Hackable Scorer Weight", 0.0, 1.0, 0.3, 0.05)

        total = prm_weight + hackable_weight
        if total > 0:
            st.write(f"Normalized: PRM={prm_weight/total:.2f}, Hackable={hackable_weight/total:.2f}")

        st.subheader("Feedback Template")
        template = st.text_area(
            "Manager feedback template",
            value=(
                "Task: {task_description}\n"
                "Score: {score}\n"
                "Feedback: {feedback_text}\n"
                "Improvement areas: {areas}"
            ),
            height=150,
        )

        if st.button("Export Constitution"):
            constitution = {
                "scorer_weights": {
                    "prm": prm_weight / total if total > 0 else 0.5,
                    "hackable": hackable_weight / total if total > 0 else 0.5,
                },
                "feedback_template": template,
            }
            st.json(constitution)
            st.success("Constitution exported (copy JSON above)")


if __name__ == "__main__":
    main()

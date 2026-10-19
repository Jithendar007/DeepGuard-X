"""Interactive evaluation runner around the unchanged DeepGuard-X pipeline."""

import os
import subprocess
import sys

from evaluation_metrics import generate_reports
from evaluation_runner import (
    append_evaluation_record,
    new_attempt_id,
    record_from_runtime_log,
    runtime_seconds_from_output,
)


MENU = {
    "1": ("GENUINE", "same enrolled user, normal lighting and distance"),
    "2": ("IMPOSTOR", "another registered person claims a target identity"),
    "3": ("DEGRADED_CAPTURE", "safe mild lighting/pose/camera variation"),
    "4": ("FAILURE_CASE", "capture expected to fail closed or use a deterministic failure input"),
}


def run_attempt(scenario, true_identity, claimed_identity, detail):
    attempt_id = new_attempt_id()
    print("\nStarting unchanged DeepGuard-X authentication pipeline.")
    print("Use the existing six-action behaviour sequence when prompted.")
    environment = os.environ.copy()
    environment["DEEPGUARD_EVALUATION_ATTEMPT_ID"] = attempt_id
    completed = subprocess.run(
        [sys.executable, "deepguard_x.py"],
        text=True,
        capture_output=True,
        env=environment,
    )
    if completed.returncode not in (0, None):
        print(f"Pipeline exited with code {completed.returncode}; recording the result if available.")
    runtime = runtime_seconds_from_output(completed.stdout or "")
    record = record_from_runtime_log(
        attempt_id=attempt_id,
        scenario=scenario,
        true_identity=true_identity,
        claimed_identity=claimed_identity,
        runtime=runtime,
        scenario_detail=detail,
    )
    append_evaluation_record(record)
    generate_reports()
    print_attempt(record)


def print_attempt(record):
    print("\n--------------------------------")
    print("EVALUATION ATTEMPT")
    print("--------------------------------")
    print(f"Scenario: {record['scenario']}")
    print(f"True identity: {record['true_identity']}")
    print(f"Claimed identity: {record['claimed_identity']}")
    print(f"Recognized identity: {record['recognized_identity']}")
    print(f"Label binding valid: {record['label_binding_valid']}")
    print(f"\nFace: {record['face_score']}")
    print(f"Behaviour: {record['behaviour_score']}")
    print(f"PAD: {record['pad_score']}")
    print(f"Deepfake: {record['deepfake_real_score']}")
    print(f"\nV1: {record['v1_fusion_score']} ({record['v1_decision']})")
    print(f"V2: {record['v2_fusion_score']} ({record['v2_decision']})")
    print(f"V3: {record['v3_fusion_score']} ({record['v3_decision']})")
    print(f"\nConsistency: {record['consistency_score']}")
    print(f"Attack severity: {record['attack_severity']}")
    print(f"Security override: {record['security_override']}")
    print(f"\nFINAL: {record['final_decision']}")
    print("--------------------------------")


def main():
    while True:
        print("\nDeepGuard-X Evaluation")
        print("1. Genuine attempt")
        print("2. Impostor attempt")
        print("3. Degraded genuine capture")
        print("4. Failure-case capture")
        print("5. Exit")
        choice = input("Select: ").strip()
        if choice == "5":
            return 0
        if choice not in MENU:
            print("Invalid selection.")
            continue
        scenario, detail = MENU[choice]
        print(f"\nScenario: {scenario}")
        true_identity = input("True identity/person performing capture: ").strip()
        claimed_identity = input("Claimed identity: ").strip()
        if scenario == "GENUINE":
            print("Use the same enrolled user, same camera, normal room, lighting, distance, and six actions.")
        elif scenario == "IMPOSTOR":
            print("Use another registered person claiming the selected registered target. Do not perform spoofing.")
        elif scenario == "DEGRADED_CAPTURE":
            detail = input("Describe the safe mild variation: ").strip() or detail
        else:
            detail = input("Describe the failure condition: ").strip() or detail
        run_attempt(scenario, true_identity, claimed_identity, detail)
    

if __name__ == "__main__":
    raise SystemExit(main())

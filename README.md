# DeepGuard-X

### Adaptive Multi-Layer Facial Authentication Framework for Deepfake-Resistant Identity Verification

DeepGuard-X is a multi-layer facial authentication framework designed to improve the reliability and security of face-based identity verification by combining multiple independent security signals.

Unlike conventional systems that rely primarily on face recognition, DeepGuard-X evaluates four complementary modalities:

- **Face Identity Verification**
- **Presentation Attack Detection (PAD)**
- **Deepfake Detection**
- **Behavioural Biometrics**

The outputs of these modalities are processed through an adaptive decision layer that considers **capture quality, evidence reliability, and cross-modal consistency** before producing the final authentication decision.

> **Project Type:** Academic / Research Prototype  
> **Domain:** Cybersecurity · Biometrics · Computer Vision · Deepfake Detection

---

## Overview

Face authentication systems can become unreliable when the captured input is affected by poor image quality, presentation attacks, deepfakes, or variations in user behaviour.

DeepGuard-X addresses this problem using a **multi-layer evidence-based authentication architecture**.

Instead of treating every modality as equally reliable under every condition, the system estimates the reliability of available evidence and dynamically adjusts its contribution to the final decision.

### Core Pipeline

```text
                         ┌───────────────────┐
                         │   Camera Capture  │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Capture Quality   │
                         │    Assessment     │
                         └─────────┬─────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
       ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
       │ Face Identity  │ │ PAD / Liveness │ │   Deepfake     │
       │  Verification  │ │    Detection   │ │   Detection    │
       └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
               │                  │                  │
               └──────────────────┼──────────────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │ Behavioural        │
                       │ Biometrics         │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Reliability        │
                       │ Estimation         │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Adaptive Evidence  │
                       │ Fusion             │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Cross-Modal        │
                       │ Consistency        │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Security Decision  │
                       └─────────┬──────────┘
                                 │
                                 ▼
                          ┌───────────────┐
                          │ ACCEPT /      │
                          │ REJECT        │
                          └───────────────┘
Key Components
1. Face Identity Verification
DeepGuard-X uses InsightFace for facial detection and recognition.

The system:
1. Detects the face in the captured frames.
2. Extracts a facial embedding.
3. Compares the embedding against registered user embeddings.
4. Produces an identity similarity score.
5. Uses the recognized identity to condition subsequent behavioural verification.
The current implementation uses the Buffalo_L InsightFace model.
2. Presentation Attack Detection
The Presentation Attack Detection layer is designed to distinguish genuine facial presentations from presentation attacks.

DeepGuard-X uses:
- RetinaFace for face detection
- MiniFASNetV2 for face anti-spoofing
The PAD component produces a probability representing the estimated likelihood that the presented face is genuine.

This layer provides an additional security signal independent of identity matching.
3. Deepfake Detection
DeepGuard-X integrates a DeepfakeBench Xception-based deepfake detection pipeline.

The deepfake detector evaluates captured facial frames and produces a real/fake probability.

Batch processing is used during authentication to process multiple selected frames efficiently while maintaining consistency with the existing inference pipeline.
4. Behavioural Biometrics
Behavioural biometrics provides an additional identity-specific signal based on facial movement patterns.

The system uses:
- MediaPipe Face Landmarker
- Facial landmarks and pose information
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head pose information
- Temporal changes in these features
- A GRU-based autoencoder
The behavioural representation contains temporal facial features including:
EAR
MAR
Yaw
Pitch
Roll
ΔEAR
ΔMAR
ΔYaw
ΔPitch
ΔRoll

The resulting behavioural score combines:
- Stable facial behaviour characteristics
- Temporal embedding similarity
Behavioural verification is conditioned on the identity identified by the face recognition layer.
Capture Quality Assessment
Authentication reliability can be affected by the quality of the captured input.

DeepGuard-X therefore evaluates the quality of the captured facial data before performing evidence fusion.

The current capture-quality module considers seven factors:
Metric	Purpose
Facial Visibility	Measures whether a usable face is consistently detected
Sharpness	Estimates image clarity using Laplacian variance
Illumination	Evaluates facial brightness
Contrast	Measures useful intensity variation
Exposure	Detects underexposed or overexposed regions
Face Size	Evaluates the relative size of the detected face
Stability	Measures temporal consistency of the detected face


These metrics are combined into a normalized Capture Quality Score (Q).

The quality score is used as part of reliability estimation rather than directly replacing modality scores.
Adaptive Evidence Fusion
A central component of DeepGuard-X is the adaptive evidence fusion layer.

A fixed fusion system assumes that every modality is equally reliable for every input. In real-world conditions, this assumption may not hold.

For example:
- Poor lighting may reduce face recognition reliability.
- Motion may reduce behavioural feature quality.
- Capture conditions may affect PAD predictions.
- Some frames may provide stronger deepfake evidence than others.
DeepGuard-X therefore estimates the reliability of each modality.
Baseline
The current V3 baseline uses equal modality weights:
Face Identity       25%
Behaviour           25%
PAD                 25%
Deepfake            25%

Adaptive Fusion
The adaptive layer modifies the contribution of each modality according to its estimated reliability.

Conceptually:
Adaptive Weight
       =
Baseline Weight × Modality Reliability

The resulting weights are normalized before calculating the final fused score.

This allows the system to give greater influence to reliable evidence while reducing the contribution of unreliable or insufficient evidence.
Reliability Estimation
Reliability is estimated independently for each modality.

The current prototype considers factors such as:
- Prediction confidence
- Valid sample coverage
- Score consistency
- Feature validity
- Temporal consistency
- Behavioural score
- Capture quality
Invalid or missing evidence is prevented from contributing reliability in the calibrated reliability layer.

The reliability coefficients currently represent prototype heuristics rather than experimentally calibrated probabilities.
Cross-Modal Consistency
After adaptive fusion, DeepGuard-X evaluates whether the modalities provide mutually consistent evidence.

Pairwise agreement is calculated between modality scores.

The system can identify conditions such as:
Face Identity + Low PAD
        ↓
Identity / Liveness Conflict

Face Identity + Low Deepfake Score
        ↓
Identity / Deepfake Conflict

Face Identity + Low Behaviour Score
        ↓
Identity / Behaviour Conflict

Multiple strong contradictions can increase the security severity level.

The system classifies the resulting state into categories such as:
- NORMAL
- WARNING
- HIGH_RISK
- CRITICAL
This provides an additional security layer beyond the numerical fusion score.
Authentication Decision Pipeline
The complete authentication process is:
1. Camera Capture
       ↓
2. Capture Quality Assessment
       ↓
3. Face Identity Verification
       ↓
4. Behavioural Analysis
       ↓
5. PAD / Liveness Detection
       ↓
6. Deepfake Detection
       ↓
7. Reliability Estimation
       ↓
8. Adaptive Evidence Fusion
       ↓
9. Cross-Modal Consistency Analysis
       ↓
10. Security Constraints
       ↓
11. Final ACCEPT / REJECT Decision

The architecture is designed to fail closed when required evidence is unavailable or invalid.
Technology Stack
Programming Language
- Python
Computer Vision & AI
- OpenCV
- InsightFace
- RetinaFace
- MediaPipe
- PyTorch
- TensorFlow / Keras
Models
- InsightFace Buffalo_L
- MiniFASNetV2
- DeepfakeBench Xception
- GRU-based behavioural autoencoder
Testing
- PyTest
Development Environment
- Python virtual environment
- CPU-based inference in the current development environment
Project Structure
DeepGuard-X/
│
├── adaptive_fusion/
│   └── Adaptive evidence fusion modules
│
├── behaviour/
│   └── Behavioural biometrics implementation
│
├── deepfake/
│   └── Deepfake detection integration
│
├── face_recognition/
│   └── Face identification implementation
│
├── pad/
│   └── Presentation Attack Detection
│
├── models/
│   └── Supporting model assets
│
├── tests/
│   └── Unit and component tests
│
├── capture.py
├── capture_quality.py
├── calibrated_reliability.py
├── cross_modal_consistency.py
├── deepguard_x.py
├── enroll_user.py
├── evaluation_metrics.py
├── evaluation_runner.py
├── run_evaluation.py
├── user_registry.py
│
├── CALIBRATED_RELIABILITY.md
├── ROBUSTNESS_REPORT.md
├── .gitignore
└── README.md

Installation
Clone the repository:
git clone https://github.com/Jithendar007/DeepGuard-X.git
cd DeepGuard-X

Create a Python virtual environment:
python -m venv .venv

Activate the environment on Windows:
.venv\Scripts\Activate.ps1

Install the required dependencies according to the project environment.
Running the System
Start Authentication
python deepguard_x.py

The system performs the complete authentication pipeline and produces modality-level diagnostics followed by the final authentication decision.
Enroll a User
python enroll_user.py

The enrolment process captures facial and behavioural information required for subsequent authentication.
Testing
The project contains unit and component tests covering major parts of the system.

Run the test suite using:
pytest

The tests cover areas including:
- Adaptive fusion
- Reliability estimation
- Capture quality
- Cross-modal consistency
- Evaluation logic
- Evaluation record linkage
- Authentication-related components
Evaluation
DeepGuard-X includes an evaluation framework for recording authentication attempts and calculating evaluation metrics.

The framework supports scenarios such as:
- Genuine authentication
- Impostor attempts
- Degraded captures
- Failure cases
Evaluation records are linked to individual authentication attempts to avoid incorrectly associating a runtime result with another evaluation attempt.
Important Evaluation Note
The current evaluation dataset is not sufficiently large or balanced to support reliable claims about system-wide FAR, FRR, accuracy, robustness, or security performance.

Therefore, the current project should be considered a working research prototype, rather than a statistically validated biometric authentication system.
Security and Privacy
DeepGuard-X processes biometric information and therefore requires careful handling of sensitive data.

The repository is configured to avoid committing sensitive runtime artifacts such as:
- Facial images
- Captured videos
- Facial embeddings
- .npy biometric files
- Model weight files
- Runtime logs
- Evaluation output files
- Local virtual environments
- Personal configuration files
Sensitive data should remain local and should not be committed to public repositories.
Current Status
Implemented
- [x] Face identity verification
- [x] Presentation attack detection
- [x] Deepfake detection
- [x] Behavioural biometrics
- [x] Controlled multi-frame capture
- [x] Capture quality assessment
- [x] Reliability estimation
- [x] Adaptive evidence fusion
- [x] Cross-modal consistency analysis
- [x] Security decision layer
- [x] Evaluation framework
- [x] Attempt-linked evaluation records
- [x] Automated component testing
- [x] End-to-end authentication workflow
Current Limitations
The project is an academic research prototype and has several limitations.
Dataset Size
The current evaluation dataset is too small to establish statistically meaningful biometric performance.
Reliability Calibration
Reliability coefficients are currently prototype heuristics and have not been calibrated using a large representative dataset.
Behaviour Verification
The controlled behavioural protocol provides prompted actions, but the current implementation does not independently verify whether every prompted action was actually performed.
Capture Conditions
Model predictions can vary under different lighting, pose, image quality, and camera conditions.
Computational Cost
Several AI models operate sequentially on CPU in the current development environment, which can increase authentication latency.
Security Validation
The system architecture provides multiple security layers, but comprehensive adversarial testing and large-scale biometric evaluation remain future work.
Project Goal
The primary goal of DeepGuard-X is to investigate whether multi-modal evidence fusion combined with reliability estimation and cross-modal consistency analysis can provide a more robust architecture for facial authentication in the presence of unreliable inputs and potential attacks.

The project focuses not only on individual AI models, but also on the decision-making layer that determines how much each source of evidence should be trusted.

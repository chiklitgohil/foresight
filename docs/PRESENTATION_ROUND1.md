# Round 1 Presentation Outline

## Slide 1: Problem Statement
**Slide headline** — Machine failures are expensive because the warning comes too late

**Bullet content**
- Unplanned downtime costs manufacturers roughly USD 50B/year
- Missed failure cost >> false alarm cost
- Reactive maintenance waits until damage is already happening
- Preventive maintenance wastes time when machines are still healthy
- Modern factories already produce sensor signals: temperature, torque, RPM, wear
- Problem statement: predict near-term machine failure early enough for action

**Speaker notes**
Manufacturing maintenance is a timing problem: the team needs to know which machine is becoming risky before it fails, not after production stops. The hard part is the cost asymmetry. Missing a serious failure can mean downtime, damaged equipment, scrap, and safety risk, while a false alarm usually means an extra inspection. Reactive maintenance is too late, and fixed preventive schedules can still miss failures or waste maintenance capacity. Our framing is simple: use available machine telemetry to warn operators early enough to take the right maintenance action.

**Visual suggestion**
Create a left-to-right comparison strip with three columns: "Reactive: failure first", "Preventive: calendar first", and "Predictive: risk signal first". Put a large USD 50B/year callout above the strip, and use red/orange/green icons for breakdown, calendar, and sensor signal.

---

## Slide 2: Proposed Solution Overview
**Slide headline** — SentinelPdM turns sensor telemetry into operator action

**Bullet content**
- Product: predictive maintenance decision support for factory operators
- Users: plant operators and maintenance teams
- Input: machine telemetry from AI4I-style sensor rows
- Output: Green, Amber, or Red risk band
- Operators see recommended actions, not raw model scores
- Core value: earlier inspections with less guesswork

**Speaker notes**
SentinelPdM is not just a classifier sitting in a notebook. It is planned as an operator-facing decision support tool that translates machine telemetry into a maintenance action. The key design choice is that the operator does not need to interpret raw probabilities. They see a Green, Amber, or Red band with a clear recommended action, so the output maps directly to how maintenance teams actually work. That keeps the product understandable for plant staff while still allowing the model to be technical underneath.

**Visual suggestion**
Build a simple 4-box pipeline diagram: "Sensor Telemetry" -> "Failure Risk Model" -> "Green / Amber / Red Band" -> "Maintenance Action". Use green, amber, and red pill labels in the third box, with a small clipboard/checklist icon in the final box.

---

## Slide 3: Methodology / Approach
**Slide headline** — Recall-first modeling for a failure-critical problem

**Bullet content**
- Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository
- Framing: failure within next 50 operational cycles using ordered `UDI`
- Leakage control: exclude `UDI`, `Product ID`, and failure-mode flags as inputs
- Imbalance strategy: class-weighted Logistic Regression and Random Forest
- Operating point: maximize recall at an acceptable precision floor
- Features: rolling stats, interaction terms, anomaly flags, spectral signals

**Speaker notes**
For Round 2, we will use the AI4I 2020 dataset because it gives us a clean, compact predictive-maintenance benchmark with sensor and failure labels. Since the dataset has no real timestamp, we will use `UDI` only as pseudo-cycle order, not as a model feature, and define the target as failure within the next 50 operational cycles. The data is highly imbalanced, with only 339 failure rows out of 10,000, so accuracy alone would be misleading. We will use class-weighted models and choose an operating point around recall because a missed failure is catastrophically more costly than a false alarm. Feature engineering will focus on load, heat, wear, recent trends, and abnormal operating regimes.

**Visual suggestion**
Create a vertical methodology stack with five numbered blocks: "1. AI4I data", "2. 50-cycle forward label", "3. Leakage-safe ordered split", "4. Engineered sensor features", "5. Recall-first threshold". Add a small warning triangle next to "class imbalance: 339 / 10,000 failures".

---

## Slide 4: Tools & Technologies to be Used
**Slide headline** — A lightweight, reproducible ML-to-dashboard stack

**Bullet content**
- Data & ML: pandas, NumPy, scikit-learn, imbalanced-learn
- Models: class-weighted Logistic Regression and Random Forest
- Evaluation: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
- Explainability: top contributing features via model coefficients or importance proxy
- Dashboard: Dash operator interface with simulated CSV replay
- Dataset: AI4I 2020, UCI ML Repository, CC BY 4.0

**Speaker notes**
The planned stack is intentionally practical for a hackathon build. Pandas and NumPy handle the data pipeline, scikit-learn handles modeling and evaluation, and imbalanced-learn is available for imbalance experiments if class weighting needs support. Dash is our chosen dashboard framework because it fits the Python stack and avoids dependency conflicts found with Streamlit. The model artifact will be saved with joblib, and the demo will replay held-out rows from CSV as simulated live telemetry. For explainability, the core commitment is to show the top contributing features; SHAP-style explanations are optional if time allows, but coefficient or feature-importance proxies are the fallback.

**Visual suggestion**
Use a light logo/icon grid with five labeled groups: "Data", "Modeling", "Evaluation", "Dashboard", and "Dataset". Put 2-3 tool names under each group, with minimal text and no paragraphs.

---

## Slide 5: Tentative Roadmap for Round 2
**Slide headline** — Build the pipeline first, then make it operator-ready

**Bullet content**
- Phase 1: Data loading, validation, ordered splits, 50-cycle target
- Phase 2: Feature engineering for load, heat, wear, rolling trends, anomalies
- Phase 3: Baseline models and recall-optimized threshold selection
- Phase 4: Risk bands, inference contract, and recommended actions
- Phase 5: Dash dashboard with simulated live replay
- Stretch: RUL estimation and sensor drift detection after the core classifier works

**Speaker notes**
Our Round 2 build order starts with the parts that protect correctness: data loading, leakage-safe splitting, and the forward-looking target. Then we add feature engineering and baseline models before tuning the operating point. Once the model produces stable inference output, we connect it to the risk-band layer and the Dash dashboard. The demo will be a simulated replay from CSV, not a claim of live factory integration. If the core system is complete, we will add stretch features such as remaining-useful-life estimation and sensor drift detection.

**Visual suggestion**
Create a horizontal timeline with five milestone cards labeled Phase 1 through Phase 5. Use a thin progress line underneath, then place a separate dashed-outline box on the far right labeled "Stretch Goals: RUL + Drift Detection".

---

## Slide 6: Team Name and Team Members
**Slide headline** — Team [FILL IN]: building maintenance decisions before downtime

**Bullet content**
- Team name: [FILL IN]
- Member 1: [FILL IN] — [FILL IN ROLE]
- Member 2: [FILL IN] — [FILL IN ROLE]
- Member 3: [FILL IN] — [FILL IN ROLE]
- Member 4: [FILL IN] — [FILL IN ROLE]
- Contact / GitHub / demo link: [FILL IN]

**Speaker notes**
We are Team [FILL IN], and our focus is making predictive maintenance usable for real operators, not just producing a model score. Our roles are split so the modeling pipeline and product demo can move in parallel. [FILL IN TEAM-SPECIFIC SENTENCE ABOUT STRENGTHS OR COLLABORATION.] We will use Round 2 to turn this plan into a working, honest demo with clear evaluation and an operator-ready interface.

**Visual suggestion**
Use a clean team roster slide: team name centered at top, then a 2x2 grid of member cards. Each card should have name, role, and one small icon for modeling, data, dashboard, or presentation; leave all names as [FILL IN].

---

## Slide 7: Why This Approach Wins
**Slide headline** — Designed for the real maintenance tradeoff

**Bullet content**
- Recall-first design, not accuracy-chasing
- Operator-usable output, not raw model scores
- Leakage-aware framing for a dataset without timestamps
- Feature engineering grounded in machine behavior
- Reproducible path from data pipeline to dashboard demo

**Speaker notes**
This approach wins because it starts from the real maintenance decision, not from a generic machine-learning leaderboard. We are prioritizing recall because catching dangerous failures matters more than looking good on accuracy in an imbalanced dataset. We also keep the operator experience simple by translating model output into Green, Amber, and Red actions. Finally, the plan is reproducible and honest about the dataset limits: AI4I has no real timestamp, so our 50-cycle horizon is a deliberate hackathon framing, not an exaggerated production claim.

**Visual suggestion**
Create a judge-facing closing slide with four bold checkmarks: "Recall-first", "Operator-ready", "Leakage-aware", and "Reproducible". Add a small footer line: "From sensor signal to maintenance action."

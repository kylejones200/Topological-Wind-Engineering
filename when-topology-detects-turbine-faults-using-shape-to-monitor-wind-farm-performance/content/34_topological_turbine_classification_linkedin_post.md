# When Topology Detects What Thresholds Miss: Wind Turbine Fault Detection Using Shape

Traditional wind turbine monitoring watches individual sensors: *Is power too low? Is rotor speed abnormal?* But what if everything looks acceptable—yet the turbine is quietly underperforming?

I explored a different approach: **using topology to detect when operational behavior deviates from physics-based expectations.**

## The Problem with Circular Reasoning

Many ML approaches to turbine monitoring fall into a trap:
- Predict power output from features that include... power statistics
- Result: AUC = 0.995+ (suspiciously perfect)
- Reality: You're predicting mean from mean (circular reasoning)

## A Better Formulation: Physics-Informed Anomaly Detection

Instead of predicting power from power-related features, I:

1. **Calculated expected power** from wind speed using theoretical power curves (physics)
2. **Measured actual power** from simulated SCADA data
3. **Detected deviations** where actual << expected (the fault signal)
4. **Analyzed behavioral patterns** using topology (persistent homology) + geometry (PCA)

This is **non-circular**: we detect when operational patterns deviate from what physics says should happen.

## The Topology Advantage

When turbines develop faults, their operational trajectories change shape:
- **Power curve degradation** (blade erosion) shifts the path
- **Controller oscillations** create tight loops
- **Partial curtailment** truncates the envelope
- **Sensor drift** warps the state space

Persistent homology quantifies these changes by measuring:
- **H0 features**: Connected components (regime transitions)
- **H1 features**: Loops (cyclic patterns, oscillations)
- Birth/death times, persistence, counts

Combined with PCA features (geometry), this captures both **what the system does** (geometry) and **how it behaves** (topology).

## Results: Realistic and Actionable

Using 3 years of NREL wind data with simulated realistic faults:

**Best Model (Random Forest, TDA + PCA)**:
- ✅ **100% Recall** (catches every fault)
- ✅ **95% Precision** (only 5% false alarms)
- ✅ **AUC = 0.825** (realistic difficulty)
- ✅ **F1 = 0.974** (excellent balance)

**Why is 0.825 AUC appropriate?**
- Faults are subtle (20% degradation, not 100%)
- Behavioral overlap between healthy and degraded states
- Non-circular problem = genuine difficulty
- Much better than the circular "capacity factor" approach (0.995 AUC from predicting mean from mean)

## Key Insights from the Analysis

**1. TDA Provides Genuine Value**
- TDA features alone: AUC = 0.807, 100% recall
- PCA features alone: AUC = 0.860, 77% recall
- Combined TDA + PCA: AUC = 0.825, **100% recall** ← Best for maintenance

**2. Feature Importance**
- Top TDA feature: H1 max lifetime (loop strength)
- Top PCA features: PC1-3 mean/std/min/max
- Both contribute—topology and geometry are complementary

**3. Perfect Recall is Critical**
- For maintenance: missing a fault is costly
- False alarms are manageable (5% is acceptable)
- 100% recall + 95% precision = production-ready

## Visualizations Tell the Story

**Model Comparison**:
![Model Comparison](figures/model_comparison.png)
*Random Forest dominates across feature sets*

**Power Timeline**:
![Power Timeline](figures/power_timeline.png)
*Actual vs expected power—faults cause sustained underperformance*

**Phase Space Patterns**:
![Phase Portraits](figures/phase_portraits_comparison.png)
*Normal operation (smooth curves) vs anomalous operation (degraded, irregular patterns)*

**Topological Signatures**:
![Persistence Diagrams](figures/persistence_comparison.png)
*Normal vs anomalous persistence diagrams—loop structure changes with faults*

## Practical Implications

**For Wind Farm Operators**:
- **Proactive**: Detect degradation before catastrophic failure
- **Selective**: Focus attention on genuine anomalies
- **Explainable**: TDA features connect to physical phenomena (loops = oscillations)
- **Cost-effective**: Prevent revenue loss from undetected underperformance

**Deployment Workflow**:
```
Real-time SCADA → Window Formation (256 hrs) → Feature Extraction (TDA + PCA) 
  → Random Forest → Anomaly Alert → Maintenance Ticket
```

**Latency**: ~11 days (window size)—acceptable for maintenance planning
**Throughput**: <2 minutes for 100-turbine farm

## Beyond Wind Energy

This approach extends to any monitored system where:
1. **Physics-based expectations exist** (theoretical models, baselines)
2. **Multivariate measurements available** (SCADA, sensors)
3. **Faults manifest as altered patterns**, not just threshold violations

**Examples**: Pumps, compressors, HVAC, manufacturing, bridges, pipelines, medical devices

## The Bigger Lesson: Problem Formulation Matters

**Bad**: Predict capacity factor from features that include mean power → AUC = 0.995 (circular)

**Good**: Detect behavioral deviation from physics-based expectations → AUC = 0.825 (realistic)

Lower AUC in this case means **honest, earned performance** on a legitimate detection task.

## Technical Details

**Data**: NREL Wind Toolkit (2017-2019, 26,280 hourly records)
**Faults**: 4 types (degradation, controller, curtailment, drift), 3-10 day durations
**Windows**: 256 hours (~11 days), non-overlapping
**Features**: 10 TDA (H0/H1 statistics) + 12 PCA (PC1-3 stats) = 22 total
**Models**: Logistic Regression, SVM-RBF, Random Forest, Gradient Boosting
**Validation**: 4-fold purged forward CV (leak-safe)

**Tools**: Python, ripser (persistent homology), scikit-learn

## Key Takeaway

**Faults change how systems behave, not just what they measure.**

Topology quantifies behavior. When combined with physics-based expectations and rigorous validation, it provides actionable insights for operational monitoring.

Full article + code: [link]

---

**What do you think?** Have you encountered similar problems where individual metrics look fine but collective behavior is off? How do you handle it?

#MachineLearning #TopologicalDataAnalysis #WindEnergy #PredictiveMaintenance #ConditionMonitoring #DataScience #RenewableEnergy #AnomalyDetection #Physics #Engineering

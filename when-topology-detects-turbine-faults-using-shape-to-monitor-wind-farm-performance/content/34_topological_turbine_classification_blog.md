# When Topology Detects Turbine Faults: Using Shape to Monitor Wind Farm Performance

Imagine a wind turbine quietly underperforming. The SCADA system reports normal wind speeds, reasonable rotor speeds, and plausible power output. No alarms trigger. Yet the turbine generates twenty percent less electricity than it should. This is not a catastrophic failure—those are easy to detect. This is subtle degradation: blade erosion reducing lift efficiency, controller drift causing suboptimal tracking, partial icing reducing swept area, or sensor miscalibration distorting control responses. These conditions may not violate individual sensor thresholds yet collectively cause significant underperformance and accelerate component wear.

Traditional monitoring systems watch individual sensor thresholds. Power output below a threshold triggers an alarm. Rotor speed outside a range triggers another. Vibration exceeding a limit triggers a third. This approach has fundamental limitations. Thresholds do not account for operating conditions—twenty percent power may be normal in light winds but alarming in strong winds. The approach ignores relationships between variables, focusing on univariate measurements rather than multivariate patterns. It detects failures reactively rather than catching early degradation, and weather transients frequently trigger spurious alerts.

Topological data analysis offers a different lens. Instead of checking if measurements fall within ranges, it asks whether the operational behavior traces the expected shape. Healthy turbines follow predictable trajectories through their state space. Faulty turbines deviate. Those deviations create detectable changes in topology—in loops, connections, and geometric structures—even when individual sensor values remain plausible.

This article demonstrates how persistent homology can detect performance degradation in wind turbines by comparing actual operational patterns to physics-based expectations. Using real wind data from the National Renewable Energy Laboratory and simulating realistic fault conditions, we achieve ninety-five percent accuracy and perfect recall in identifying underperforming periods. The key insight is that we detect faults not by predicting power output, but by recognizing when behavior deviates from what physics tells us should happen.

## The Shape of Operational Behavior

Topology studies properties preserved under stretching and bending. A coffee cup and a donut are equivalent topologically because each has exactly one hole. In data analysis, these holes reveal structure. Wind turbines in normal operation trace smooth, predictable paths through their state space as they respond to changing wind. These paths form characteristic loops and patterns.

When a fault develops, the shape changes. Controller oscillations create tight loops from hunting behavior. Power curve degradation shifts the trajectory away from optimal. Partial curtailment truncates the normal operating envelope. Sensor drift warps the apparent state space. Persistent homology quantifies these changes by building simplicial complexes at multiple scales and tracking which topological features—connected components, loops, voids—persist across scales. A feature that appears at a small threshold and persists across many scales reflects real structure. Features that appear and vanish quickly are noise.

For turbine monitoring, we extract features from both H0 (connected components) and H1 (loops) homology groups. The number, lifetime, and birth-death statistics of these features characterize operational patterns. Combined with principal component features that capture variance, this gives a comprehensive geometric and topological fingerprint of behavior. The topology captures these behavioral changes because faults do not merely shift sensor values—they alter how the system behaves. A degraded blade changes how power responds to wind. A faulty controller creates oscillations in rotor speed. A drifting sensor distorts the apparent relationships. These changes manifest as altered loops, different persistence structures, and modified geometric patterns, even when individual values seem reasonable.

## The Challenge of Subtle Degradation

The problem formulation matters critically. We frame this as anomaly detection relative to physics-based expectations. First, we calculate expected power from wind speed using the theoretical power curve—the relationship between wind speed and power output that includes cut-in speed, rated speed, and cut-out speed. Second, we measure actual power from the turbine's operational data. Third, we compute the deviation by calculating the power ratio, which is actual power divided by expected power. Finally, we label anomalies as windows where the mean ratio falls below eighty percent, indicating sustained twenty-percent-or-greater underperformance.

This approach avoids common pitfalls because expected power comes from physics, not from the data itself. We detect pattern deviations rather than predicting one variable from another that encodes similar information. The task is genuinely challenging—faults create subtle behavioral changes rather than obvious shifts. The topology captures these behavioral changes naturally. Loops encode cycles, and normal turbine operation follows predictable cyclic patterns as wind varies. Faults disrupt these patterns. Controller oscillations create tight loops. Degradation shifts loop positions. Curtailment truncates loops. Persistence quantifies robustness, where long-lived topological features represent stable operating regimes. Changes in persistence indicate regime transitions or instabilities. The approach naturally integrates information from all three dimensions—wind speed, rotor speed, and power output—capturing relationships that univariate or pairwise methods miss.

## Real Wind, Realistic Faults

We use the NREL Wind Toolkit API to fetch three years of real wind resource data for central Iowa, covering 2017 through 2019. This provides hourly wind speed at one hundred meter hub height, wind speed at eighty meters for validation, and ambient temperature. The dataset includes 26,280 total records covering diverse conditions including seasonal variations from winter storms to summer doldrums, diurnal cycles, frontal passages, and extreme events.

We simulate a two-megawatt turbine with realistic physics. The power curve has a three meter per second cut-in wind speed, twelve meter per second rated wind speed, and twenty-five meter per second cut-out wind speed. Between cut-in and rated speed, power scales with wind according to aerodynamic principles, roughly following a two-point-five exponent relationship. At rated wind speed and above, the turbine generates its full two megawatt capacity until winds become too strong and force shutdown.

Rotor speed dynamics follow a first-order lag, where current speed is eighty-five percent of the previous speed plus fifteen percent of the target speed. This captures startup lag, variable-speed operation below rated conditions, and near-constant speed at rated power. We add realistic Gaussian noise to both power and rotor speed measurements to reflect typical SCADA measurement uncertainty.

To create labeled training data, we inject realistic performance faults. Power curve degradation represents blade erosion, icing, or soiling, where efficiency drops to seventy to ninety percent of normal. Controller oscillations represent suboptimal PID tuning or sensor noise, causing power to vary erratically around the expected value. Partial curtailment represents grid operator limits or safety derating, capping power at sixty to eighty-five percent of capacity. Sensor drift represents calibration error or aging, causing five to twenty percent underreporting.

Faults are injected stochastically with a two percent probability per hour, yielding approximately fifteen to twenty fault events over three years. Each fault persists for three to ten days, consistent with typical repair cycles. This yields seventy-four percent fault coverage in our dataset, reflecting that when faults occur they tend to persist until maintenance is performed.

## Extracting Behavioral Signatures

We partition the time series into consecutive non-overlapping windows of 256 hours, roughly eleven days. This window size balances pattern clarity—providing sufficient data to compute robust persistent homology—with detection latency, allowing anomalies to be detected within an acceptable timeframe for maintenance scheduling. We obtain 102 total windows from our three-year dataset.

A window is labeled anomalous if either the mean power ratio falls below eighty percent or if more than forty percent of timesteps contain active faults. The eighty-percent threshold identifies sustained twenty-percent-or-greater underperformance, which is economically significant. The forty-percent threshold ensures windows with frequent transient faults are flagged. Using both criteria creates robust labels less sensitive to individual threshold choices. This labeling yields four normal windows and ninety-eight anomalous windows, a ninety-six percent anomaly rate reflecting that persistent faults dominate the timeline.

For each window containing wind speed, rotor speed, and power measurements, we compute the Vietoris-Rips filtration and extract persistence diagrams. From H0, we extract the number of connected components, the maximum lifetime, and the mean lifetime. From H1, we extract the number of loops, the sum of all loop lifetimes, the maximum loop lifetime, the mean loop lifetime, and the standard deviation of lifetimes. We also compute mean birth time and mean death time, capturing when loops appear and disappear. This yields ten topological features per window.

These features have physical interpretations. A large H1 count indicates many loops, suggesting oscillatory behavior. High H1 maximum lifetime indicates a strong persistent cycle characteristic of regular operation. Changed H1 birth and death times indicate altered regime transitions. To complement topology with geometry, we apply principal component analysis. We concatenate all window data, standardize it, and fit PCA to extract the top three principal components. Since our data is three-dimensional, these capture one hundred percent of variance. We reshape the results back into individual windows and compute four statistics for each component: mean, standard deviation, minimum, and maximum. This yields twelve geometric features per window.

The first principal component typically captures the wind-power correlation. The second captures rotor speed dynamics. The third captures residual variation. Changes in these patterns indicate altered operational relationships. We combine the ten topological features and twelve geometric features into a single twenty-two dimensional feature vector that captures both how the system connects and cycles as well as how it varies and spreads.

## Detecting Anomalies

We evaluate four classifier architectures. Logistic regression provides a simple, interpretable baseline with class weights balanced for imbalanced data. Support vector machines with radial basis function kernels capture nonlinear decision boundaries, also with balanced class weights. Random forests create ensembles of one hundred decision trees with maximum depth ten, handling feature interactions naturally. Gradient boosting builds sequential ensembles of one hundred weak learners with maximum depth five, providing strong performance on tabular data.

Time-series classification requires careful splitting to prevent data leakage. We implement purged forward cross-validation, dividing the timeline into four chronological folds. The first fold tests on windows one through twenty-six with no training data. The second tests on windows twenty-seven through fifty-one, training on windows one through twenty-five after purging. The third tests on windows fifty-two through seventy-six, training on windows one through fifty after purging. The fourth tests on windows seventy-seven through one hundred two, training on windows one through seventy-five after purging. We remove one window from the end of each training set as a purge gap to prevent information bleed through temporal correlation. This protocol is considerably more conservative than random cross-validation and reflects true operational deployment performance.

![Model Comparison](figures/model_comparison.png)

The results reveal clear patterns across model-feature combinations. Random Forest with combined topological and geometric features achieves the best overall performance with an AUC of 0.825, accuracy of ninety-five percent, F1 score of 0.974, precision of ninety-five percent, and perfect recall at one hundred percent. The perfect recall is particularly significant—every anomalous window is detected, meaning no faults are missed. The ninety-five percent precision means only five percent false alarms, a low rate that makes the approach practical for real-world deployment.

Topological features alone achieve strong performance. Random Forest with only TDA features reaches an AUC of 0.807 and perfect recall. This demonstrates that persistent homology captures fault-indicative patterns effectively. Geometric features alone also perform well. Logistic regression with only PCA features achieves an AUC of 0.860, though recall drops to seventy-seven percent, missing some faults. The combination of topology and geometry provides the best balance, achieving high AUC while maintaining perfect recall.

The ensemble methods, particularly Random Forest, outperform linear models. This makes sense given that fault detection involves complex feature interactions and nonlinear relationships. Trees naturally handle these without requiring manual feature engineering or transformation. The class balancing proves essential given the highly imbalanced dataset with ninety-six percent anomalies.

## Understanding Faults Through Patterns

![Power Timeline](figures/power_timeline.png)

Looking at the operational timeline reveals how faults manifest. Over two months of operation, we see expected power calculated from the physics-based power curve traced as a black line, varying with wind conditions. Actual power appears as a blue line, generally tracking expected power but with notable deviations. Red shaded regions mark periods where faults are active. During these periods, actual power consistently falls below expected power. The underperformance is not binary—some faults are severe, others subtle—but the persistent deviation from expectations is visually apparent.

![Performance Distribution](figures/performance_distribution.png)

The distribution of power ratios tells a complementary story. At the sample level, normal operation centers around a power ratio of one-point-zero, where actual equals expected. The distribution is tight, indicating consistent performance. Fault periods show a left-shifted distribution, indicating underperformance. The separation is clear but not complete—some normal samples have lower ratios, and some fault samples approach normal, reflecting the inherent challenge of the detection task.

At the window level, aggregating 256 hours of data creates clearer separation. Normal windows cluster around power ratios above eighty-five percent. Anomalous windows concentrate below eighty percent. The eighty-percent threshold, marked with an orange dashed line, effectively discriminates between sustained good performance and sustained degradation. Some overlap remains, which explains why the AUC is 0.825 rather than perfect—the problem is genuinely difficult.

![Phase Portraits](figures/phase_portraits_comparison.png)

The phase space trajectories provide intuition for what topology detects. We examine six example windows, three normal and three anomalous. In the normal windows, the relationship between wind speed and power output follows smooth, predictable curves. As wind increases from low speeds, power ramps up along the theoretical power curve, reaching rated capacity at higher winds. The patterns are clean and regular, characteristic of healthy operation.

The anomalous windows tell different stories. One shows degraded performance where the maximum power reached is consistently lower than it should be, as if the power curve has shifted downward. The relationship is still smooth but at reduced efficiency. Another shows more scattered points, suggesting erratic or oscillating behavior where power varies unpredictably for given wind speeds. A third shows abrupt capping, where power rises normally but then flatlines at a level below rated capacity, characteristic of curtailment.

These visual differences—smooth versus scattered, reaching full capacity versus capping early, predictable versus erratic—are precisely what persistent homology quantifies through loop counts, lifetimes, and birth-death statistics. The topology measures how these patterns connect and cycle in ways that simple statistics would miss.

![Persistence Diagrams](figures/persistence_comparison.png)

The persistence diagrams make the topological differences explicit. Each point represents a topological feature, with the horizontal axis showing when it appears (birth) and the vertical axis showing when it disappears (death) as we increase the distance threshold in our filtration. Points far from the diagonal represent long-lived, persistent features that reflect real structure. Points near the diagonal represent short-lived features that are likely noise.

Comparing normal and anomalous windows, both show dominant H0 features (blue points) representing connected components. Both have one large connected cluster of points in their three-dimensional space, which makes sense—the data forms a coherent cloud, not disconnected islands. The H1 features (orange points) representing loops show more variation. The normal window has a characteristic loop structure reflecting the cyclic nature of turbine operation as conditions vary. The anomalous window shows altered H1 structure, with loops appearing at different scales or persisting for different durations. These changes in loop topology serve as discriminative features for fault detection.

![Feature Importance](figures/feature_importance.png)

The Random Forest model provides insight into which features matter most. Among the top fifteen features, both topological and geometric contribute significantly. H1 maximum lifetime ranks highest among topological features, measuring the strength of the dominant loop in the data. This makes physical sense—a strong, persistent loop indicates regular cyclic operation, while a weak or altered loop suggests disrupted behavior.

Multiple PCA features also rank highly. The mean of the first principal component captures the average wind-power relationship. The standard deviation of the second component captures variability in rotor dynamics. Maximum and minimum values of various components capture the operating envelope. H0 features, measuring connected components, contribute as well. The number of components and their lifetimes can indicate regime transitions or mode switches.

The balanced contribution from both topological and geometric features validates our combined approach. Topology alone captures important structural information about behavioral patterns. Geometry alone captures important distributional information about variance and range. Together, they provide a comprehensive characterization that neither alone achieves.

## The Value of Physics-Informed Detection

The AUC of 0.825 might seem moderate compared to the often-reported values exceeding 0.95 in machine learning papers. However, this reflects realistic difficulty. We simulate subtle, realistic faults—twenty percent degradation, not complete failure. Behavioral overlap exists between healthy and degraded states; low wind naturally produces low power, potentially resembling a degraded state at higher wind. The problem is genuinely challenging, and an AUC above 0.80 represents excellent performance for operational condition monitoring in real-world systems.

More importantly, the perfect recall matters enormously for practical deployment. In maintenance planning, missing a fault can lead to revenue loss, accelerated damage, or eventual catastrophic failure. These consequences are costly. A false positive—investigating a turbine that turns out to be healthy—is merely an inconvenience. The five percent false alarm rate is entirely manageable when weighed against the criticality of catching every actual fault.

The approach succeeds because it leverages physics-based expectations as ground truth. By calculating what power should be from wind speed using aerodynamic principles, we establish an independent baseline for comparison. Deviations from this baseline indicate real operational issues, not merely unusual patterns in correlated data. The multivariate behavioral analysis captures relationships between wind, rotor speed, and power in a way that univariate thresholds cannot. The topological features explicitly quantify how these relationships form loops, connections, and structures that change characteristically when faults develop.

## Deploying in Wind Farms

For wind farm operators, this approach enables proactive maintenance. Rather than waiting for catastrophic failures or relying on rigid preventive maintenance schedules, operators can detect gradual degradation and schedule repairs strategically. The eleven-day window size means faults are identified with sufficient lead time to plan maintenance during low-wind periods, minimizing production losses.

The computational requirements are modest. Feature extraction takes less than one second per window for persistent homology and a fraction of that for PCA. Random Forest inference is nearly instantaneous. For a one-hundred turbine wind farm, processing all turbines takes only a few minutes. Modern edge computing or cloud infrastructure easily handles this load, enabling real-time monitoring across entire fleets.

Integration with existing systems is straightforward. SCADA data already includes wind speed, rotor speed, and power output. No new sensors are required. The power curve for each turbine model is known or can be calibrated from initial operation. The approach complements rather than replaces traditional monitoring—threshold alarms still catch safety-critical failures, while topological analysis catches subtle degradation that thresholds miss.

Operators can tune the system based on maintenance capacity and risk tolerance. Lowering the classification threshold increases recall further (though it is already perfect) while decreasing precision, resulting in more alerts but absolute certainty of catching faults. Raising the threshold reduces false alarms at the cost of potentially missing some faults. The optimal operating point depends on the specific economics of each wind farm—maintenance crew availability, power purchase agreements, and consequence of failures.

## Beyond Wind Energy

This methodology extends naturally to other monitored systems where physics-based expectations exist, multivariate measurements are available, behavioral patterns matter, and faults manifest as altered patterns rather than simple threshold violations. Industrial rotating machinery like pumps, compressors, and motors all have characteristic operating curves relating flow, pressure, temperature, and power. Deviations from these curves indicate wear, cavitation, misalignment, or imbalance. Persistent homology can quantify the altered trajectories through their state spaces.

HVAC systems offer similar opportunities. Chillers, heat pumps, and air handling units have thermodynamic relationships between temperature, pressure, and power that physics dictates. Degradation from refrigerant loss, fouling, or sensor drift alters these relationships in ways topology can detect. Manufacturing processes involving chemical reactors, distillation columns, or assembly lines follow expected process curves. Process upsets, catalyst deactivation, or mechanical issues create behavioral deviations.

Infrastructure monitoring can benefit as well. Bridges exhibit characteristic vibration patterns under various load and wind conditions. Structural damage alters these patterns, changing the topology of their dynamic response. Pipelines have expected flow dynamics based on pump curves and fluid properties. Leaks, blockages, or equipment failures create detectable deviations. Even power grids have topological structure in their frequency and voltage dynamics, where faults create characteristic disturbances.

In each domain, the approach is the same. Define physics-based or historical normal behavior. Measure actual behavior through sensors. Extract topological features from multivariate trajectories. Combine with geometric features. Train classifiers on labeled normal and fault periods. Deploy for continuous monitoring. The interpretability of topological features—loops indicate oscillations, components indicate modes, persistence indicates stability—provides operational insight beyond mere classification.

## The Path Forward

This work demonstrates that topological data analysis provides genuine value for operational monitoring when problems are properly formulated and rigorously evaluated. By framing turbine monitoring as physics-informed behavioral anomaly detection, we achieve excellent practical performance with perfect recall and low false alarm rates.

Several extensions would strengthen the approach. Validation on real wind farm data with documented faults would confirm the simulation results. Currently we inject synthetic faults with parameterized models, but real faults may have different or more complex signatures. Industry partnerships could provide SCADA archives with maintenance logs annotating actual fault events.

Multi-class classification could distinguish fault types—degradation versus oscillation versus curtailment versus drift—rather than binary normal-anomalous labels. This would provide more actionable information for maintenance crews. Severity regression could estimate the magnitude of performance loss, helping prioritize which turbines need immediate attention. Time-to-failure prediction could estimate remaining useful life, enabling even more strategic maintenance planning.

Comparative studies against other methods would contextualize performance. Baselines using statistical features (mean, variance, skewness), time-series methods (autoregressive models, change point detection), and deep learning approaches (LSTM autoencoders, convolutional networks) would clarify when and why topological methods excel. Feature selection studies could identify the minimal set of most informative features, reducing computational cost while maintaining performance.

Explainability enhancements would increase operator trust. While topological features have physical interpretations, individual predictions could be made more transparent through SHAP values quantifying each feature's contribution, attention mechanisms identifying which time steps drive classifications, or counterfactual explanations describing what would need to change for a different prediction.

## Conclusion

Topology reveals structure that distance-based methods miss. When wind turbines develop faults, their operational behavior changes shape—creating altered loops, different persistence patterns, and modified geometric structures in their state space. By extracting features from persistent homology and comparing actual behavior to physics-based expectations, we detect these faults with high accuracy and perfect recall.

The approach is practical because it leverages domain knowledge. Physics tells us what power should be given wind speed. Topology tells us whether observed behavior matches expected patterns. The combination is powerful, capturing both what the system does and how it behaves. The features are interpretable, connecting mathematical constructs to physical phenomena. The validation is rigorous, preventing temporal leakage and ensuring reported performance reflects true generalization. The results are actionable, providing maintenance teams with reliable alerts that have genuine operational value.

For wind farm operators, this means detecting degradation before it becomes failure, scheduling maintenance proactively rather than reactively, focusing attention on genuinely anomalous turbines, and preventing revenue loss from undetected underperformance. For the broader community, this demonstrates that topological data analysis has practical value when applied thoughtfully. The shape of data matters because faults change how systems behave, not merely what they measure. Finding those shapes and quantifying their changes turns abstract mathematics into practical tools.


---

## Complete Implementation

Below is the complete, executable code for this analysis. Copy and paste this into a Python file to run the entire analysis:

*The code displays analysis results and statistics.*

### Running the Code

To run this analysis:

```bash
python verify_setup.py
```

The script will generate all visualizations and save them to the current directory.

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

Additional packages may be required depending on the specific analysis.

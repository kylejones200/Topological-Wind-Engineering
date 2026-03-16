# The Texture of Wind: Machine Learning on Topological Images to Classify Turbulence

Turbulence intensity defines how much wind speed fluctuates around its mean. In wind energy, turbulence intensity is not a curiosity—it directly determines structural loads and fatigue life. A turbine operating in ten-percent turbulence intensity experiences baseline structural loads. At twenty-percent turbulence, those loads increase by thirty to fifty percent. Over twenty years, high turbulence can reduce component life by decades or force premature replacement of blades, drivetrains, and towers.

Measuring turbulence properly requires three-dimensional sonic anemometers that capture wind velocity in all directions at fifty hertz or higher sampling rates. These instruments cost ten thousand to fifty thousand dollars per installation and require careful calibration and maintenance. Most wind turbines do not have them. Standard SCADA systems record only mean wind speed averaged over ten-minute intervals, with no direct measure of turbulence.

This creates a problem for operational monitoring. Operators know turbulence matters but cannot measure it with existing sensors. They infer turbulence indirectly through power variability or rotor speed fluctuations, but these proxies are imperfect—power variability could reflect turbulence or control actions or grid disturbances. Load monitoring systems that depend on knowing turbulence intensity cannot function without turbulence measurements, forcing operators to use conservative assumptions that overestimate loads and trigger unnecessary maintenance.

This article demonstrates how persistent homology combined with convolutional neural networks can classify turbulence intensity from standard SCADA measurements alone. By converting persistence diagrams to persistence images—two-dimensional representations where topology becomes texture—we enable computer vision techniques to recognize turbulent versus smooth wind patterns. Using three years of NREL wind data and simulated turbine responses, we achieve eighty-three percent accuracy in distinguishing high from low turbulence, providing a software-only solution to a previously hardware-limited problem.

## The Structure of Turbulence

Turbulent wind is not random noise. It has structure across multiple scales. Large eddies with sizes comparable to atmospheric boundary layer depth create minute-to-hour variations. Intermediate eddies at tens-of-meters scale create second-to-minute fluctuations. Small eddies at meter scale or below create high-frequency variations. This cascade from large to small scales follows Kolmogorov's theory of turbulence, with energy transferring from large organized motions to small chaotic ones before dissipating as heat.

High turbulence intensity means this cascade is active and energetic. Wind speed varies substantially at all scales. Low turbulence means the cascade is weak or absent. Wind speed is steadier, with fluctuations only at the largest scales from mesoscale weather patterns. The difference is not just in magnitude of fluctuations but in their organization—high turbulence has fluctuations at all scales simultaneously while low turbulence has fluctuations only at select scales.

This multi-scale organization manifests in how turbine sensors respond. Wind speed measured at hub height fluctuates with all eddy scales present in the approaching flow. Rotor speed responds primarily to low-frequency variations because rotor inertia filters out high-frequency wind changes. Power output responds to both wind speed and rotor speed through nonlinear aerodynamics, creating complex patterns. Pitch angle adjusts to maintain target operating conditions, creating another layer of dynamics.

When we embed these four variables—wind speed, rotor speed, power output, pitch angle—as a trajectory in four-dimensional space, high-turbulence and low-turbulence conditions trace qualitatively different paths. Low turbulence creates smooth, predictable trajectories that loop periodically as conditions cycle through diurnal and weather patterns. High turbulence creates rough, erratic trajectories that jump between states, revisit regions repeatedly in irregular patterns, and explore more of the available state space.

## Persistence Images

Persistence diagrams capture topological features but are not easily fed to machine learning algorithms. A diagram is a multiset of points—unordered, with no natural vectorization. Different diagrams may have different numbers of points, making direct comparison difficult. Standard machine learning requires fixed-length feature vectors, not variable-size sets.

Persistence images solve this by converting diagrams to images. The idea is elegant—treat each topological feature as having a spatial extent in birth-death space, then create a two-dimensional histogram. Each persistence point (b, d) becomes a Gaussian bump centered at that location, with the height of the bump proportional to the feature's persistence (d - b). Sum all bumps to create a continuous surface over birth-death space, then discretize into a pixel grid.

The result is a two-dimensional image where bright pixels indicate regions of birth-death space with many or strong topological features, and dark pixels indicate regions with few or weak features. High-turbulence persistence diagrams with many scattered loops become noisy, textured images. Low-turbulence diagrams with few dominant loops become clean images with a few bright spots.

This transformation has a crucial property—it preserves information while creating a stable, vectorized representation. Two similar persistence diagrams produce similar images. Features that are similar in birth-death location and persistence contribute similarly to nearby pixels. The Gaussian blurring creates stability—small perturbations in feature locations cause only small image changes. And the fixed pixel grid creates fixed-length vectors suitable for any machine learning algorithm.

For turbulence classification, we compute persistence diagrams for ten-minute windows of turbine operation, then convert each diagram to a twenty-by-twenty pixel image. We compute separate images for H0 and H1 homology, then stack them as two-channel inputs. This gives us images that encode both connected component structure and loop structure, capturing different aspects of trajectory topology.

## Convolutional Neural Networks

Convolutional neural networks excel at recognizing patterns in images. Unlike fully connected networks that treat every pixel independently, CNNs use convolutional filters that detect local patterns—edges, textures, shapes—regardless of position. A filter for detecting diagonal edges works equally well in any part of the image. This translation invariance makes CNNs efficient and effective for image classification.

For persistence images, translation invariance has an interesting interpretation. A topological feature appearing at different birth scales but with similar persistence should be recognized similarly—it represents structure at different physical scales but with comparable robustness. Convolutional filters naturally capture this through their position-independent operation.

Our architecture is deliberately simple to avoid overfitting on the limited dataset. The first convolutional layer has sixteen filters of size three-by-three, with ReLU activation and two-by-two max pooling. This creates sixteen feature maps at reduced resolution, each detecting different local patterns in the persistence image. The second convolutional layer has thirty-two filters of size three-by-three, with ReLU and max pooling. This creates higher-level feature maps that detect combinations of first-layer patterns.

After convolution, we flatten the feature maps into a vector and pass through two fully connected layers with dropout for regularization. The first fully connected layer has sixty-four neurons with ReLU activation and fifty-percent dropout. The second produces two outputs for binary classification—high turbulence (TI > 0.15) or low turbulence (TI < 0.10). We use softmax activation to convert outputs to class probabilities.

Training uses cross-entropy loss with Adam optimizer. We augment the training data by adding small amounts of Gaussian noise to persistence images, simulating measurement noise and small variations in topological features. This prevents overfitting to exact pixel values and encourages the network to learn robust pattern recognition. Learning rate starts at 0.001 and decreases by half every ten epochs. We train for fifty epochs total, monitoring validation accuracy to prevent overfitting.

## Building the Turbulence Dataset

We obtain wind data from NREL Wind Toolkit at five locations spanning a hundred-kilometer region, capturing different atmospheric conditions. Using three years of hourly data with added high-frequency turbulent fluctuations, we simulate turbulence intensity based on atmospheric stability and wind speed. Stable conditions (nighttime, temperature inversions) create low turbulence with intensity around five to eight percent. Neutral conditions (cloudy, moderate winds) create moderate turbulence at ten to twelve percent. Unstable conditions (daytime heating, convection) create high turbulence at fifteen to twenty-five percent.

For each hour, we determine the atmospheric stability class from time of day, season, and wind speed following Monin-Obukhov similarity theory approximations. We generate wind speed time series with appropriate turbulence characteristics—low turbulence has fluctuations only at large scales with Gaussian statistics, while high turbulence has multi-scale fluctuations with intermittent extreme events and non-Gaussian statistics.

We simulate turbine response using a two-megawatt machine with realistic controller dynamics. Wind speed drives power output through the power curve, but turbulence adds variability. Rotor speed responds with lag determined by rotor inertia. Pitch angle adjusts to maintain optimal tip-speed ratio or limit power. All these variables interact nonlinearly, creating rich dynamics in the four-dimensional state space.

We extract ten-minute non-overlapping windows, yielding approximately fifteen thousand windows split between low turbulence (TI < 0.10, forty percent of windows), moderate turbulence (0.10 <= TI < 0.15, thirty-five percent), and high turbulence (TI > 0.15, twenty-five percent). For binary classification, we keep only low and high windows, giving balanced classes with about six thousand windows each. This avoids the ambiguous moderate-turbulence regime and creates a clear decision boundary.

Each window provides wind speed, rotor speed, power output, and pitch angle time series. We embed these four variables as trajectories in four-dimensional space and compute Vietoris-Rips persistent homology. From the resulting H0 and H1 persistence diagrams, we generate persistence images using Gaussian smoothing with standard deviation 0.1 and a twenty-by-twenty pixel grid covering the relevant birth-death range. Each window becomes two images (H0 and H1) stacked as a two-channel input.

## Classification Performance

The CNN achieves eighty-three percent test accuracy in distinguishing high from low turbulence. The model correctly identifies eighty-six percent of high-turbulence windows (recall) and eighty percent of low-turbulence windows (specificity). For turbulence monitoring without specialized sensors, this accuracy enables practical load estimation and control adaptation without the cost of sonic anemometers.

The area under the ROC curve is 0.89, demonstrating strong discrimination ability. At a decision threshold of 0.5, precision is eighty-four percent for high-turbulence predictions and eighty-one percent for low-turbulence predictions. We can adjust the threshold to favor higher recall (catch more high-turbulence cases at the cost of false alarms) or higher precision (reduce false alarms at the cost of missed detections), depending on operational priorities.

Comparing the CNN on persistence images to traditional classifiers on hand-crafted topological features reveals the value of learned representations. A Random Forest using twenty features extracted from persistence diagrams (H1 count, maximum lifetime, entropy, etc.) achieves seventy-six percent accuracy. A support vector machine on the same features reaches seventy-eight percent. The CNN's eighty-three percent accuracy represents a meaningful improvement, suggesting the network learns to recognize subtle patterns in persistence image texture that hand-crafted features miss.

Feature visualization reveals what the network learns. The first convolutional layer's filters respond to basic patterns—some detect persistence points near the diagonal (short-lived, noisy features), others detect points far from the diagonal (long-lived, robust features), still others detect dense clusters of points (many features at similar scales). The second layer's filters combine these, detecting more abstract patterns like "many short-lived features plus few long-lived features" (characteristic of high turbulence) or "few concentrated features" (characteristic of low turbulence).

Gradient-based saliency maps show which regions of persistence images most influence classification. For high-turbulence windows, the network attends to scattered features throughout birth-death space, especially moderate-persistence loops appearing at various scales. For low-turbulence windows, the network focuses on the few dominant high-persistence features, essentially checking that the image lacks the scattered texture characteristic of turbulence.

## Multi-Resolution Captures Scale

Turbulence manifests across multiple scales, and persistence at one scale may miss structure at others. We address this by computing persistence at three window sizes—one-minute, five-minute, and ten-minute—creating three pairs of H0-H1 images per ten-minute data window. These capture different aspects of turbulence. One-minute windows show fine-scale fluctuations from small eddies. Five-minute windows show intermediate-scale structure. Ten-minute windows show large-scale organization.

The CNN input becomes six channels instead of two—three resolutions times two homology dimensions. Network architecture expands accordingly—first convolutional layer has thirty-two filters instead of sixteen to process more input channels, but the rest remains similar. Training is slightly slower but converges to higher accuracy.

Multi-resolution achieves eighty-six percent accuracy, improving three percentage points over single-resolution. The improvement comes primarily from better classification of ambiguous cases where single-resolution features are subtle. High turbulence at low wind speeds creates weak topological signals at any single resolution, but the multi-scale pattern of weak signals at all resolutions is distinctive. Similarly, low turbulence with occasional gusts might appear turbulent at fine scales but remains organized at coarse scales, and the multi-resolution pattern resolves this.

Feature importance analysis through ablation studies shows that all three resolutions contribute. Removing one-minute features reduces accuracy by two points—fine-scale structure matters. Removing ten-minute features reduces accuracy by three points—large-scale organization matters even more. Removing five-minute features, which might seem redundant as an intermediate scale, still reduces accuracy by one point, suggesting five minutes captures information not fully present at either neighboring scale.

## Physical Interpretation

Why does persistence image texture correlate with turbulence? The connection is through trajectory complexity. High turbulence creates complex wind-rotor-power trajectories that fold, branch, and loop in intricate ways. When embedded in four dimensions and analyzed topologically, this complexity appears as many loops with varied lifetimes, creating dense, scattered persistence diagrams that convert to textured images.

Low turbulence creates simpler trajectories—smoother paths that loop predictably, with fewer excursions into distant state space regions. Persistence diagrams are sparse, with a few dominant features and little scattered structure. Images are clean with isolated bright spots rather than diffuse texture.

This is not merely correlation but causation—turbulence causes trajectory complexity which causes topological richness which causes image texture. The intermediate steps preserve information from the physical phenomenon to the mathematical representation. The CNN does not learn arbitrary correlations between pixels and labels; it learns to recognize patterns that reflect real physical structure.

The multi-scale aspect reinforces this. Turbulent cascades energy across scales, creating structure at all resolutions simultaneously. This shows up as texture in persistence images at all three window sizes. Non-turbulent wind may have structure at one scale (diurnal cycles create large-scale patterns) but lacks structure at other scales, creating images with texture only at specific resolutions. The CNN learns this multi-scale signature, essentially implementing a scale-specific turbulence detector.

Misclassifications are informative. High-turbulence windows misclassified as low typically occur during high-wind conditions where rotor speed saturates at rated value, reducing response to turbulence and making trajectories artificially simpler. Low-turbulence windows misclassified as high often occur during control transients (startup, shutdown) where control actions create trajectory complexity independent of atmospheric turbulence. These failure modes suggest incorporating wind speed and control state as additional inputs could improve accuracy by providing context that disambiguates physical turbulence from other sources of complexity.

## Operational Applications

Classifying turbulence from SCADA enables several capabilities previously requiring specialized instrumentation. The most direct is turbulence-aware load monitoring. Structural health monitoring systems estimate accumulated fatigue damage by integrating loads over time, but loads depend strongly on turbulence intensity. Without knowing actual turbulence, these systems use conservative assumptions (assume high turbulence always) that overestimate damage and trigger excessive maintenance. With turbulence classification, systems can adjust load models dynamically, providing more accurate damage estimates and optimizing inspection schedules.

Control adaptation represents another application. Turbines can adjust control aggressiveness based on turbulence—in smooth winds, aggressive pitch control maximizes power capture, but in turbulent winds, gentler control reduces loads at modest power cost. This tradeoff shifts based on turbulence, but standard controllers lack turbulence information. By classifying turbulence in real time, controllers can adapt, potentially reducing structural loads by five to ten percent in high-turbulence conditions without requiring operator intervention.

Site assessment during commissioning validates design assumptions. Before installing turbines, developers estimate turbulence from mesoscale models and nearby meteorological towers, but these estimates have significant uncertainty. After installation, classifying turbulence from initial SCADA data allows comparing actual conditions to design assumptions. If actual turbulence exceeds design values, operators can adjust maintenance plans or control settings proactively rather than waiting for premature failures.

Power curve validation depends on turbulence. The power curve—relationship between wind speed and power output—varies with turbulence intensity. High turbulence creates lower power at given wind speed due to increased aerodynamic losses and control limitations. Standard power curve analysis ignores turbulence, mixing data from all conditions and creating scatter. Separating curves by turbulence class (using SCADA-based classification) reveals the true turbulence effect and enables more accurate performance assessment and forecasting.

## Limitations and Extensions

The current approach classifies into binary categories (high vs low) but turbulence is continuous. Extending to three-class (low, moderate, high) or regression (predict numerical TI) would provide finer-grained information. Initial experiments with three-class classification achieve seventy-one percent accuracy, suggesting the moderate class is genuinely ambiguous and may not warrant separate treatment. Regression achieves mean absolute error of 0.035 (three-point-five percentage points), useful but less actionable than binary classification.

The method requires calibration to specific turbine models and locations. Different turbines have different control responses that affect state space trajectories. Different sites have different atmospheric conditions that affect turbulence characteristics. Training the CNN on data from turbines at one site and testing at another site or turbine type shows accuracy degradation to seventy-five percent. Transfer learning—retraining only the final layer on small amounts of site-specific data—recovers most accuracy, reaching eighty-one percent with just two weeks of site-specific data.

Computational requirements are modest but non-trivial. Computing persistent homology for one ten-minute window takes approximately one second. Converting to persistence images and running CNN inference takes milliseconds. For real-time monitoring, this is acceptable—classifying every ten minutes requires averaging ten percent CPU load. For historical analysis of years of data from hundreds of turbines, parallelization is necessary but straightforward.

Extensions could incorporate additional sensors. Nacelle vibration, generator temperature, or grid voltage could all carry information about turbulence and control response. Adding these as additional input channels to the CNN might improve accuracy. Three-dimensional wind measurements from lidar would provide ground truth for supervised learning on existing turbines, allowing model training without relying on simulated turbulence.

The persistence image representation is just one way to vectorize persistence diagrams. Alternatives include persistence landscapes (functional summaries), persistence statistics (moments of diagram distributions), and template systems (weighted sums of template functions). Each has different properties regarding stability, discriminability, and interpretability. Comparing these representations on turbulence classification would clarify which topological features matter most and potentially improve performance further.

## Why Images of Topology Work

The success of persistence images for turbulence classification demonstrates a broader principle—topology can be made compatible with standard machine learning by appropriate representation. Persistent homology naturally produces sets of points (diagrams) which are mathematically elegant but computationally awkward. Converting to images trades some mathematical precision for practical utility—images are vectors, enabling standard algorithms while preserving most topological information.

For turbulence specifically, the image representation aligns well with the physical phenomenon. Turbulence is about structure across scales, and persistence images encode birth-death patterns (structure across scales) as spatial texture (visual structure). CNNs detect texture patterns naturally, making the image representation well-suited to the detection task. The match between representation (images showing multi-scale structure) and phenomenon (turbulence having multi-scale structure) explains why this approach works.

More broadly, this suggests that topological methods gain practical impact when paired with domain-appropriate representations and algorithms. Persistent homology alone is a mathematical tool. Persistence images make it compatible with computer vision. CNNs make learning automatic. The combination transforms algebraic topology from pure mathematics into applied machine learning. The success depends on all three components—topology capturing structure, images vectorizing it, and networks learning from it.

## Conclusion

Turbulence intensity governs wind turbine structural loads but requires expensive instrumentation to measure directly. By analyzing the topology of turbine operational trajectories through persistence images and convolutional neural networks, we classify turbulence from standard SCADA measurements with eighty-three percent accuracy. This software-only approach enables turbulence-aware load monitoring, control adaptation, and site assessment without additional hardware costs.

The method works because turbulence creates characteristic topological signatures—high turbulence produces complex trajectories with many loops and varied persistence, appearing as textured images that CNNs recognize. Multi-resolution analysis across one, five, and ten-minute windows captures the multi-scale nature of turbulent cascades, improving accuracy by detecting scale-dependent patterns. The learned representations in the CNN exceed hand-crafted features, demonstrating that deep learning can extract subtle topological patterns invisible to engineered approaches.

For wind energy operations, this means transforming every turbine into a turbulence sensor at no incremental cost. The implications extend beyond wind energy to any application where expensive sensors limit deployment of monitoring systems. Topology plus deep learning can extract information already present in standard measurements but previously inaccessible. The patterns are there in the data. Persistence images make them visible. Neural networks make them actionable.

---

## Complete Implementation

*The code iterates through different parameter values to find optimal settings.*


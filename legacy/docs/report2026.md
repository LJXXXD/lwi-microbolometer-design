Project Status Report: V2 Multispectral Sensor Optimization Summary

1. Project Foundations & V1 Retrospective

Technical Overview This report details the advancement of our computational spectral filter optimization framework for a single-pixel, non-imaging multispectral sensor. Operating strictly within the spectral domain, the device utilizes a discrete array of sub-pixels, each characterized by a specific spectral response function. The physical substrate for these filters is a microbolometer metasurface, where the goal is to optimize the metasurface geometry to maximize the differentiability of target substances.

Physical Modeling The sensor's response is modeled via Gaussian basis functions, defined by center wavelength (\mu) and bandwidth (\sigma). Our physics pipeline employs a forward-modeling approach to simulate sensor output by integrating:

* Planck’s Law: Specifically incorporating the n^2 factor to account for thermal emission within a medium of refractive index n.
* Atmospheric Transmittance: Modeling signal attenuation across defined path lengths.
* Emissivity Fingerprints: Utilizing infrared emissivity spectra for a challenging test set of four visually identical white powders: Cocaine, Heroin, Sodium Bicarbonate, and Sucralose.

These components are combined via spectral convolution to generate an M-channel output vector for each substance.

Critique of V1 Grid Search The V1 framework relied on an exhaustive grid search, evaluating combinations within discrete bounds: \mu \in [4, 20]\, \mu m at 0.5\, \mu m increments, and \sigma \in \{0.1, 0.5, 1.0, 2.0, 4.0\}\, \mu m. The O(N^M) complexity of an exhaustive search for M-channel configurations scales poorly and is restricted by the granularity of the grid, which likely bypasses local optima. While it established a baseline differentiability score of 47.41, the inefficiency of evaluating tens of millions of configurations necessitated a transition to heuristic global optimizers.

2. Evolutionary Optimization: Initial V2 Progress & Challenges

Genetic Algorithm (GA) Implementation To surpass the V1 baseline, we implemented an evolutionary search using the pygad library, allowing for exploration of a continuous parameter space. Initial runs successfully improved the differentiability score from 47.41 to approximately 55.50.

The Diversity Challenge: Mode Collapse The primary obstacle in standard GA implementation was "mode collapse," where the population converged prematurely to a single "best" solution. While achieving high fitness, the algorithm failed to preserve diverse design candidates, resulting in a final population of functional clones.

Diversity Mitigation Strategies We evaluated several configurations to counteract convergence and maintain population diversity:

Configuration	Niching (\sigma_{share})	Mutation	Best F	Full Div	Elite Div	High Perf (\ge 50)	Top 5 Cloned
Baseline GA	OFF	Standard	58.51	8.22	1.62	13	YES
Strong Niching	2.0	Custom	59.02	10.95	7.45	2	NO
Weak Niching	5.0	Custom	59.01	14.01	8.46	9	YES

Note: Custom mutations utilized adaptive step-sizing and stagnation detection to escape local optima.

Multi-Start Ensemble Strategy To verify the multimodality of the design landscape, we deployed an ensemble of 50 parallel independent GA runs. This strategy confirmed that the landscape contains at least 4–6 distinct "families" of high-performing architectures, suggesting that multiple divergent metasurface designs can achieve near-identical fitness levels (~59).

3. Quality Diversity (QD) and the MAP-Elites Pivot

The Conceptual Shift Recognizing that standard GAs are fundamentally designed for convergence rather than landscape mapping, we pivoted to MAP-Elites (Multi-dimensional Archive of Phenotypic Elites). This shifts our objective from "Survival of the Fittest" to the "Illumination of the Design Map," where we seek the optimal solution for every possible niche in the architectural space.

Feature Space Construction We partitioned the design space into a 20x20 feature grid (400–900 possible cells) indexed by the smallest center wavelength (\mu_1) and the second smallest center wavelength (\mu_2). This specific indexing ensures that \mu_1 < \mu_2, preventing redundant cell assignments and forcing the algorithm to preserve the best design for each unique spectral combination.

Architectural Outcomes MAP-Elites generated a library of approximately 20 distinct design families. This provides a significant strategic advantage: a broad selection of viable sensing strategies that offer flexibility for fabrication and manufacturing constraints without sacrificing performance.

4. Advanced Refinement: Local Polish and Hybridization

The Hybrid Strategy While MAP-Elites is robust at global exploration (identifying high-fitness "basins"), it is less efficient at the fine-grained exploitation required to reach true mathematical peaks. We addressed this by implementing a hybrid strategy:

1. Global Exploration: MAP-Elites identifies the broad architectural families.
2. Local Polish: We applied gradient-free Hill Climbing to the top individuals in the archive to maximize their specific fitness.

Quantifying Performance Peaks Through this hybrid refinement, the framework achieved a final optimized fitness score of 59.49. This represents a ~25.5% improvement over the original V1 baseline of 47.41, demonstrating the power of combined global illumination and local search.

5. Environmental Robustness & Future Trajectory

Addressing the Simulation-Reality Gap Conference feedback highlighted the "simulation-reality gap" as a critical hurdle. We are addressing this by building robustness margins into the design. If a sensor achieves a 50° Spectral Angle Mapper (SAM) score in simulation, it can theoretically withstand up to 30° of real-world degradation while remaining effective. We are currently testing designs against:

* Thermal Perturbations: Evaluating performance at 273K, 293K, and 313K.
* Atmospheric Variability: Adjusting path lengths and refractive index ratios.

Minimax Optimization Approach We are transitioning to a Robust Optimization (Minimax) framework. In this paradigm, fitness is redefined as the worst-case performance across a range of environmental scenes: Fitness_{robust} = \min(fitness_{273K}, fitness_{293K}, fitness_{313K}) By maximizing the minimum performance, we ensure that the sensor design is resilient to environmental fluctuations.

Development Roadmap

1. Noise Robustness Integration: Expanding the minimax framework to account for detector noise and stochastic perturbations.
2. RLC Physics Integration: Replacing Gaussian approximations with a high-fidelity RLC physics model to more accurately represent the microbolometer metasurface's absorptance.
3. Surrogate-Assisted Optimization: Developing Neural Network-based surrogate models to approximate the physics integration, potentially accelerating the MAP-Elites search by 100x.

6. Technical Glossary of Metrics

SAM (Spectral Angle Mapper) :   A metric measuring the angular separation (in degrees) between substance fingerprints. It treats sensor outputs as vectors in M-dimensional space; an angle of 90° indicates maximal differentiability.

Differentiability Score :   The minimum off-diagonal value in the SAM distance matrix. This prioritizes the "worst-case" separability to ensure the two most similar substances in a set remain distinguishable.

Hungarian Algorithm :   A combinatorial optimization method used for permutation-invariant distance calculations. It ensures that two sensor designs are recognized as identical even if their channels are listed in a different order (e.g., [Channel_A, Channel_B] vs [Channel_B, Channel_A]).

iVAT (Improved Visual Assessment of Cluster Tendency) :   An unsupervised visualization tool used to reorder distance matrices. In this project, it is used to visually identify and verify the distinct design "families" within the high-performance population.

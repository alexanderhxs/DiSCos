# Projekt-Tracker: DiSCo Python (Master Thesis)

## 📌 Executive Summary & Status
**Aktueller Fokus:** Ausweitung der DiSCo-Methodik auf multivariate Verteilungen unter Verwendung von Mixture-Modellen (L1) und Sliced Wasserstein (S-Wasserstein). Die Kern-Architektur (Solver, Preprocessing, Visualisierungen) ist auf multivariaten Support ausgelegt, aktuelle Baustelle ist die Rückprojektion der Sliced Quantile, sowie Inferenz-/Testmethoden im multivariaten Raum.

---

## 🚧 Current Sprint (In Progress)
- [ ] **Umgang bei Verschiedenen sample größen**: Aktell kann die radon_transform und der SlicedWassersteinSolver noch nicht unterschiedliche sample größen bei den controls handeln. Das stellt besonders ein Problem bei quantils filterungen dar. 

- [ ] **S-Wasserstein "evaluate_counterfactual" implementieren**: Aus den errechneten DiSCo-Weights müssen im S-Wasserstein-Solver die korrekten multivariaten Quantile / CDFs rekonstruiert werden. *(Aktuell offenes TODO in `solvers.py`)*.

- [ ] **Auswertungsframework (TEA)**: Logik zur Berechnung der *Treatment Effects on the Treated Distributions* (TEA) definieren und für mehrdimensionale Ergebnisse anpassen.

- [ ] **Fit-Metriken entwickeln**: Quantitative Metriken zur Evaluation des Distribution-Fits (z.B. Out-of-Sample Losses) implementieren.

---

## 🗣️ Open Discussion Points (Für Meetings)
- **Konfidenzintervalle mehrdimensional**: Wie genau definieren und berichten wir CIs für multivariate Daten? (Uniform vs. Point-wise Quantiles über gemeinsame Dichten?).
- **Permutationstests**: Aufbau der Permutation-Inference im multivariaten Fall abstimmen - auf Basis der marginalen Verteilungen oder zwingend Joint-Distribution-Teststatistiken?
- **Optimierung mit LHS**: Bietet Latin Hypercube Sampling (LHS) beim Sampling der Gitterpunkte (`getGrid` / `disco_weights_reg`) tatsächlich die erhoffte Performance/Konvergenz-Steigerung?

---

## 📋 Backlog: Data Science & Methodology
- [ ] **LHS Implementierung evaluieren**: Optimalen Einbau von LHS beim Datenpunkt-Sampling testen.
- [ ] **Multivariate Inference-Pipeline**: `disco_ci` und Parsing-Logiken (`parse_boots`) in `inference.py` vollständig für Nd-Arrays kompatibel gestalten (momentan Broadcasting-Risiken).
- [ ] **Multivariate Permutationstests**: Das klassische Permutationsverfahren auf die Joint-Distribution Abstände ausweiten.

---

## 💻 Backlog: Software Engineering & Refactoring
- [ ] **Tests schreiben**: Unit-Tests für `swasserstein.py` und die modifizierten `solvers.py` Methoden hinzufügen, um Dimensions-Bugs dauerhaft abzusichern.
- [ ] **Performance Profiling**: Die Radar-Rückprojektionen und Copula-Schätzungen auf große Datensätze Stresstesten und Laufzeiten analysieren.

---

## ✅ Recently Done (Changelog)
- [x] **S-Wasserstein Dimensions-Bugs**: `radon_transform` trennt nun Target- und Control-Samples fehlerfrei, auch bei unbalancierten Sample-Sizes (gefixt in `fit_weights`).
- [x] **Agnostische Visualisierungen**: Plots (`plot_fit_quantiles`, `plot_fit_cdf`, `plot_fit_joint_contour`, `plot_fit_copula`) dynamisieren sich jetzt solver-unabhängig für jede beliebige Datendimension.
- [x] **Multivariates Preprocessing**: Filterlogik für Quantile (`q_min`, `q_max`) ist jetzt vollständig broadcast-fähig für multidimensionale Zielarrays.
- [x] **OOP Signatur-Harmonisierung**: Verdeckten Parameter-Bug (`grid_ord` als Keyword-Arg) in den Unterklassen von `BaseSolver` beseitigt, was das Bootstrapping blockierte.

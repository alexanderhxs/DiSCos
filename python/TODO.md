# Projekt-Tracker: DiSCo Python (Master Thesis)

## 📌 Executive Summary & Status
**Aktueller Fokus:** Ausweitung der DiSCo-Methodik auf multivariate Verteilungen (Mixture-Modelle & Sliced Wasserstein). Die Code-Architektur ist grundlegend multivariat-fähig. Fokus liegt nun auf fundierter methodischer Auswertung (Inferenz, Tests, evaluative Visualisierung) sowie einer sauberen, wartbaren Code-Basis.

---

## Daten
- Arbeitstunden gegen Stundenlohn -> Mindestlohn als treatment
- Suchen nach eventuell interessantem Textdatensatz für sentiment shifts oder ähnliches.


## 🧠 Methodik & Konzeption (Für den Professor / Methodische Diskussionen)
*Theoretische, statistische und konzeptionelle Punkte für Besprechungen.*

### 🚧 Actively Researching / Developing:

- [ ] **Vergleich 2D vs. Bedingte 1D-Verteilung**: Konzeptionell eine Methode entwickeln, um die gemeinsame 2D-Verteilung gegen eine bedingte 1D-Verteilung (mit der einen Variablen als stetigen Einflussfaktor) vergleichen zu können.
- [ ] **Fit-Metriken entwickeln**: Quantitative Metriken zur strikten Evaluation des Distribution-Fits (z.B. Out-of-Sample Losses, Abstandsmaße) im multivariaten Raum konzipieren.
- [ ] **Auswertungsframework (TEA)**: Logik zur Berechnung der *Treatment Effects on the Treated Distributions* (TEA) auf mehrdimensionale Ergebnisse anpassen.

### 🗣️ Open Discussion Points (Methodik):
- **Konfidenzintervalle mehrdimensional**: Wie genau definieren und berechnen wir CIs für multivariate Daten sinnvoll? (Uniforme Range vs. Point-wise Quantiles über gemeinsame Dichten?).
- **Permutationstests / Inference**: Ablauf der Permutations-Inferenz für Nd-Fälle: Soll der Fokus auf den marginalen Verteilungen liegen oder setzen wir zwingend Joint-Distribution-Teststatistiken ein?
- **Optimierung mit LHS**: Bietet Latin Hypercube Sampling beim Sampling der Gitterpunkte (`getGrid` / S-Wasserstein Slices) eine signifikante Performance- oder Konvergenz-Steigerung?

---

## 💻 Implementierung & Softwaretechnik (Für Entwicklung / Code-Qualität)
*Konkrete To-Dos für den Codebau, Algorithmen und die Systemarchitektur.*

### 🚧 Current Sprint & Blocker:
- [ ] **Softwaretechnische Implementierung verbessern**: Das gesamte Paket konzeptionell und architektonisch aufräumen. Ziel: Eine konsistente, einheitliche objektorientierte Struktur (OOP) über alle abstrakten Solver und Utilities hinweg. Code-Smells minimieren!
- [ ] **Umgang bei verschiedenen Sample-Größen**: Im S-Wasserstein und der Radon-Transformation die Restriktionen bei unbalancierten Control-Gruppen endgültig auflösen (insbesondere problematisch nach asymmetrischen Quantils-Filterungen).
- [ ] **S-Wasserstein "evaluate_counterfactual" implementieren**: Aus den errechneten DiSCo-Weights müssen im S-Wasserstein-Solver noch die korrekten multivariaten Quantile / CDFs algorithmisch rekonstruiert werden *(Offen in `solvers.py`)*.

### 📋 Backlog & Testing:
- [ ] **Multivariate Inference-Pipeline refactoren**: `disco_ci` und Parsing-Logiken (`parse_boots`) in `inference.py` vollständig und robust für Nd-Arrays und deren Broadcasting-Verhalten gestalten.
- [ ] **Tests schreiben**: Unit-Tests für `swasserstein.py` und modifizierte `solvers.py` Methoden hinzufügen, um Dimensions-Bugs dauerhaft abzusichern.
- [ ] **Performance Profiling**: Radar-Rückprojektionen und Copula-Schätzungen auf große Datensätze anwenden und Laufzeiten / Memory überprüfen.

---

## ✅ Recently Done (Changelog)
- [x] **Agnostische Visualisierungen**: Plots (`plot_fit_quantiles`, `plot_fit_cdf`, etc.) dynamisieren sich jetzt solver-unabhängig für jede beliebige Datendimension.
- [x] **S-Wasserstein Dimensions-Bugs**: `radon_transform` trennt nun Target- und Control-Samples fehlerfrei, auch bei unbalancierten Sample-Sizes (gefixt in `fit_weights`).
- [x] **Multivariates Preprocessing**: Filterlogik für Quantile (`q_min`, `q_max`) ist jetzt vollständig broadcast-fähig für multidimensionale Zielarrays.
- [x] **OOP Signatur-Harmonisierung**: Verdeckten Parameter-Bug (`grid_ord` als Keyword-Arg) in den Unterklassen von `BaseSolver` beseitigt, was das Bootstrapping blockierte.

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

- [ ] **Vergleich 2D vs. Influencing Factors**: Konzeptionell eine Methode entwickeln, um die gemeinsame 2D-Verteilung gegen eine 1D-Verteilung (mit der einen Variablen als stetigen Einflussfaktor) vergleichen zu können. Dabei generell ein Framework entwickeln um diese mit einzubeziehen
- [ ] **Auswertungsframework (TEA)**: Logik zur Berechnung der *Treatment Effects on the Treated Distributions* (TEA) auf mehrdimensionale Ergebnisse anpassen.
- [ ] **Tangential Wasserstein**: Einbauen von Tangential wasserstein methode als zusätzliche vergleichsmethode
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

##### 🗣️ Open Discussion Points (Methodik):
- **Konfidenzintervalle mehrdimensional**: Wie genau definieren und berechnen wir CIs für multivariate Daten sinnvoll? (Uniforme Range vs. Point-wise Quantiles über gemeinsame Dichten?).
- **Permutationstests / Inference**: Ablauf der Permutations-Inferenz für Nd-Fälle: Soll der Fokus auf den marginalen Verteilungen liegen oder setzen wir zwingend Joint-Distribution-Teststatistiken ein?
- **Optimierung mit LHS**: Bietet Latin Hypercube Sampling beim Sampling der Gitterpunkte (`getGrid` / S-Wasserstein Slices) eine signifikante Performance- oder Konvergenz-Steigerung?


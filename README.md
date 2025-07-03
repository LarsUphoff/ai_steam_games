# Künstliche Intelligenz – Steam Games Analyse-Tool

## Datensatz
**Steam Games Dataset** – Nutzung historischer Spieledaten von der Plattform Steam zur Analyse und Vorhersage von Erfolgspotenzialen.
https://www.kaggle.com/datasets/artermiloff/steam-games-dataset

## Forschungsfrage
**Künstliche Intelligenz in der Spielentwicklung**

*Wie kann Künstliche Intelligenz Entwickler bei der datenbasierten Planung erfolgreicher Steam-Releases unterstützen?*

## Mögliches Endprodukt
**Dashboard-Tool für Gameentwickler**

### Ziel
Unterstützung bei der **Analyse und Planung** neuer Spiele basierend auf historischen Steam-Daten.

## Eingabeparameter (Beispiel)

| Kategorie | Beschreibung |
|-----------|--------------|
| Kategorie(n) | Haupt- und Unterkategorien des Spiels |
| Genre | Genre des Spiels (z. B. RPG, FPS, Simulation etc.) |
| Betriebssystem (OS) | Zielplattform(en): Windows, macOS, Linux etc. |
| Sprache | Unterstützte Sprachen |
| Spielbeschreibung | Ausführliche Beschreibung des geplanten Spiels |
| DLC | Angabe zu möglichen Erweiterungen (Downloadable Content) |
| Tags | Schlagwörter für die Spielklassifikation |
| Achievements | Anzahl der freischaltbaren Erfolge im Spiel |

## Analysefunktionen & Ausgaben

| Funktion | Beschreibung |
|----------|--------------|
| Charts & Statistiken | Genre-Trends, Spielerzahlen, historische Marktanalysen |
| Erfolgsvorhersage | Prognose des Spielpotenzials (Score oder Prozentwert) |
| Erwartete Downloads | Schätzung der ersten Downloads oder des Marktinteresses |
| Key Facts (Optional) | Übersicht zu Marktlücken, z. B. wenig vertretene Genres oder wachsende Trends |

## Zielgruppe
Indie-Studios, Game Designer, Publisher, Marktanalysten im Gaming-Bereich

## Project Organization

```
├── README.md          <- Die oberste README-Datei für Entwickler, die dieses Projekt verwenden
├── requirements.txt   <- Die Requirements-Datei zur Reproduktion der Analyseumgebung
│
├── data/
│   ├── processed/     <- Die finalen, kanonischen Datensätze für die Modellierung
│   │   ├── cleaned_dataset_head.csv
│   │   ├── pre_scaling_dataset_head.csv
│   │   └── steam-users-by-country-2021-with-languages.json
│   └── raw/           <- Der ursprüngliche, unveränderliche Daten-Dump
│       └── games.csv
│
├── models/            <- Trainierte und serialisierte Modelle, Modellvorhersagen oder Modellzusammenfassungen
│   ├── gradient_boosting/
│   ├── random_forest/
│   └── xgb_regressor/
│
├── notebooks/         <- Jupyter Notebooks für Datenexploration und Modelltraining
│
└── src/                        <- Quellcode für dieses Projekt
   ├── metadata_options.py      <- Metadaten und Optionen für das Interface
   ├── scaler_comparison.py     <- Vergleich verschiedener Skalierungsverfahren
   ├── data_preprocessing/      <- Module für Datenvorverarbeitung
   └── ...
```

## APIs
- FastAPI ist ein leistungsstarkes Webframework zum Erstellen von HTTP-basierten Service-APIs in Python 3.8+. Wir nutzen diese API zur Verbindung zwischen Python-Backend-Logik und Frontend.

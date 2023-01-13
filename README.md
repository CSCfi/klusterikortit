![AuroraAI-logo](assets/auroraai-small.png)

# Klusterikortit

Tässä tietovarastossa on saatavilla ohjelmakoodit AuroraAI-projektin Tilannekuvan rakentajan käsikirjassa esitellyille Klusterikortit-piloteille: *Mun ripari* ja *Kouluterveyskysely*. Mun riparin pilotti on tehty yhteistyössä Kirkkohallituksen kanssa ja Kouluterveyskyselyn pilotti yhteistyössä THL:n kanssa.

## Aineistot

* Marketing Campaign: Vapaasti saatavilla oleva esimerkkiaineisto Klusterikorttien testaamiseeen. Aineisto on ladattavissa osoitteesta https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign (`marketing_campaign.csv`, Version 8). Aineistoon liittyvä muuttujamatriisi (`marketing-campaign-muuttujat.xlsx`) sijaitsee alihakemistossa [data](data).
* Mun ripari:
* Kouluterveyskysely:

## Lähdekoodit

Pilottiprojektit on toteutettu Python-ohjelmointikielellä. Esikäsittelyssä ja klusterien muodostamisessa käytetään Jupyter Notebookeja, jotka sijaitsevat alihakemistossa [notebooks](notebooks). Klusterikorttien visualisoinnit on toteutettu Dash-kirjastolla, ja nämä skriptit sijaitsevat alihakemistossa [dashboards](dashboards).

## Analyysin vaiheet

### Esikäsittely

* Marketing Campaign: [notebooks/prepare-data-mc.ipynb](notebooks/prepare-data-mc.ipynb)
* Mun ripari: [notebooks/prepare-data-munripari.ipynb](notebooks/prepare-data-munripari.ipynb)
* Kouluterveyskysely: [notebooks/prepare-data-ktk21.ipynb](notebooks/prepare-data-ktk21.ipynb)

### Klusterien muodostaminen

* [notebooks/generate-clusters.ipynb](notebooks/generate-clusters.ipynb)

### Visualisointi

* Marketing Campaign: [dashboards/app-mc.py](dashboards/app-mc.py)
* Mun ripari: [dashboards/app-munripari.py](dashboards/app-munripari.py)
* Kouluterveyskysely: [dashboards/app-ktk21.py](dashboards/app-ktk21.py)
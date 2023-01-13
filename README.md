![AuroraAI-logo](assets/auroraai-small.png)

# Klusterikortit

Tässä tietovarastossa on saatavilla ohjelmakoodit AuroraAI-projektin Tilannekuvan rakentajan käsikirjassa esitellyille Klusterikortit-piloteille: *Mun ripari* ja *Kouluterveyskysely*. Mun riparin pilotti on tehty yhteistyössä Kirkkohallituksen kanssa ja Kouluterveyskyselyn pilotti yhteistyössä THL:n kanssa.

## Aineistot

* Marketing Campaign: Vapaasti saatavilla oleva esimerkkiaineisto Klusterikorttien testaamiseeen. Aineisto on ladattavissa osoitteesta https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign (`marketing_campaign.csv`, Version 8). Aineistoon liittyvä muuttujamatriisi (`marketing-campaign-muuttujat.xlsx`) sijaitsee alihakemistossa [data](data).
* Mun ripari: [Mun ripari](https://evl.fi/uutishuone/puheenvuorot/-/article/91182786/Mun+ripari+-sovellus+rippikoululaisten+hyvinvoinnin+tueksi) on Suomen evankelis-luterilaisen kirkon seurakuntien rippikoululaisille tarkoitettu mobiilisovellus, jossa tarkoituksena on kartoittaa nuoren elämäntilannetta yksi hyvinvoinnin ulottuvuus kerrallaan. Sovellus on toteutettu vaiheittaisena polkuna yhdeltä kuvitteelliselta leirinuotiolta toiselle. Pilotissa on käytetty Mun ripari -sovelluksen prototyyppiä, mitä on testattu useilla rippikouluilla eri puolella Suomea.
* Kouluterveyskysely: THL:n [Kouluterveyskysely](https://thl.fi/fi/tutkimus-ja-kehittaminen/tutkimukset-ja-hankkeet/kouluterveyskysely) tehdään joka toinen vuosi alakoulun 4-5-luokkalaisille, yläkoulun 8-9-luokkalaisille sekä toisella asteella 1-2 vuoden opiskelijoille. Kouluterveyskyselyssä kysytään laajasti muun muassa mielenterveyteen, päihteisiin, koulunkäyntiin, perhesuhteisiin, harrastuksiin ja kiusaamiseen liittyviä asioita. Pilotissa käytetty aineisto on muodostettu vuosina 2019 ja 2021 toteutetuista yläkoululaisten Kouluterveyskyselyistä.

## Lähdekoodit

Pilottiprojektit on toteutettu Python-ohjelmointikielellä. Esikäsittelyssä ja klusterien muodostamisessa käytetään Jupyter Notebookeja, jotka sijaitsevat alihakemistossa [notebooks](notebooks). Klusterikorttien visualisoinnit on toteutettu Dash-kirjastolla, ja nämä skriptit sijaitsevat alihakemistossa [dashboards](dashboards).

## Analyysin vaiheet

### 1. Esikäsittely

* Marketing Campaign: [notebooks/prepare-data-mc.ipynb](notebooks/prepare-data-mc.ipynb)
* Mun ripari: [notebooks/prepare-data-munripari.ipynb](notebooks/prepare-data-munripari.ipynb)
* Kouluterveyskysely: [notebooks/prepare-data-ktk21.ipynb](notebooks/prepare-data-ktk21.ipynb)

### 2. Klusterien muodostaminen

* [notebooks/generate-clusters.ipynb](notebooks/generate-clusters.ipynb)

### 3. Visualisointi

* Marketing Campaign: [dashboards/app-mc.py](dashboards/app-mc.py)
* Mun ripari: [dashboards/app-munripari.py](dashboards/app-munripari.py)
* Kouluterveyskysely: [dashboards/app-ktk21.py](dashboards/app-ktk21.py)
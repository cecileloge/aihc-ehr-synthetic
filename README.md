# Generating Synthetic Structured EHR Data

Excerpt from exploration work done in the Stanford ML Group | Winter 2021

---

## Description   
The goal of the project was to explore techniques to generate Synthetic EHR Data that:
* Is ***indistinguishable*** from Real EHR Data (Data Generation Model)
* Is as ***useful*** for medical research as Real EHR Data (Proof of Utility - via a selection of prediction tasks)
* Cannot be traced back to any real patient and is ***fully shareable*** (Privacy Preservation)

We used OMOP-CDM data for all our explorative and experimental work - this data is highly protected and obviously not available here. 
You can find documentation online at [OHDSI.org](https://www.ohdsi.org/data-standardization/).
> Our mission at OHDSI is to improve health by empowering a community to collaboratively generate the evidence that promotes better health decisions and better care. The Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM) is an open community data standard, designed to standardize the structure and content of observational data and to enable efficient analyses that can produce reliable evidence.

---

## In this repository
You will find code samples from my exploration work, including:
* An efficient setup for pulling OMOP-CDM data - using SQL.
* A great setup for experimental tasks at scale - using Pytorch, Pytorch Lightning. It is complete with a full DataLoader + Pipeline.
* Various models (Logistic Regression, RNNs, Transformers) for Prediction tasks and tests.

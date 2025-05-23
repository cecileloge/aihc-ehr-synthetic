
---
EoL & Surgical Procedure Pipeline:

- add non-temporal features to the data (e.g. sex, age)?
- tune the Transformer model - current is AUC = 0.59

---
TBD:

- make the time embedding sinusoïdal? ON HOLD / DECIDE LATER

---
DONE:

- credentials to be auto adjusted based on user >> DONE
- data shape to be encoded (right now it's manual) >> DONE
- add Surgical Procedure as a second prediction task (and create corresponding cohort & data on GCP) >> DONE 
- add Accuracy as a metric to track in model >> DONE
- add temporal marker in DataLoader & DataModule >> DONE
- check temporal data format with team >> DONE
- check SARD code and set it up into a Lightning Module >> DONE
- Transformer Model = set up the data format right (either before or after masking) DONE
- Solve the Masking issue DONE
- add more metrics to track (Precision, Recall, etc.) >> DONE ADDED AUC
- make the train/val/test breakdown a parameter >> DONE

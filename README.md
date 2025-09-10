# X-ray Image Stitcher

## Descrizione
Questo progetto implementa lo **stitching di immagini radiografiche (X-ray)** mediante tecniche di **Feature Detection** e **Matching**. L'obiettivo è unire sequenze di immagini dicom prodotte dalla macchina, applicando algoritmi di rilevamento e matching dei keypoint per ottenere un risultato finale coerente e uniforme.  

---

## Algoritmi utilizzabili

- **Keypoint detection**: SIFT, ORB, BRISK, AKAZE, KAZE, SuperPoint.
- **Keypoint matching**: BFMatcher, FLANN, Superglue.
- **Calcolo omografia**: RANSAC, PROSAC, LMEDS.

---

## Funzionalità implementate

- **Lettura file DICOM**: estrazione delle immagini dai file prodotti dalla macchina.  
- **Collimazione**: possibilità di applicare un "cropping" per eliminare le zone coperte fisicamente dal collimatore.  
- **Keypoint detection** sulle zone di sovrapposizione tra immagini.  
- **Lowe's ratio test**: algoritmo di filtraggio dei match per rimuovere corrispondenze errate.  
- **Alpha blending**: modifica il valore di opacità delle immagini per ottenere un risultato finale "smooth".  
- **Stitch di sequenze di immagini**: possibilità di elaborare e unire più immagini in sequenza.  
- **Supporto per immagini raw**: possibilità di caricare immagini non elaborate e applicare i risultati di detection/matching delle immagini preprocessate.  

---

## SuperPoint & Superglue

- Porting degli algoritmi da Python (PyTorch) a **C++** (**ONNX**), eseguibili anche su CPU.
- Interfaccia unificata con i detector e matcher di OpenCV, permettendo combinazioni miste.

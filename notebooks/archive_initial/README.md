# Initial notebooks (archived)

These are the notebooks from my first analysis during the internship, kept unchanged apart from the data path. They use a **random** train/test split, so rows from the same paper can appear in both sets. That is why their scores (for example R² = 0.85 for chemical-synthesis size) are higher than the paper-grouped results in `../Chemical_Synthesis_v2.ipynb` and `../Green_Synthesis_v2.ipynb`. See the main README, sections 5 and 6.

To rerun them, export each sheet of the Excel files in `data/` to CSV with the same base name (for example `Nanoparticles_Synthesis_Dataset - Chemical_Synthesis.csv`) and place it in `data/`. They were written for Python 3.9 and scikit-learn 0.24.2.

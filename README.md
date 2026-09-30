# ARC-AGI-2 DSL Baseline

This repository contains my baseline solution for the **[ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)** Kaggle competition. 

## 🧠 Project Overview
The Abstraction and Reasoning Corpus (ARC) evaluates artificial general intelligence through grid-based puzzle-solving. 

This repository highlights a search-based solution using a Domain Specific Language (DSL). The current entry is a **DSL baseline with a depth <= 2 primitive search**, which acts as a foundational placeholder. A more advanced Test-Time Training (TTT) approach is being developed iteratively based on this structure.

### 🏆 Kaggle Submission Details
*   **Competition:** ARC Prize 2026 - ARC-AGI-2
*   **Kernel/Notebook:** [ARC-AGI-2 DSL Baseline](https://www.kaggle.com/code/shohinurpervezshohan/arc-agi-2-dsl-baseline)
*   **Kaggle Profile:** [@shohinurpervezshohan](https://www.kaggle.com/shohinurpervezshohan)

## 📁 Repository Structure

*   **`src/`**: The core source code containing the DSL implementation and search algorithms.
    *   `dsl.py`: Minimal grid-transform primitive library (Hodel/icecuber-style DSL). Every primitive is a pure function `Grid -> Grid`.
    *   `search.py`: The search module that composes the DSL primitives to find rules.
*   **`scripts/`**: Utilities for evaluation (e.g., `eval_ttt.py`).
*   **`kaggle_kernel/` & `kaggle_kernel_output/`**: Assets and notebooks configured for seamless execution within the Kaggle environment.
*   **`data_repo/` & `dataset.csv`**: Local copies of the required datasets used for feature extraction and search validation.
*   **`ranker.npz`**: Compressed NumPy weights for the model that evaluates and scores candidate solutions.

## 🚀 Future Roadmap
*   Integrate and evaluate Test-Time Training (TTT) strategies.
*   Expand the DSL primitive library for better coverage of complex transitions.
*   Refine the ranker model to improve candidate selection accuracy.

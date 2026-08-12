# Documents Directory Overview

This repository contains all research literature, project reports, documentation, diagrams, and interactive simulators related to the Agentic Digital Twin project for Fused Deposition Modeling (FDM).

The folder is systematically organized by topic to make it easy to find related papers and project assets. 

---

## 📂 Folder Structure

```text
Documents/
├── 01_BAL_and_Bayesian_Optimization/
├── 02_Digital_Twins_in_AM/
├── 03_FDM_Process_Optimization/
├── 04_Books_and_Foundations/
├── 05_Project_Reports_and_Documentation/
├── 06_Figures_and_Diagrams/
└── 07_Interactive_Simulations/
```

---

## 📑 Detailed File Previews


### 📁 `01_BAL_and_Bayesian_Optimization`
*Research papers focused on Bayesian Active Learning, Local Penalization batch sampling, and Active Constraint Learning.*

* **`Gonzalez_2016_Batch_Bayesian_Optimization_Local_Penalization.pdf`**
  * *Preview:* Introduces the local penalization approach for batch Bayesian optimization, a method to efficiently sample multiple diverse points per iteration.
* **`Khatamsaz_2025_Constrained_Bayesian_Optimization.pdf`**
  * *Preview:* Explores Bayesian optimization over problem formulation space to accelerate alloy development and autonomous experimentation.
* **`Lee_2022_Failure_Averse_Active_Learning.pdf`**
  * *Preview:* Proposes a failure-averse active learning framework for physics-constrained systems to avoid sampling unsafe or physically impossible states.
* **`Li_2026_Bayesian_Optimization_Active_Constraint_Learning.pdf`**
  * *Preview:* Presents Bayesian optimization combined with active constraint learning for advanced manufacturing process design.
* **`Nasrin_2023_Active_Learning_Tensile_Properties_FDM.pdf`**
  * *Preview:* Details active learning methodologies specifically tailored for the prediction of tensile properties in material extrusion additive manufacturing.
* **`Shahriari_2016_Taking_Human_Out_of_Loop_Review_Bayesian_Optimization.pdf`**
  * *Preview:* A seminal review paper on Bayesian optimization, highlighting its mathematical foundations, acquisition functions, and real-world applications.

### 📁 `02_Digital_Twins_in_AM`
*Papers discussing Digital Twin architectures, Agentic AI integration, and sensor applications in Additive Manufacturing.*

* **`Kantaros_2022_3D_Printing_Digital_Twins_Trends_and_Limitations.pdf`**
  * *Preview:* Discusses current trends, sensor integration challenges, and limitations in the implementation of digital twins for 3D printing.
* **`Multiscale_Digital_Twin_Laser_DED_Process.pdf`**
  * *Preview:* Explores a multiscale digital twin approach specifically for the laser-DED (Directed Energy Deposition) additive manufacturing process.
* **`Shen_2024_Digital_Twins_in_Additive_Manufacturing_Review.pdf`**
  * *Preview:* A state-of-the-art critical review of digital twin methodologies and their applications in additive manufacturing.
* **`Timms_2025_Agentic_AI_for_Digital_Twin.pdf`**
  * *Preview:* Details the architecture and implementation of autonomous Agentic AI frameworks to drive Digital Twin systems.

### 📁 `03_FDM_Process_Optimization`
*Literature on Fused Deposition Modeling (FDM) parameter optimization, mechanical properties, and toolpath strategies.*

* **`Fleming_Toolpath_Planning_Continuous_Extrusion_AM.pdf`**
  * *Preview:* Discusses continuous extrusion toolpath planning strategies for additive manufacturing to minimize defects.
* **`Maniraj_2026_Optimizing_FDM_Parameters_PLA.pdf`**
  * *Preview:* Focuses on optimizing FDM process parameters to enhance the dimensional accuracy and tensile strength of bio-based PLA.
* **`Shen_2025_Review_FDM_Application_Materials_Optimization.pdf`**
  * *Preview:* A comprehensive review of FDM applications, materials, parameter optimization techniques, and numerical studies.

### 📁 `04_Books_and_Foundations`
*Core textbooks and fundamental references for additive manufacturing.*

* **`Gibson_2010_Additive_Manufacturing_Technologies_Book.pdf`**
  * *Preview:* The foundational textbook "Additive Manufacturing Technologies" covering the core principles, processes, and materials of 3D printing.

### 📁 `05_Project_Reports_and_Documentation`
*Internal project documentation, LaTeX sources, compiled reports, and presentation scripts.*

* **`Brain_of_Digital_Twin_Documentation.pdf`**
  * *Preview:* The compiled technical documentation of the autonomous pipeline using LangGraph, GPR, and BAL.
* **`Project_Overview_and_Plan_of_Action.pdf`**
  * *Preview:* The master project overview, explaining what AM is, its advantages, limitations, and the strategic plan of action.
* **`FDM_Agentic_Digital_Twin_Complete_Documentation.md`**
  * *Preview:* Exhaustive technical markdown documentation covering the FDM Agentic Digital Twin concept, architecture, and code logic.
* **`GPR_and_BAL_PPT_Explanation.md`**
  * *Preview:* A simplified, intuitive explanation of GPR and BAL designed for a non-technical audience or evaluation panel.
* **`Assesement.md`**
  * *Preview:* A brutally honest self-assessment of the FDM Digital Twin Pipeline's strengths, limitations, and areas for future improvement.
* **`results_presentation_script.md`**
  * *Preview:* Script and visual guide designed to help structure the "Results" section of the final presentation.

### 📁 `06_Figures_and_Diagrams`
*Process flowcharts, system architecture diagrams, and mathematical explanation assets.*

* **`fig_active_learning_loop.png`**: Flowchart depicting the active learning iterative loop.
* **`fig_bo_acquisition_surface.png`**: Visual representation of the Bayesian Optimization acquisition function surface.
* **`fig_frenkel_sintering.png`**: Diagram illustrating the Frenkel sintering physics and neck growth in FDM.
* **`fig_system_architecture.png`**: The overall system architecture diagram of the LangGraph agentic pipeline.
* **`bal_diagram.txt`**: Prompt and structural layout for generating a publication-quality diagram of the Bayesian Active Learning pipeline.
* **`gpr_working.txt`**: Prompt and structural layout for generating a detailed diagram of the Gaussian Process Regression mathematical pipeline.

### 📁 `07_Interactive_Simulations`
*HTML-based interactive simulators and visualizers.*

* **`BAL_Sobol_Active_Acquisition_Interactive_Simulator.html`**
  * *Preview:* Interactive web simulator visualizing Bayesian Active Learning and Sobol sequence acquisition mechanics.
* **`GPR_Surrogate_Model_Interactive_Simulator.html`**
  * *Preview:* Interactive web simulator visualizing Gaussian Process Regression surrogate modeling and local penalization.

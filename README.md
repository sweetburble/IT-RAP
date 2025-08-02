# Abstract
Adversarial perturbations that proactively defend against generative AI technologies, such as Deepfakes, often lose their effectiveness when subjected to common image transformations. 
This is because existing schemes focus on perturbations in the spatial domain (i.e., pixel) while image transformation often targets both the spatial and frequency domains. 
To overcome this, this paper presents Image Transformation-Robust Adversarial Perturbation (IT-RAP), a framework that learns a robust, multi-domain perturbation policy using Deep Reinforcement Learning (DRL).
ITRAP employs DRL to strategically allocate perturbations across a hybrid action space that includes both spatial and frequency domains.
This allows the agent to discover optimal strategies for perturbation and improve robustness against image transformations.
Our comprehensive experiments demonstrate that IT-RAP successfully disrupts deepfakes with an average success rate of 64.62% when targeting various image transformations.



## Adversarial perturbation procedure with and without image transformations
<img width="500" alt="figure2" src="https://github.com/user-attachments/assets/4adb03ba-608b-4c7d-94a0-b3c282d7a0e3" />

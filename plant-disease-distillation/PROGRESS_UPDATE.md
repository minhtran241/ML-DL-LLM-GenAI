# **Progress Update Report**

**Project Title:** _Model Distillation for Plant Disease Classification_  
**Course:** Applied Machine Learning (W25)  
**Student:** Minh Tran, Leo Ho  
**Instructor:** Dr. Denton Bobeldyk

## **1. Progress Made**

So far, significant progress has been achieved on the project:

-   **Dataset Preparation:** The PlantVillage dataset (Version 3) has been successfully downloaded, preprocessed, and split into training, validation, and test sets.
-   **Teacher Model (ResNet152):** The ResNet152 model was successfully finetuned on the dataset using TensorFlow Keras, achieving strong classification performance with over **97% accuracy** on the validation set.
-   **Student Model (MobileNetV2):** MobileNetV2 has been initialized and baseline training was conducted without distillation for comparison.
-   **Distillation Framework:** A custom knowledge distillation pipeline was implemented, where the MobileNetV2 model learns from the softened outputs (soft targets) of the ResNet152 model using a combined loss function (distillation loss + student loss).

## **2. Roadblocks and Solutions**

-   **Issue:** The initial model distillation loss was not decreasing significantly, leading to underwhelming student performance.  
    **Solution:** Adjusted the temperature parameter and loss weighting in the distillation process, which helped improve gradient flow and resulted in better learning for MobileNetV2.

-   **Issue:** High memory usage during ResNet152 training caused instability on local machines.  
    **Solution:** Moved heavy training tasks to cloud-based environments (Free GPU provided by Kaggle) to ensure stable and efficient model training.

## **3. Next Steps**

-   **Fine-tune Student Model:** Further optimize MobileNetV2 with hyperparameter tuning and regularization techniques.
-   **Model Evaluation:** Compare the performance of the distilled student model with the teacher and baseline models using metrics like accuracy, F1-score, and inference speed.
-   **Final Report & Presentation:** Prepare the final documentation, performance analysis, and a short presentation for submission.

---

This project continues to progress toward delivering a lightweight, accurate, and deployable AI solution for plant disease detection.

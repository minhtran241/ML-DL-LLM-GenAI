# **Project Title: Model Distillation for Plant Disease Classification**

## **Project Overview**

In this project, we focus on **model distillation**, an approach where a larger, more complex model is used to train a smaller, more efficient model. Specifically, we distill knowledge from **ResNet152** into **MobileNetV2** to perform plant disease classification using the **PlantVillage dataset**. The goal is to achieve high classification accuracy while reducing computational complexity, making the model more suitable for deployment on mobile and edge devices.

## **Dataset: PlantVillage**

We utilize the **PlantVillage dataset**, which contains over **50,000 images** of **healthy and diseased** plant leaves, labeled accordingly. This dataset plays a crucial role in advancing **AI-driven plant disease detection**, aiding farmers in identifying and mitigating crop diseases efficiently.

### **Why This Matters**

-   **Global Food Security:** With food production needing a **70% increase by 2050**, early disease detection can significantly reduce crop losses.
-   **Smartphone Integration:** By distilling the model, we enable real-time **disease classification on mobile devices**, making AI-driven solutions accessible to farmers worldwide.

## **Methodology**

1. **Base Model (Teacher):** ResNet152, a deep CNN, serves as the **teacher model**, trained on the dataset to achieve high accuracy.
2. **Distillation Process:** Knowledge from ResNet152 is transferred to MobileNetV2 using **soft targets** and **logit-based training**.
3. **Student Model (MobileNetV2):** A lightweight CNN optimized for mobile deployment, trained using **TensorFlow Keras**.

## **Expected Outcome**

-   A **compact yet highly accurate** model capable of classifying plant diseases with efficiency.
-   A **deployable solution** that allows farmers to use smartphone cameras for real-time disease detection.
-   Contribution towards **sustainable agriculture** through AI-driven early disease identification.

## **Tech Stack**

-   **Deep Learning Framework:** TensorFlow Keras
-   **Models Used:** ResNet152 (Teacher), MobileNetV2 (Student)
-   **Dataset:** PlantVillage (Version 3)

This project paves the way for **AI-powered agricultural solutions**, making plant disease detection **faster, more accessible, and cost-effective**.

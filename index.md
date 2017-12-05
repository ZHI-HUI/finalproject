---
title: MCI reversion prediciton with Alzheimer's Disease Neuroimaigng Initiative databse
---

## Contents
{:.no_toc}
*  
{: toc}


## 0. Problem Statement and Motivation: 

In the current project, we plan to build and evaluate classification models to predict reversion from mild cognitive impairment (MCI) to normal cognitive functioning within 2 years using the ADNI (Alzheimer’s Disease Neuroimaging Initiative) database. MCI is defined as a transitional state between normal cognition and dementia, and much of the existing literature has aimed to determine the predictors of conversion from MCI to dementia. However, many MCI patients do not progress to AD and up to 55% revert back to a cognitively normal state. It is therefore important to identify important predictors of a good prognosis among patients with MCI, which may inform intervention targets. 						 							

## 1. Introduction and Description of Data: 

MCI, a high risk condition for dementia, is regarded as a transitional state between cognitive normal	and dementia. Prevalence of MCI reversion to normal has varied widely, ranging from 4 to 55%, and multiple explanations are proposed for this variability, including individuals being misclassified initially, having unstable MCI, and having true improvements in cognitive functioning upon follow-up. Identifying which individuals classified as MCI are more likely to revert could help optimise resource allocation among MCI patients, with those considered least likely to revert receiving greater levels of intervention and follow-up contact. Therefore, the aim of this project is to build predictive model for MCI reversion using a wide range of predictors including socio-demographic, clinical, genetic, imaging characteristics, and biospecimen biomarkers. 

The goal of the Alzheimer’s Disease Neuroimaging Initiative (ADNI) study is to track the progression of the disease over different disease stages. The study has three phases: ADNI 1, ADNI GO, and ADNI 2. A total of 400 MCI patients were enrolled at the baseline of ADNI 1 phase. The definition used by ADNI was based on the following criteria: (1) subjective memory complaint; (2) Mini-Mental State Examination (MMSE) scores of 24-30; (3) global Clinical Dementia Rating = 0.5; (4) abnormal memory function; (5) did not meet AD criteria for deterioration in general cognitive and functional performance. After exclusion, 187 MCI patients that were followed for two years and did not develop AD within the study period were included for analyses. 
			 							
## 2. Literature Review/Related Work: 

Conversion from MCI to cognitively normal (CN) is commonly observed in the population, and several explanations for this phenomenon have been proposed in the existing literature. It has been reported that patients who revert from MCI to CN, or MCI-reverters, may have a temporary decline in cognitive functioning due to poor mental health (e.g. depression or stress related conditions/disorders) (Kumar, 2006; Olazaran, 2011) or physical health (Wisotsky, 1965). Others have postulated that these unstable MCI states are mis-labeled as MCI, and are in fact pre-MCI conditions that only develop to stable MCI under certain conditions or exposures (Petersen, 2011). Furthermore, it is possible that there are true causes of MCI that are reversible, such as metabolic disorders or acute conditions (e.g. brain injury, substance abuse) that may improve over time, leading to subsequent reversion back to CN. Lastly, pharmacological interventions, lifestyle factors, including diet and physical activity, and baseline health may be true predictors of a better MCI prognosis (Olazaran, 2011). 

While it is of great importance to understand which individuals with MCI have a favorable prognosis, the literature in this area is sparse. Some studies have reported that cognitive factors and other measures used to diagnose MCI are significantly associated with MCI reversion (Manly, 2008; Diniz, 2009; Loewenstein, 2009; Ganguli, 2011; Olazaran, 2011; Dolcos, 2012;  Koepsell, 2012; Park 2015). Of the few studies that examined non-diagnostic factors (e.g. sociodemographic characteristics; physical and mental health), some found that age, educational attainment, arthritis, mental activity, better vision and smelling ability, and blood pressure drop was associated with MCI reversion (Tokuchi, 2014; Sachdev, 2013), but others did not find any associations (Koepsell, 2012). To the best of our knowledge, no study has set out to develop a prediction algorithm for MCI reversion to CN. Identifying individuals with diagnosed MCI, who have a high likelihood of reverting back to normal cognition can: (1) avoid unnecessary antidementia treatments, where the individual is exposed to potential side effects and health risks; (2) prevent the enrollment of clinical trial subjects who may only be in a temporary phase of cognitive decline, and do not in fact have the target disease (Koepsell, 2012). 
			
After a thorough literature review, potential diagnostic and non-diagnostic predictors of MCI reversion to CN were chosen a priori for preliminary EDA and prediction modeling for MCI reversion at two years from baseline. These included the following: 
Demographic characteristics: age, gender, ethnicity, race, education, and marital status;
Clinical factors: baseline blood glucose level, and baseline homocysteine level;
Lifestyle factors: baseline smoking, baseline alcohol abuse;
Neurocognitive/neuropsychological assessments: baseline MMSE score, Auditory Verbal Learning Test (AVLT) scores, Alzheimer’s disease assessment scale-cognitive 11-item/13-item progression model,Trail Making Test A/B, Animal Fluency Test, Clinical Dementia Rating score, Functional Activities Questionnaire (FAQ) (Ewer, 2012; Sachdev, 2013; ; Park, 2015);
Neuroimaging biomarkers: baseline FDG-PET (Park, 2015);
Cerebrospinal Fluid (CSF) Biomarkers: baseline Aβ42, total tau, phosphorylated tau and ApoE 4 genotype (Sachdev, 2013; Thomas, 2017; Park, 2015)
				 							
## 3. Modeling Approach and Project Trajectory: 

Data were split into 75% training set (N=130) and 25% test set (N=57). Classification models were built with the training set and compared with the baseline model of all 0’s classifier. MCI conversion to CN is a dichotomous response variable (0=stable MCI; 1=MCI-conversion to CN), and therefore only classification models that are appropriate for binary outcome were considered, including logistic regression, linear discriminant analysis (LDA), KNN classifier, decision tree, random forest, and boosting classifier. Normalization was performed on the continuous predictors, and the optimal parameters for each of the classification models fitted were chosen using a 5-fold cross-validation procedure. The proportion of MCI reversions in our sample was only 6.95%. Therefore, considering the unbalanced dataset, true positive rate (TPR), false positive rate (FPR), and area under the curve (AUC) were calculated on the test set to assess the model performance. 

We had originally proposed to develop a machine learning model for predicting preclinical Alzheimer’s Disease. However, given the large number of post-baseline measures (including biomarkers and neuropsychological measures) required for the construction of the outcome definition, the sample that remained was too small for meaningful analysis. We have therefore changed our outcome of interest, and our current study aims to develop machine learning solutions for predicting reversion from mild cognitive impairment (MCI) to normal cognitive functioning within 2 years.
					 				
## 4. Results, Conclusions, and Future Work: 

Preliminary EDA was conducted on pre-selected predictors. For continuous predictors, the probability of MCI reversing to normal was positively associated with homocysteine levels, baseline AVLT score, baseline animal fluency score, baseline beta-amyloid levels, baseline tau levels, and baseline phospho-tau levels, and negatively associated with age and baseline ADAS-cog/13 score. For binary/categorical predictors, females had a lower probability of MCI reversion as compared to males; widowed people had a lowest probability of MCI reversion, followed by married and divorced people. Increased copy of APOE4 allele was associated with lower probability of MCI reversion.

Among the six classification models, the AdaBoost classifier with optimal maximum depth of 2 and optimal number of trees of 16 has the highest TPR of 0.6 and the highest test AUC of 0.88. Therefore, the AdaBoost classifier was selected as our final model.

---
type: "project" # DON'T TOUCH THIS ! :)
date: "2023-06-09" # Date you first upload your project.
# Title of your project (we like creative title)
title: ""

# List the names of the collaborators within the [ ]. If alone, simple put your name within []
names: [Dylan Sutterlin]

# Your project GitHub repository URL
github_repo: https://github.com/brainhack-school2023/sutterlin_project

# If you are working on a project that has website, indicate the full url including "https://" below or leave it empty.
website: 

# List +- 4 keywords that best describe your project within []. Note that the project summary also involves a number of key words. Those are listed on top of the [github repository](https://github.com/PSY6983-2021/project_template), click `manage topics`.
# Please only lowercase letters
tags: [ASL-fMRI, hypnosis, connectome, ML]

# Summarize your project in < ~75 words. This description will appear at the top of your page and on the list page with other projects..

summary: "This project aims to better understand neural correlates of hypnosis, which is defined as an experiential experience of focused attention and heighten reponse to suggestions. Hypnotic experience can be assessed in part with the automaticity associated with hypnotic experience, with hypnotic depth and with hypnotizalibility scores. The brain's functionnal connectivity might reflect which brain regions interact to produce the subjective change in phenomenological experience associated with hypnosis."

# If you want to add a cover image (listpage and image in the right), add it to your directory and indicate the name
# below with the extension.
image: "brain.png"
---
<!-- This is an html comment and this won't appear in the rendered page. You are now editing the "content" area, the core of your description. Everything that you can do in markdown is allowed below. We added a couple of comments to guide your through documenting your progress. -->


# 




<div style="text-align: center;">
   <img src="img/tulpa.jpg" height="160px;" alt=""/>
   <img src="img/deities.jpg" height="160px;" alt=""/>
   <img src="img/brain.png" height="160px;" alt=""/>
</div>

## Background

<iframe width="560" height="315" src="https://www.youtube.com/watch?v=f3nf6NOPyr8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Data



<div style="text-align: center; background-color: white; border: 1px solid #000; padding: 20px;">
   <img src="img/sc_protocol.png" height="250px;" alt=""/>
   <img src="img/MRI.png" height="250px;" alt=""/>
   <br /><sub><b>protocol of the sentence completion task</b></sub>
</div>




```{warning}
T
```

All the code used for the project can be found here (https://github.com/brainhack-school2023/jonas_project.git).

# Results


## Progress overview

//The [brainhack school](https://school-brainhack.github.io/) provided four weeks of space during which I could work full time on the analysis of this Tulpa dataset. At the end of these four weeks I was able to complete a first draft of all major analyses intended. These analyses include: //
* GLM (first and second level).
* Task related connectivity.
* ML classifier to differentiate conditions using the connectome;
* A deep learning decoding appraoch using PyTorch to distinguish the task conditions. 

## Tools used

This project was intended to upskill in the use of the following
 * `Nilearn` to analyse fMRI data in python.
 * `Scikit-learn`to realise ML classification tasks on fMRI related measures.
 * `Jupyter {book}`and `Github pages`to present academic work online.
 * `Markdown`, `testing`, `continuous integration` and `Github` as good open science coding practices.



### Overview

### // Deliverables // 

## <em> Timeseries extraction</em>

## <em>_Atlas choices and comparison_</em>
### Yeo et al. 7 networks
<div style="text-align: center; background-color: White; border: 5px solid #000; padding: 0px;">
   <img src="images\yeo7_atals.png" height="200px;" alt=""/>
</div>

* Atlas from [Yeo et al, 2011](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3181453/)
   
* Bilateral mask 
       
      from nilearn import datasets
      atlas = datasets.fetch_atlas_yeo_2011()["thick_7"]

### *Sparse inverse covariance (precision)*

* Citation from [6.2.1 Nilearn](https://nilearn.github.io/stable/connectivity/connectome_extraction.html ) :
>"As shown in [Smith 2011], [Varoquaux 2010], it is more interesting to use the inverse covariance matrix, ie the precision matrix. __It gives only direct connections between regions, as it contains partial covariances__, which are covariances between two regions conditioned on all the others[...]To recover well the interaction structure, a sparse inverse covariance estimator is necessary."


### *__Pre-Hyp__ condition Comparison of covariance / precision / correlation*
<div style="text-align: center; background-color: White; border: 1px solid #000; padding: 0px;">
   <img src="images\precon_covar_yeo7.png" height="200px;" alt=""/>
   <img src="images\precon_preci_yeo7.png" height="200px;" alt=""/>
   <img src="images\precon_correlation_yeo7.png" height="200px;" alt=""/>
</div>

### *__Post-Hyp__ condition comparison of covariance / precision / correlation*
<div style="text-align: center; background-color: White; border: 5px solid #000; padding: 0px;">
   <img src="images\postcon_covar_yeo7.png" height="200px;" alt=""/>
   <img src="images\postcon_preci_yeo7.png" height="200px;" alt=""/>
   <img src="images\postcon_correlation_yeo7.png" height="200px;" alt=""/>
</div>

### __Connectivity matrices from Post-pre hypnosis__
<div style="text-align: center; background-color: White; border: 5px solid #000; padding: 0px;">
   <img src="images\premat_yeo7.png" height="150px;" alt=""/>
   <img src="images\postmat_yeo7.png" height="150px;" alt=""/>
   <img src="images\contrastmat_yeo7.png" height="150px;" alt=""/>
   <img src="images\contrastcon_yeo7.png" height="200px;" alt=""/>
</div>

## DifuMo
   *Fine-grain atlas of functional modes for fMRI analysis*

* article from [Dadi et al, 2020](https://www.sciencedirect.com/science/article/pii/S1053811920306121)
* Can be downloaded [here](https://parietal-inria.github.io/DiFuMo/64)
* **Labels provided**

<div style="text-align: center; background-color: White; border: 5px solid #000; padding: 0px;">
   <img src="images\scr_difumo_comp1.png" height="240px;" alt=""/>
   <img src="images\scr_difumo_comp2.png" height="230px;" alt=""/>
</div>

<div style="text-align: center; background-color: White; border: 5px solid #000; padding: 0px;">
   <img src="images\contrastmat_diffumo.png" height="240px;" alt=""/>
   <img src="images\contrastcon_diffumo.png" height="230px;" alt=""/>
</div>
g
## <em>Inter-subject connectivity analysis </em>



```{warning}
Algorithm and covariance estimation choices
-  Covariance estimation
- Prediction algorithms
- Cross validation ?

```
> Plot connectomes

## <em>Hypnosis-related variables' prediction</em>
- Linear assumptions

## <em>Machine learning classification</em>

```{warning}
```
### classification

// A z-score of 3.0 is equivalent to a 99% confidence interval, a z-score of 2.3 is equivalent to a 95% confidence interval. // 

<div style="text-align: center; background-color: white; border: 1px solid #000; padding: 20px;">
   <img src="img/dm_first.png" height="420px;" alt=""/>
   <img src="img/dm_second.png" height="420;" alt=""/>
   <img src="img/SSW-STW_paired_thres.png" width="400px;" alt=""/>
   <img src="img/SSW-STW_unpaired.png" width="400px;" alt=""/>
</div>


> : //information_source: matched pairs are between software, within subject, within contrast, within run.//



<div style="text-align: center; background-color: white; border: 1px solid #000; padding: 20px;">
   <img src="img/spm_vs_nilearn.png" height="420px;" alt=""/>
</div>


<div style="text-align: center">
   <img src="img/connectome.png" width="600px;" alt=""/>
</div>

### Seed-to-Voxel connectivity results



//The aim was to train a classifier that can accurately distinguish between different task conditions. I used a majority vote ensemble classifier that combines `LogisticRegression`, `RandomForestClassifier`, and a `SVC`. When classifying the conditions `prep` vs. `write`, the classifier achievs an acciracy of ~80%, so well above chance. However, this is not surprising as the writing conditions will have muhc stronger motor cortex activation and the two conditions are quite different. I then classified the `self`, `tulpa`, and `friend` conditions for preparation and writing respectively. In both conditions, we have an accuracy of ~51% for this 3-group classification problem. Given the three groups, chance levels are at 33.3%,//

<br/>

> //information_source: all classifiers were trained and tested on the **connectome**. Next, the plan is to train classifiers on the beta maps of the GLM//. 


****

> Average accuracy = 0.76 <br/>
> P-value (on 100 permutations): p=0.00
<div style="text-align: center">
   <img src="img/ml/ml_p-w_conf.png" width="400px;" alt=""/>
   <br/>
   <img src="img/ml/ml_p-w_weights.png" width="400px;" alt=""/>
   <img src="img/ml/ml_p-w_glass.png" width="400px;" alt=""/>
</div>

<br/>

**Write condition**

> Average accuracy = 0.52 <br/>
> P-value (on 100 permutations): p=0.32
<div style="text-align: center">
   <img src="img/ml/ml_w_conf.png" width="400px;" alt=""/>
   <br/>
   <img src="img/ml/ml_w_weights.png" width="400px;" alt=""/>
   <img src="img/ml/ml_w_glass.png" width="400px;" alt=""/>
</div>

<br/>

**Prep condition**

> Average accuracy = 0.37 <br/>
> P-value (on 100 permutations): p=0.00
<div style="text-align: center">
   <img src="img/ml/ml_p_conf.png" width="400px;" alt=""/>
   <br/>
   <img src="img/ml/ml_p_weights.png" width="400px;" alt=""/>
   <img src="img/ml/ml_p_glass.png" width="400px;" alt=""/>
</div>


   - histograms are powerful because they don't collapse either time or space 
   - raindrop plots are powerful as they show all individual points 
   - examples in the [python graph gallery](https://www.python-graph-gallery.com/)
- Python virtual environment
   - To make and actiavte a python virtual environment, use the following two lines:
      - `python3 -m venv venv`
      - `source venv/bin/activate`
   - To generate a `requirements.txt` file
      - Option 1: `pip freeze`.
      - Option 2: [Pipreqs](https://pypi.org/project/pipreqs/) automatically detects what libraries are actually used in a codebase. 
- //Python local pip install 
   - make sure to have an __init__.py file in the src directory 
   - also make sure to have a setup.py file
   - Navigate to root directory and run: `pip install -e .`

## About the Author


<div style="text-align: center;">
<a href="https://github.com/jonasmago">
   <img src= width="150px;" alt=""/>
   <br /><sub><b></b></sub>
</a>
</div>

## Acknowledgement


</div>
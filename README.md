# Castcle's Data Science Description
## 1. Feed Algorithm Overview
Cascle's feed algorithm has tailored from 2 regimes i.e. **App** and **Analytics** which is undergo by some parts of data science processes. The database coloring represents location such as blue color stands for `app-db`, while green color represents `analytics-db`. This file will explain only **Analytics** regime with mild touch the **App**. Features are extracted in **Feature Extractors** i.e. [content_stats_update](https://github.com/castcle/castcle-trigger/blob/main/content_stats_update.py) and [creator_stats_update](https://github.com/castcle/castcle-trigger/blob/main/creator_stats_update.py) to facilitating **Model Training** and **Model Prediction** of such [coldstart_trainer](https://github.com/castcle/castcle-trigger/blob/main/coldstart_trainer.py), [personalize_content_trainer](https://github.com/castcle/castcle-trigger/blob/main/personalize_content_trainer.py) and [coldstart_predictor](https://github.com/castcle/castcle-ds-predict/blob/main/coldstart_predictor.py), [per_ct_predictor](https://github.com/castcle/castcle-ds-predict/blob/main/per_ct_predictor.py), respectively.

Moreover, **Aggregators** are respond to logically filter contents from **Inventory** in different ways. **Ranker** consumes features from **Feature Extractors** , Trained model artifact from **Model Training** and aggregated/filtered contents from **Aggregators** to ranking/scoring tend of engagement for the user using **Model Prediction** then feeds a certain amount to user's UI. More precisely, user interaction or engagement will be recorded to database for purpose of further **Analytics** and model evaluation.

![castcle_ds_overviewdrawio drawio (7)](https://user-images.githubusercontent.com/90676485/147076667-e5de903e-ebde-488a-8cf0-a14ace50a777.png)

## 2. Workflow Process
Data science workflow process of Castcle can be illustrated by the bottom diagram of the below figure exhibits overall workflow process interacts across databases i.e. blue blocks represent collections in `app-db` and green blocks represents collections in `analytics-db` databases in Mongodb Atlas, respectively. The bold arrows stand for presence of entity key relation between collections, the dot arrows reflect data extraction by either aggregation or calculation, and the two-headed arrows represents swap event.

In the other hands, the workflow process can be separated in to 3 steps which are shown as the top orange blocks in below diagram (orange arrows) corresponding in vertical axis to the bottom diagram. However, non-machine learning (ML)-involved processes will only carry 2 steps (purple arrow).

However, steps 1. and 2. are explained in [castcle-trigger](https://github.com/castcle/castcle-trigger/blob/develop/README.md), are located in another repository i.e. [castcle-trigger](https://github.com/castcle/castcle-trigger/tree/main). Hence, this will explain only this repository on **step 3**.

![castcle_ds_er drawio (5)](https://user-images.githubusercontent.com/90676485/145951393-3af8140e-cc63-429e-b034-94b63de75dfb.png)

### 3. Saving Persistent
This step possesses response for both schedule and on demand execution available for guest and registered users, respectively. There are also 2 processes in this step which are coldstart for guest users and personalize content for registered users. Another function to saving persistent is **topic classify** which is described in [castcle-trigger](https://github.com/castcle/castcle-trigger),
  - **coldstart predictor** consumes features from `analytics-db.contentStats` and `analytics-db.creatorStats` and model artifacts from `analytics-db.mlArtifacts_country` to making model prediction every cron(*/30 * * * ? *). The output i.e. top 2000 content IDs and their scores for each country will be inserted into `app-db.guestfeeditemtemp`. Whenever insertion finishs, the whole documents in `app-db.guestfeeditemtemp` will be swapped with `app-db.guestfeeditem` to support eliminate downtime prior to usage as feed items for guest users.
  - **personalize content predictor** takes request message of account ID and list of content IDs as input then consider to loading presence of the correspond artifact in `analytics-db.mlArtifacts`, if the account artifact is absent, the model will employ correspond geolocation artifact in `analytics-db.mlArtifacts_country` instead. In contrast, if the geolocation artifact is also absent, the model will employ default geolocation artifact which is "us". Therefore, the model also take features from both `analytics-db.contentStats` and `analytics-db.creatorStats` then perform prediction resulting in scores for each input contents and sent back the output as message with status code "200"

![castcle_ds_er_persist drawio (1)](https://user-images.githubusercontent.com/90676485/145952267-eb54a6ce-a9ca-48c7-a0e9-630e2d09a1af.png)

## 3. Collections description
In this section we will describe only collections those are interacted as output from data science processes. The data science-related collections can be separated into 2 groups depend on its location.
  1. collections in `analytics-db`
  - `contentStats`: prepares feature for model utilization in part of content and is obtained by extracting data from `app-db.contents`. It updates itself every hour by removing contents which have age over threshold and upsert outputs from **update content stats**.
  - `creatorStats`: prepares feature for model utilization in part of content creator user and is obtained by extracting data from `app-db.contents`. Similar to `contentStats`, it updates itself every hour but keep all content creator user using outputs from **update creator stats**.
  - `mlArtifacts`: collects personalize content model artifacts which are the output from **personalize content trainer** and also collect model version.
  - `mlArtifacts_country`: collects country-based model artifacts which are the output from **coldstart trainer** and also collect model version.
  - `topics`: master collection of topics that is gathered hiearachically; contain children and parents topics (if have) from outputs of **topics classify**. It updates when `app-db.contents` has created or updated and topics can be classified.  
 
  2. collections in `app-db`
  - `contentinfo`: duplicated collection amount with `app-db.contents` and updates when `app-db.contents` has created or updated in the same process from **topics classify**. It collects both language and topics of the contents.
  - `guestfeeditemstemp`: It is output from **coldstart predictor** designed for eliminate downtime which will swap with `guestfeeditems` after successfully when upserted and contains item type.
  - `guestfeeditems`: a collection that exists for utilize as feed to guest users which is swap from `guestfeeditemstemp`. It is output from **coldstart predictor** and also contains item type.

## 4. Repositiory Description
There are 2 repositories those involved with data science process i.e. [castcle-trigger](https://github.com/castcle/castcle-trigger) and [castcle-ds-predict](https://github.com/castcle/castcle-ds-predict) which have similar structure but different functionality. More precisely, one is response for part 1. and 2., while another is response for part 3. of workflow process, respectively. 

In this section we will describe only collections those are interacted as output from data science processes which are located in this repository. For another data science-related collections [click here](https://github.com/castcle/castcle-trigger).
 1. [requirements.txt](https://github.com/castcle/castcle-ds-predict/blob/develop/requirements.txt): contains necessary libraries.
 2. [serverless.yml](https://github.com/castcle/castcle-ds-predict/blob/develop/serverless.yml): contains configuration.
 3. python caller files (.py): responses for calling main function in [modules](https://github.com/castcle/castcle-ds-predict/tree/develop/modules),
  - [x] [coldstart_predictor.py](https://github.com/castcle/castcle-ds-predict/blob/main/coldstart_predictor.py): responses for calling to execute [coldstart_predictor.py](https://github.com/castcle/castcle-ds-predict/blob/main/modules/coldstart/coldstart_predictor.py) to update `app-db.guestfeeditemstemp` and `app-db.guestfeeditems`.
  - [x] [per_ct_predictor.py](https://github.com/castcle/castcle-ds-predict/blob/main/per_ct_predictor.py): responses for calling to execute [personalize_content_predictor.py](https://github.com/castcle/castcle-ds-predict/blob/main/modules/personalized_content/personalize_content_predictor.py). In contrast to `coldstart_predictor.py`, the output is returned as message response.
  4. [modules](https://github.com/castcle/castcle-ds-predict/tree/develop/modules) python files (.py): contain main function files which are located inside sub-folders for operate with correspond python caller file.

## 5. Model Explanation: Cold-Start
This model will be used to ranking/scoring within threshold contents based on engagement behavior in each specified country. The model will be re-trained everyday then stored in `analytics-db.mlArtifacts_country` collection. These models support users those do not have their own personalized model and can be used to give a wider range of content recommendation.
  1. Model inputs
  - Country engagement 
  - Content features 
  - Country code (250 countries, ISO3166)

  2. Model detail
  - Model: XGBOOST Regression Model
  - Target variable: Weight engagement 
  - Time using: 1 mins (12/13/2021)
  - Countries that do not have training data will adopt "us" model

  3. Output
  - Collection contains "countryCode", model artifacts, and timestamp

  4. Model workflow
  This file explain only model prediction section a.k.a. scoring section. If you would like to see another section, [click here](https://github.com/castcle/castcle-trigger/blob/develop/README.md)
   4.1. Content features preparation
     
     1. Engagement List
     - Like
     - Comment 
     - Quote
     - Recast
     
     2. Aggregation : Sum
     
     3. Group By: "countryCode" (ISO3166), "contentId"   
  
  4.2. Load model artifacts based on country 
  
  4.3 Save top 2000 contents
    Output list
    
    - "content"
    - "score"
    - "countryCode"
    - "type"
    - "updatedAt"
    - "createdAt"
     
![Cold-start (1)](https://user-images.githubusercontent.com/90676485/147532472-2da1afb3-267c-44d4-8dde-7d3f05160f33.jpg)

## 6. Model Explanation: Personalized Content Model
This model will be used to rank requested contents based on user’s engagement behaviors. The model will be re-trained everyday in the morning and stored in the mlArtifact collection in db_analytics. The model is for users that have their own personalized model meaning that they have at least one engagement history and can be used to give a wider range of content recommendation combined with cold start model.

This model will be used to ranking/scoring the requested contents based on user’s engagement behaviors. The model will be re-trained everyday then stored in `analytics-db.mlArtifacts`. These models support users that have their own personalized model meaning that they have at least one engagement history and can be used to give a wider range of content recommendation combined with cold start model.
 1. Model inputs
 - User engagement 
 - Content features 

 2. Model detail 
 - Model Used: XGBOOST Regression Model
 - Target variable : Weight engagement 
 - Time using : 1 mins (12/13/2021)

 3. Output
 - Collection contains "userId", model artifacts, and timestamp 

 4. Model workflow (Scoring Section)
  
  4.1 Content features preparation
    
    1. Content Feature List
    - likeCount : Total like of each content based on subject
    - commentCount : Total comment of each content based on subject 	
    - recastCount : Total recast of each content based on subject 	
    - quoteCount : Total quote of each content based on subject 
    - photoCount : Total photo of each content	
    - characterLength : Number of charecter	
    - creatorContentCount : Total content of the creator of this content
    - creatorLikedCount : Total like of the creator of this content 
    - creatorCommentedCount : Total comment of the creator of this content 
    - creatorRecastedCount : Total recast of the creator of this content 
    - creatorQuotedCount : Total quote of the creator of this content
    - ageScore : age score of this content
    2. Aggregation : Sum, Count
  
  4.2.Load ML artifacts based on userId
  
  4.3.Scoring given contents based on user’s model: Send content score to the Castcle app. 

![Personalized_content](https://user-images.githubusercontent.com/90676485/147532468-b3a68d3e-4228-4412-b097-d1ef41862693.jpg)

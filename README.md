# Exam readme

## Workload

### System definition and design
- idea for interface
- time schedule
- defining simplest working system
- defining complete system
- target audience

### Data-scraping pipeline (Omar)
- download list of interesting Wikipedia pages 
- method to download wikipedia page 
- method to convert page to BoW 
- create wiki BoW dataset 
- method to update BoW

### User data (Stefano)
- Method to create new user
- Method to add user preferences and create sample user dataset
- Method to make recommendation
- Interface to database

### Model (Stefano)
- ~Create a SVM on sample user data~
  - Try with other architectures
  - Create a Logistic Regression for each user 
- Make recommendation system 

### User Interface (Andrea, Alessio)
- basic browser UI
- Streamlit
- ...

### Maintainance 
- updating models
- updating keywords, wiki database and user databases
- updating recommendation algorithm


## Milestones

- Minimal working product (single user, few pages, few keywords)
- Additions: many keywords, many pages
- Multiple users
- Final product testing


## Notes

- BoW keywords are the same for all user, they only change during an update
- one model for each user
- the models are trained (mostly) on the user's dataset
- during recommendation, the model is called on a subset of the Wiki dataset, and the highest scoring page gets recommended
- deploy a week before the exam, then prof shares exam agenda


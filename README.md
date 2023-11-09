# Exam readme


## Workload

### System definition and design
- defining simplest working system
- defining complete system
- target audience

### Data-scraping pipeline (Omar)
- download list of interesting Wikipedia pages 
- method to download wikipedia page 
- method to convert page to BoW 
- create wiki BoW dataset 
- method to update BoW

### User data (Omar, Stefano)
- method to create new user
- method to add user preferences and create sample user dataset
- method to make recommendation 

### Model (Stefano, Omar)
- create a SVM on sample user data (Omar, Stefano)
- make recommendation system 
- try with other architectures

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

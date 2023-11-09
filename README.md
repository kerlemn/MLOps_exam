# Exam readme

## Workload

Data-scraping pipeline
- download list of interesting Wikipedia pages
- method to download wikipedia page
- method to convert page to BoW
- create wiki BoW dataset
- method to update BoW

User data
- method to create new user
- method to add user preferences and create sample user dataset
- method to make recommendation 

Model 
- create a SVM on sample user data
- make recommendation system

User Interface
- basic page
- ...


## Milestones

- Minimal working product(single user, few pages, few keywords)
- Many keywords, many pages
- Multiple users
- Final product testing


## Notes

- BoW keywords are the same for all user, they only change during an update
- one model for each user
- the models are trained (mostly) on the user's dataset
- during recommendation, the model is called on a subset of the Wiki dataset, and the highest scoring page gets recommended

# Exam readme

## Project description

### 1. Problem Definition
The goal of the project is to provide an interface where the user can discover new and interesting Wikipedia pages to read

### 2. Clear definition of a software development method and software process model
back-end...
front-end...

### 3. Identification of Software Requirements, Software Development Methods and Software Test
...

### 4. Dataset Information
The datasets consist of: 
- a list of bag-of-words that contain information about the content of that page
- the target feature is a binary value that encodes user's feedback (1=like, 0=dislike)

### 5. Proposed Pipeline 
1. download data from wiki
2. convert to BoW
3. generate initial user dataset to train first model on
4. content recommandation system
5. collect user's feedback
6. fine-tune model for the user
...

### 6. Software organization 
...

### 7. Software functionalities
The UI provides the user with an experiance of endless article-form content, fine-tuned to their preferences...

### 8. Proposal limitations
...


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
- deploy a week before the exam, then prof shares exam agenda


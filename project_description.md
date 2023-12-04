# 1. Problem Definition
The goal of the project is to provide an interface where the user can discover new and interesting Wikipedia pages to read

# 2. Clear definition of a software development method and software process model
The project will be developed with in Incremental build model; the product is designed, implemented and tested incrementally, adding requirements and refinements until the product meets the expectations.

# 3. Identification of Software Requirements, Software Development Methods and Software Test
## Requirements
Since the project follows an incremental approach, the requirements will be grouped in the version in which they were implemented.
### V1.0
- The service will be composed by a web interface and an API that will implement the project logic
- The interface will request Wikipedia pages (one at the time) from the API and sho them to the user
- The API chooses the page to send to the interface by piking the most similar (from a bunch of randomly picked Wikipedia pages) to a static set of pages marked as "interesting"

# 4. Dataset Information
The datasets consist of: 
- a list of bag-of-words that contain information about the content of that page
- the target feature is a binary value that encodes user's feedback (1=like, 0=dislike)

# 5. Proposed Pipeline 
1. download data from wiki
2. convert to BoW
3. generate initial user dataset to train first model on
4. content recommandation system
5. collect user's feedback
6. fine-tune model for the user
...

# 6. Software organization 
...

# 7. Software functionalities
The UI provides the user with an experiance of endless article-form content, fine-tuned to their preferences...

# 8. Proposal limitations
...
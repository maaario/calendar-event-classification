### Ideas
Event: summary, start time, end time, is_repeated, calendar

Create features from datetime: year, month, date, WEEKDAY, start(mins), duration(mins)

- dissimilarity between times 23:59 and 00:00 / days 7 and 1 ?

Adding more events

- Convert repeated events to normal events
- Some events are an aggregate of multiple tasks: use comma to split them to seaprate activities?

For prediction of the next event

- events may overlap: have to make a consensus of what I was doing at a specified time
- select only relevant days (covered by at least 75%? ... stg, make statististics first)

multiple categories? social+food?
hierarchical clustering?
looking at sequences of events?
add month instead of day of the year attribute?

### Create dataset
the result should be a matrix that can be loaded to r / numpy / pandas

### First iteration
1) create some labels (calendar names)
2) PCA only based on metadata (time, duration, )
3) k-means
Results: sleep can be easily distinguisthed, other not so simply 

### Second iteration
0) Start in jupyter notebook, quicklearn pandas (review ML exercises) and load data
1) Repeat PCA, maybe a better visualization method?
2) Train classifier on metadata -> calendar

Results: (day number, weekday, start time of day, duration -> which calendar)

- 80% - 3 categories: 
- 96.5% - 2 categories: distinguishing sleep from school+personal
- 81% - 2 categories: distinguishing school from personal
- What if we don't provide day number?
    - The accuracy doesn't change: good not overfitting on days
- What if we don't provide weekday?
    - even higher accuracy: 82% - only based on duration and time of day

### Third iteration
1) Create advanced categories from summaries 

  - keyword -> category (for most frequent words)
  - what about conflicts?
    - print words that fall into multiple categories and update the rules
    - choose only events that fall into one category

2) Run PCA, SVM, RF on metadata predicting category 

 - 71%  - 3 categories: sleep, school, other
 - 57%  - 6 categories 
 - If we don't provide day number, accuracy drops to 54%
 - If we add month, the accuracy does not increase.
 - Normalization doesn't help for SVM or RF
 - As school is the main source of bias, training on 5 labels (no school data) gets 62%
 - However if we also remove sleep, the other 4 categories achieve only 55%

### Fourth iteration
1) create word vectors from summaries:  
 - find slovak stemmer
 - preprocessing + bag of words + PCA (future: autoencoder?)
 - use n-grams?
 - word2vec?    
2) visualize word vectors in 2D  
3) Repeat PCA and KMeans with word vectors available

### ANALYSIS
 - time spent on different categories
 - punchcards of different categories (which times of day?)
 - most and least productive days?

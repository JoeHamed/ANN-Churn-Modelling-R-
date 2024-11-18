# Artificial Neural Networks (ANN)

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[, 4:ncol(dataset)]

# Encoding categorical Variables
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('No', 'Yes'),
                                   labels = c(0, 1)))

# Splitting data into training and testing sets
#install.packages('caTools')
library(caTools)
set.seed(42) #Random_state
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-ncol(dataset)] = scale(training_set[-ncol(dataset)]) # -11
test_set[-ncol(dataset)] = scale(test_set[-ncol(dataset)])

# Fitting ANN to the training set
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1) # take all the available cores
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2) # batch_size (Automatic)

# Predicting the test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-ncol(dataset)])) # -11
# y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the confusion matrix
cm = table(test_set[, ncol(dataset)], y_pred > 0.5)

# Accuracy
# (1544 + 173)/2000
# 0.8585

h2o.shutdown()
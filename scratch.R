
library(caret)
local_data_training <- read.csv('pml-training.csv')
local_data_testing <- read.csv('pml-testing.csv')

set.seed(12345)
ts_in <- createDataPartition(y=local_data_training$classe, p=0.7, list=F)
ts_in1 <- local_data_training[ts_in, ]
ts_in2 <- local_data_training[-ts_in, ]


# nzv (nearly zero)
nzv <- nearZeroVar(ts_in1)
ts_in1 <- ts_in1[, -nzv]
ts_in2 <- ts_in2[, -nzv]

# remove variables that are NA
l_NA <- sapply(ts_in1, function(x) mean(is.na(x))) > 0.96
ts_in1 <- ts_in1[, l_NA==F]
ts_in2 <- ts_in2[, l_NA==F]

# remove variables tthe first five variables (not needed for prediction)
ts_in1 <- ts_in1[, -(1:5)]
ts_in2 <- ts_in2[, -(1:5)]

# 3 fold cross validation
fittedData <- trainControl(method="cv", number=3, verboseIter=F)

# fit model
fit <- train(classe ~ ., data=ts_in1, method="rf", trControl=fittedData)

print (fit$finalModel)

# use model to predict classe in validation set (ts_in2)
preds <- predict(fit, newdata=ts_in2)

# show confusion matrix to get estimate of out-of-sample error
confusionMatrix(ts_in2$classe, preds)

# predict on test set
preds <- predict(fit, newdata=local_data_testing)

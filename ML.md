---
title: "Machine Learning Project: Prediction of Performance of Exercise "
date: "January 28, 2016"
output: html_document
---

Background
-------------------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. We create model to predict the quality of the exercise. The data for this project come from this source:   <a href="http://groupware.les.inf.puc-rio.br/har">http://groupware.les.inf.puc-rio.br/har</a>.

### Load package

```{r,message=FALSE}
library(caret)
library(rpart)
library(e1071)
library(gbm)
library(plyr)
```

### Download Data

#### 1. Download training data set from  <a href="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv">https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv</a>.

#### 2. Download test data set from <a href="http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv">http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv</a>.

In the raw data, there are values like NA, blanks and #DIV/0!. We treat them all as NAs when importing data.

```{r}
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
```


### Clean Data

#### 1. Remove Columns with Large Proportion of NAs.

```{r}
dim(training)
dim(testing)
```

The raw training data has 19622 rows and 160 columns. The raw testing data has 20 rows and 160 columns. 
   
```{r}
na_count<-sapply(training, function(x) sum(length(which(is.na(x)))))
hist(na_count,main='Total of NA in Each Column')
```

From the histogram of percentage of NAs in each column, we can see 100 of 160 the columns has high frequency of NAs. We keep columns with NA's count smaller than 5000 out of the total 19622 row in the training data set.

```{r}
rmnatraining<-training[,which(na_count<5000)]
dim(rmnatraining)
```

#### 2. Remove Non-movement Variables

Based on the analysis requirements, we remove non-movement variables: x, user_name,  raw_timestamp_part_1,  raw_timestamp_part_2, cvtd_timestamp, new_window  and num_window, which are column 1 to column 7.

```{r}
trainingdf<-rmnatraining[,-c(1:7)]
dim(trainingdf)
```

#### 3. Remove Near Zero Variance Variables
Let's check the variance of the predictors with nearZeroVar function in R. 

```{r}
nvrmx<-nearZeroVar(trainingdf,saveMetrics=TRUE)
nvrmx$zeroVar==TRUE
nvrmx$nzv==TRUE
```

It seems none variables are constant or almost constant predictors across samples. We don't remove any columns in this step.

#### 4. Remove Highly Related Variables
Let's identify related predictors with correlation value greater than 0.75 and revome these variables.

```{r}
CorTraining<-cor(trainingdf[,-53])
summary(CorTraining[upper.tri(CorTraining)])
highcor<-findCorrelation(CorTraining,cutoff=.75)
FilteredTraining<-trainingdf[,-highcor]
CorFilTraining<-cor(FilteredTraining[,-32])
summary(CorFilTraining[upper.tri(CorFilTraining)])
```

Now we have 32 columns for the training data set. We use the same variables for testing data set for prediction.

```{r}
testingcolnames<-colnames(FilteredTraining[,-32])
FilteredTesting<-testing[,testingcolnames]
dim(FilteredTesting)
```

#### 5. Split Data for Cross Validation
We split the filtered training data set into two parts: one for creating model (70%) and the other for cross validation (30%).

```{r}
inTrain=createDataPartition(y=FilteredTraining$classe,p=0.7,list=FALSE)
mytraining=FilteredTraining[inTrain,]
mytesting=FilteredTraining[-inTrain,]
```
   
### Analyze and Predict Data
Since we need to classify the performance of the exercise, we use random forests and boosting with trees methods to analyze and predict data.

#### 1. Random Forests Modeling

```{r,message=FALSE}
modfitrm<-train(classe~.,method="rf",data=mytraining) 
predictrf<-predict(modfitrm,mytesting)
cmrf<-confusionMatrix(predictrf,mytesting$classe) 
prerm<-predict(modfitrm,testing)

```

```{r}
cmrf
prerm
```

#### 2. Boosting with Trees Modeling

```{r,message=FALSE}
modfitgbm<-train(classe~.,method="gbm",data=mytraining,verbose = FALSE) # Boosting with Trees Modeling
predictgbm<-predict(modfitgbm,mytesting) 
cmgbm<-confusionMatrix(predictgbm,mytesting$classe) 
pregbm<-predict(modfitrm,testing)
```

```{r}
cmgbm
pregbm
```

We can see accuracy values for both models are 0.9932 and 0.947. When it comes to the prediction, both predict the testing data set with the same results. So we will use the random forests model since it has a higher accuracy value and reliable prediction. The expected out-of-sample error is 1-0.9932=0.0068=0.68%.


Conclusion
-------------------
We apply the random forests model to predict the testing data. The 20 predictions of the class for the testing data set are: B A B A A E D B A A B C B A E E A B B B.


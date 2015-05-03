bank<-read.csv("bank-additional-full.csv",header=T,sep=";")
which(bank=="unknown")
# Since the unknown value are 10000, we can't delete ten thousand observation,
# so we deal with unknown value as a possible class label



# we try to build a model with dividing training data and test data set.
set.seed(123456789)
random<-sample(1:nrow(bank))
num.bank.training<-as.integer(0.75*length(random))
bank.indices<-random[1:num.bank.training]
train<-bank[bank.indices,]
testing.indices<-random[(num.bank.training+1):length(random)]
testing.set<-bank[testing.indices,]

# I divide 2/3 of whole bank data set as training data, and rest 1/3 of bank data set
# as my testing data.

logis<-glm(y ~ ., data=train,family=binomial)
summary(logis)
plot(logis)
plot(logis,which=6)


#Boxplot for dependent variable -> however, since y is categorical variable, no need to transformed
boxTidwell(y~age+duration+emp.var.rate+pdays+cons.price.idx +euribor3m+nr.employed,other.x= ~job+marital+education+default+housing+loan+contact+month
           +day_of_week+campaign+poutcome,data=train)


boxplot(train[,1:20])

bank1<-train
bank1$dffits<-0
bank1$dffits<-dffits(logis)
bank2<-bank1[!bank1$dffits>2*sqrt(21/41188),]
# the data "bank2" is dataset without outliers.
# remove column dffit
bank1$dffits<-NULL
bank2$dffits<-NULL
bank1$out<-NULL
bank2$out<-NULL



# now let's make another logistic regression

logit<-glm(y~.,data=bank2,family=binomial)
summary(logit)
plot(logit)

plot(logit,which=4)
outlierTest(logit)

# we still have outliers
# at least we remove the observation 5763,6671,20075,25236
# to remove influential outliers.
bank3<-bank2[-c(5763,6671,20075,25236,24902,36044,24092,36044,19633),]
logit.2<-glm(y~.,data=bank3,family=binomial)
summary(logit.2)
plot(logit.2,which=6)
plot(logit.2)

##############################Check multicollinearity and variable selection############################################
vif(logit.2)
# having error : there are aliaed coefficient in the model.
library(stats)
alias(logit.2)
# Seems there are perfect collinearity, use variable selection.
stepAIC(logit.2,k=2)
# From AIC test, the best model would be glm(formula = y ~ contact + month + day_of_week + duration + 
#pdays + poutcome + emp.var.rate + cons.price.idx + cons.conf.idx, 
#family = binomial, data = bank3)

logit.aic<-glm(y ~ contact + month + day_of_week + duration + pdays + poutcome + emp.var.rate + cons.price.idx + cons.conf.idx, 
               family = binomial, data = bank3)
summary(logit.aic)

plot(logit.aic)

#BIC
stepAIC(logit.2,k=log(length(bank3[,1])))
#From BIC test the best model would be 
# y ~ contact + duration + poutcome + cons.price.idx + 
#cons.conf.idx + euribor3m, family = binomial, data = bank3)

logit.bic<-glm( y ~ contact + duration + poutcome + cons.price.idx + 
                  cons.conf.idx + euribor3m, family = binomial, data = bank3)


summary(logit.bic)
plot(logit.bic)
# all predictors are significant, and it is simpler model compared to AIC model.
# Therefore, I select my best logistic model as logit.bic model

prediction <- data.frame(predict(logit.bic,bank3,type="response"))
prediction[prediction<0.5]=0
prediction[prediction>=0.5]=1
predictions <- data.frame(Prediction = as.numeric(prediction[,1]),Actual = as.numeric(bank3$y)-1)
predictions$Correct <- (predictions$Actual == predictions$Prediction)
logistic_accuracy<-table(predictions$Correct)/length(predictions$Correct)*100
# The accuracy is 99.15% which is really high.

prediction.test<-data.frame(predict(logit.bic,testing.set,type="response"))
prediction.test[prediction.test<0.5]=0
prediction.test[prediction.test>=0.5]=1
predictions.test <- data.frame(Prediction = as.numeric(prediction.test[,1]),Actual = as.numeric(testing.set$y)-1)
predictions.test$Correct <- (predictions.test$Actual == predictions.test$Prediction)
logistic_accuracy.test<-table(predictions.test$Correct)/length(predictions.test$Correct)*100

# As we see the result above, the accuracy is 89.16189 which is less than prediction of 
# training.set.

################################################################################
################################################################################
################### Next one would be decision tree model#######################
################################################################################
################################################################################

library(ElemStatLearn)
library(tree)
require(rpart)
library(rpart)
tree <- rpart(y~contact + duration + poutcome + cons.price.idx + 
                cons.conf.idx + euribor3m, data=bank3, method="class")
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(tree,main = "", sub = "",cex=0.5)
printcp(tree)
plotcp(tree) 
# visualize cross-validation results
# With the plot of CP, I select threshold cp as 0.0195
# Next, I prune my tree model to avoid over-fitting
tree.prune = prune(tree, cp = 0.0195)
fancyRpartPlot(tree.prune,main = "", sub = "",cex=0.5)

####### Make prediction#################
prediction.tree <- data.frame(predict(tree.prune, bank3, type = "class"))
predictions.tree <- data.frame(Prediction = as.numeric(prediction.tree[,1])-1,Actual = as.numeric(bank3$y)-1)
predictions.tree$Correct <- (predictions.tree$Actual == predictions.tree$Prediction)
Tree_Accuracy <- table(predictions.tree$Correct)/length(predictions.tree$Correct)*100
Tree_Accuracy

####### predict with test############
prediction.tree.test<-data.frame(predict(tree.prune,testing.set,type="class"))
predictions.tree.t <- data.frame(Prediction = as.numeric(prediction.tree.test[,1])-1,Actual = as.numeric(testing.set$y)-1)
predictions.tree.t$Correct <- (predictions.tree.t$Actual == predictions.tree.t$Prediction)
Tree_Accuracy.t <- table(predictions.tree.t$Correct)/length(predictions.tree.t$Correct)*100
Tree_Accuracy.t


################################################################################
################################################################################
################### Random Forrest##############################################
################################################################################
################################################################################

library(randomForest)
forest<-randomForest(as.factor(y)~contact + duration + poutcome + cons.price.idx + 
                       cons.conf.idx + euribor3m,data=bank2, importance=TRUE, ntree=100)
#nstead of specifying method="class" as with rpart, we force the model
#to predict our classification by temporarily changing our target 
#variable to a factor with only two levels using as.factor(). The 
#importance=TRUE argument allows us to inspect variable importance 
#as we'll see, and the ntree argument specifies how many trees we want to grow.
#If you were working with a larger dataset you may want to reduce 
#the number of trees, at least for initial exploration, or restrict the complexity 
#of each tree using nodesize as well as reduce the number of rows sampled with 
#sampsize. You can also override the default number of variables to choose
#from with mtry, but the default is the square root of the total number
#available and that should work just fine. Since we only have a small
#dataset to play with, we can grow a large number of trees and not worry 
#too much about their complexity, it will still run pretty fast.
summary(forest)
varImpPlot(forest)
#There's two types of importance measures shown above.
#The accuracy one tests to see how worse the model
#performs without each variable, so a high decrease 
#in accuracy would be expected for very predictive variables.
#The Gini one digs into the mathematics behind decision trees,
#but essentially measures how pure the nodes are at the end of the tree.
#Again it tests to see the result if each variable is taken out and 
#a high score means the variable was important.

####### Make prediction#################
prediction.forest <- data.frame(predict(forest, bank2, type = "class"))
predictions.forest <- data.frame(Prediction = as.numeric(prediction.forest[,1])-1,Actual = as.numeric(bank2$y)-1)
predictions.forest$Correct <- (predictions.forest$Actual == predictions.forest$Prediction)
forest_Accuracy <- table(predictions.forest$Correct)/length(predictions.forest$Correct)*100
forest_Accuracy

####### predict with test############
prediction.forest.test<-data.frame(predict(forest,testing.set,type="class"))
predictions.forest.t <- data.frame(Prediction = as.numeric(prediction.forest.test[,1])-1,Actual = as.numeric(testing.set$y)-1)
predictions.forest.t$Correct <- (predictions.forest.t$Actual == predictions.forest.t$Prediction)
Tree_Accuracy.t <- table(predictions.forest.t$Correct)/length(predictions.forest.t$Correct)*100
Tree_Accuracy.t


################################################################################
################################################################################
###################Support vector machine#######################################
################################################################################
################################################################################
library(e1071)
library(kernlab)

svm.fit = ksvm(y~ ., data = bank3, type="C-svc", kernel="rbfdot", C=10)
plot(svm.fit,data=bank3,y~duration,type=1)

summary(svm.fit)

# If we only have two classes, we can get a nice 2-dimensional plot. Try this out for a variety of kernels and kernel parameters.
two.class.data = bank3[,c(8,21)]
two.class.data= two.class.data[1:50,]
svm.fit1 = ksvm(y ~contact, data = two.class.data, type="C-svc", kernel="rbfdot", C=10)
plot(svm.fit1,two.class.data)


####### Make prediction#################
prediction.svm <- data.frame(predict(svm.fit, bank3))
predictions.svm <- data.frame(Prediction = as.numeric(prediction.svm[,1])-1,Actual = as.numeric(bank3$y)-1)
predictions.svm$Correct <- (predictions.svm$Actual == predictions.svm$Prediction)
svm_Accuracy <- table(predictions.svm$Correct)/length(predictions.svm$Correct)*100
svm_Accuracy

####### predict with test############
prediction.svm.test<-data.frame(predict(svm.fit,testing.set))
predictions.svm.t <- data.frame(Prediction = as.numeric(prediction.svm.test[,1])-1,Actual = as.numeric(testing.set$y)-1)
predictions.svm.t$Correct <- (predictions.svm.t$Actual == predictions.svm.t$Prediction)
svm_Accuracy.t <- table(predictions.svm.t$Correct)/length(predictions.svm.t$Correct)*100
svm_Accuracy.t


# we need to see the accuracy to select optimal gamma and cost.

################################################################################
################################################################################
###########################bayesian network#####################################
################################################################################
################################################################################
library(bnlearn)
library(RGraphics)
bank3$pdays<-as.numeric(bank3$pdays)
bank3$previous<-as.numeric(bank3$previous)
bank3$age<-as.numeric(bank3$age)
bank3$duration<-as.numeric(bank3$duration)
bank3$campaign<-as.numeric(bank3$campaign)
bank.gs<-gs(bank3)
bank.gs
bn2<-iamb(bank3)
compare(bank.gs,bn2)

bn3<-fast.iamb(bank3)
bn4<-inter.iamb(bank3)
bank.hc<-hc(bank3,score="bic-cg")
plot(bank.hc)
bn.fit<-bn.fit(bank.hc,bank3)
summary(bn.fit)
####### Make prediction#################
prediction.bayes <- data.frame(predict(bn.fit$y, bank3))
predictions.bayes <- data.frame(Prediction = as.numeric(prediction.bayes[,1])-1,Actual = as.numeric(bank3$y)-1)
predictions.bayes$Correct <- (predictions.bayes$Actual == predictions.bayes$Prediction)
bayes_Accuracy <- table(predictions.bayes$Correct)/length(predictions.bayes$Correct)*100
bayes_Accuracy

####### predict with test############
testing.set$age<-as.numeric(testing.set$age)
testing.set$duration<-as.numeric(testing.set$duration)
testing.set$campaign<-as.numeric(testing.set$campaign)
testing.set$pdays<-as.numeric(testing.set$pdays)
testing.set$previous<-as.numeric(testing.set$previous)
prediction.bayes.test<-data.frame(predict(bn.fit$y,testing.set))
predictions.bayes.t <- data.frame(Prediction = as.numeric(prediction.bayes.test[,1])-1,Actual = as.numeric(testing.set$y)-1)
predictions.bayes.t$Correct <- (predictions.bayes.t$Actual == predictions.bayes.t$Prediction)
svm_Accuracy.t <- table(predictions.bayes.t$Correct)/length(predictions.bayes.t$Correct)*100
svm_Accuracy.t
################################################################################
################################################################################
###########################Neural Network#######################################
################################################################################
################################################################################
library(grid)
library(neuralnet)
# building neural net
logit.bic<-glm( y ~ contact + duration + poutcome + cons.price.idx + 
                  cons.conf.idx + euribor3m, family = binomial, data = bank3)

# Use variable from BIC logistic model
neural.1 <- neuralnet(y ~ contact + duration + poutcome + cons.price.idx + 
                        cons.conf.idx + euribor3m, data = bank3, hidden=100)
#Since neuralnet only deals with quantitative variables, you can convert all the qualitative variables (factors)
#to binary ("dummy") variables, with the model.matrix function
#it is one of the very rare situations in which R does not perform the transformation for you.

matrix<-model.matrix(~y+contact + duration + poutcome + cons.price.idx + 
                       cons.conf.idx + euribor3m, data=bank3)
head(matrix)
neural.1 <- neuralnet(yyes ~ contacttelephone + duration+poutcomenonexistent+poutcomesuccess +cons.price.idx+cons.conf.idx+euribor3m , data = matrix,
                      hidden=100)
plot(neural.1)
####### Make prediction#################
prediction.neural <- data.frame(predict(neural.1, bank3,type="class"))
predictions.neural <- data.frame(Prediction = as.numeric(prediction.neural[,1])-1,Actual = as.numeric(matrix$yyes)-1)
predictions.neural$Correct <- (predictions.neural$Actual == predictions.neural$Prediction)
neural_Accuracy <- table(predictions.neural$Correct)/length(predictions.neural$Correct)*100
neural_Accuracy

####### predict with test############
prediction.neural.test<-data.frame(predict(neural.1,testing.set))
predictions.neural.t <- data.frame(Prediction = as.numeric(prediction.neural.test[,1])-1,Actual = as.numeric(testing.set$y)-1)
predictions.neural.t$Correct <- (predictions.neural.t$Actual == predictions.neural.t$Prediction)
neural_Accuracy.t <- table(predictions.neural.t$Correct)/length(predictions.neural.t$Correct)*100
neural_Accuracy.t


################################################################################
################################################################################
###################################PCA##########################################
################################################################################
################################################################################
library(MASS)
library(car)
library(verification)
bank.pca<-princomp(na.omit(bank3),cor=T)
bank3$age<-as.numeric(bank3$age)
bank3$duration<-as.numeric(bank3$duration)
bank3$campaign<-as.numeric(bank3$campaign)


#Since PCA cannot handle the DATA with NA values.
# Also PCa does not fit for categorical variables.
# Therfore, I conclude use non-linear principal component for this data sets(called Multiple Correspondence Analysis).
library(ROCR)
library(homals)
bank.pca<-homals(bank3,)
plot(bank.pca,cor="red")
bank.pca
summary(bank.pca)
predict.tran<-predict(bank.pca,testing.set$y)
predict.perm<-performance(predict.tran,measure="tpr", x.measure="fpr")
predict(bank.pca,testing.set)
summary(predict.tran)
summary(predict.tran)
predict.tran$cl.table$y
plot(predict.tran$cl.table$y,'tpr','fpr')
tpr<-25304/(25304+513) #tp/(tp+fn)
fpr<-1337/(1337+1041) #fp/(fp+tn)
plot(bank.pca)
####### Make prediction#################
prediction.pca <- data.frame(predict(bank.pca, bank3))
predictions.pca <- data.frame(Prediction = as.numeric(prediction.pca[,1])-1,Actual = as.numeric(matrix$yyes)-1)
predictions.neural$Correct <- (predictions.neural$Actual == predictions.neural$Prediction)
neural_Accuracy <- table(predictions.neural$Correct)/length(predictions.neural$Correct)*100
neural_Accuracy

####### predict with test############
prediction.neural.test<-data.frame(predict(neural.1,testing.set))
predictions.neural.t <- data.frame(Prediction = as.numeric(prediction.neural.test[,1])-1,Actual = as.numeric(testing.set$y)-1)
predictions.neural.t$Correct <- (predictions.neural.t$Actual == predictions.neural.t$Prediction)
neural_Accuracy.t <- table(predictions.neural.t$Correct)/length(predictions.neural.t$Correct)*100
neural_Accuracy.t




##############################Cross Validation###################################################
bank1<-bank[sample(nrow(bank)),]
random<-sample(1:nrow(bank1))
num.bank.training<-as.integer(0.60*length(random))
bank.indices<-random[1:num.bank.training]
bank_kfold<-bank[bank.indices,]
accuracy <- c()
# Create 10 equally size folds
folds<-cut(seq(1,nrow(bank_kfold)),breaks=10,labels=FALSE)
for(i in 1:10){
  testIndex <- which(folds==i,arr.ind=TRUE)
  test<-bank_kfold[testIndex,]
  train<-bank_kfold[-testIndex,]
  
  logit.bic<-glm( y ~ contact + duration + poutcome + cons.price.idx + 
                    cons.conf.idx + euribor3m, family = binomial, data = train)  
  tree <- rpart(y~contact + duration + poutcome + cons.price.idx + 
                  cons.conf.idx + euribor3m, data=train, method="class")
  tree.prune = prune(tree, cp = 0.0195)
  
  forest<-randomForest(as.factor(y)~contact + duration + poutcome + cons.price.idx + 
                         cons.conf.idx + euribor3m,data=train, importance=TRUE, ntree=100)
  
  svm.fit = ksvm(y~ ., data = train, type="C-svc", kernel="rbfdot", C=10)
  
  newvalue_bic <- data.frame(predict(logit.bic,test,interval="predict",allow.new.levels = T))
  newvalue_bic[newvalue_bic<=0.5]=0
  newvalue_bic[newvalue_bic>0.5]=1
  prediction_bic <- data.frame(Prediction_bic = as.numeric(newvalue_bic[,1]))
  row.names(prediction_bic) <- row.names(test)
  prediction_tree <-data.frame(Prediction_tree = as.numeric(predict(tree.prune,test,type="class"))-1)
  prediction_forest <-data.frame(Prediction_forest = as.numeric(predict(tree.prune,test,type="class"))-1)
  prediction_svm <-data.frame(Prediction_svm = as.numeric(predict(svm.fit,test))-1)
  Actual <- data.frame(Actual = as.numeric(test$y)-1)
  row.names(prediction_bic) <- row.names(test)
  Predictions_All <- cbind(prediction_bic,prediction_tree,prediction_forest,prediction_svm,Actual)
  
  Predictions_All$Correct_bic <- (Predictions_All$Actual == Predictions_All$Prediction_bic)
  Predictions_All$Correct_tree <- (Predictions_All$Actual == Predictions_All$Prediction_tree)
  Predictions_All$Correct_forest <- (Predictions_All$Actual == Predictions_All$Prediction_forest)
  Predictions_All$Correct_svm <- (Predictions_All$Actual == Predictions_All$Prediction_svm)
  acc <- sapply(Predictions_All[c("Correct_bic","Correct_tree","Correct_forest","Correct_svm")],table)/length(Predictions_All[,1]) * 100 
  
  accuracy <- rbind(accuracy,acc[2,])
}
accuracy
# meeting on Wednesday(Send email next wednesday at 4pm? to professor on Today or Tomorrow(4/23))
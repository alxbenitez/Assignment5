library(caret)
library(gbm)
data(scat)
data<-scat

#######Questions
#1
S<-as.factor(data[,'Species'])
data['Species']<-unclass(S)
###Bobcat=1, Coyote=2, Gray_Fox=3
#summary(data)

#2
data=data[,-(2:5)]

#3 & 4
sum(is.na(data))
#47 nulls
ma<-data.matrix(data)

preProcValues <- preProcess(ma, method = c("knnImpute"))#knnImpute forces scale and center
data_processed <- predict(preProcValues, ma)
sum(is.na(data_processed))
data_processed[,1]=as.factor(ma[,1])## setting target variable back to orginal values->1,2,3
#No Categorical Values
data_new<-data.frame(data_processed)
data_new$Species<-as.factor(data_new$Species)## keep target as categorical for classification


#5
set.seed(100)
index <- createDataPartition(data_new$Species, p=0.75, list=FALSE)
trainSet <- data_new[ index,]
testSet <- data_new[-index,]



############Random Forest
#Fit Model
model_rf<-train(trainSet[,-1],trainSet[,1],method='rf', importance=T)

#Model Summary
print(model_rf)

#Feature importance
plot(varImp(object=model_rf),main="RF - Variable Importance")

#Confusion Matrix
predictions<-predict.train(object=model_rf,testSet[,-1],type="raw")
table(predictions)

confusionMatrix(predictions,testSet[,1])$table



##################Neural Network
#Fit Model
model_nnet<-train(trainSet[,-1],trainSet[,1],method='nnet', importance=T)

#Model Summary
print(model_nnet)

#Feature importance
#reformat varImp Importance for plotting
c<-varImp(object=model_nnet)
v<-c$importance
c$importance<-as.data.frame(v)

plot(c,main="Neural Net - Variable Importance")

#Confusion Matrix
predictions<-predict.train(object=model_nnet,testSet[,-1],type="raw")
table(predictions)

confusionMatrix(predictions,testSet[,1])$table



##################Naive Bayes
model_nb<-train(trainSet[,-1],trainSet[,1],method='naive_bayes', importance=T)

#Model Summary
print(model_nb)

#Feature importance
plot(varImp(object=model_nb),main="Naive Bayes - Variable Importance")

#Confusion Matrix
predictions<-predict.train(object=model_nb,testSet[,-1],type="raw")
table(predictions)

confusionMatrix(predictions,testSet[,1])$table




################GBM
model_gbm<-train(trainSet[,-1],trainSet[,1],method='gbm',distribution = "multinomial")

#Model Summary
print(model_gbm)

#Feature importance
plot(varImp(object=model_gbm),main="GBM - Variable Importance")

#Confusion Matrix
predictions<-predict.train(object=model_gbm,testSet[,-1],type="raw")
table(predictions)

confusionMatrix(predictions,testSet[,1])$table



#6
##Extract metrics from models
results<-data.frame(ExperimentName=c('RF','NNet', 'NB','GBM'),
                    Accuracy=c(model_rf$results[order(model_rf$results$Accuracy,decreasing = T),]['Accuracy'][1,1],model_nnet$results[order(model_nnet$results$Accuracy,decreasing = T),]['Accuracy'][1,1],
                               model_nb$results[order(model_nb$results$Accuracy,decreasing = T),]['Accuracy'][1,1],model_gbm$results[order(model_gbm$results$Accuracy,decreasing = T),]['Accuracy'][1,1]),
                    Kappa=c(model_rf$results[order(model_rf$results$Accuracy,decreasing = T),]['Kappa'][1,1],model_nnet$results[order(model_nnet$results$Accuracy,decreasing = T),]['Kappa'][1,1],
                            model_nb$results[order(model_nb$results$Accuracy,decreasing = T),]['Kappa'][1,1],model_gbm$results[order(model_gbm$results$Accuracy,decreasing = T),]['Kappa'][1,1]))
##Sort on Accuracy
results[order(results$Accuracy,decreasing = T),]




#7
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)

gbm_tuned<-train(trainSet[,-1],trainSet[,1],method='gbm',distribution = "multinomial",trControl=fitControl, tuneLength = 20)
print(gbm_tuned)
plot(gbm_tuned)


#8
library(ggplot2)
library(gridExtra)
p1<-ggplot(varImp(object=model_rf))+ggtitle("RF - Variable Importance")+geom_col(fill='blue')
p2<-ggplot(c)+ggtitle("NNet - Variable Importance")+geom_col(fill='red')
p3<-ggplot(varImp(object=model_nb))+ggtitle("NB - Variable Importance")+geom_col(fill='orange')
p4<-ggplot(varImp(object=gbm_tuned))+ggtitle("GBM - Variable Importance")+geom_col(fill='green')

grid.arrange(p1, p2, p3, p4, ncol=2)

#9
new.row<-data.frame(ExperimentName=c('GBM Tuned'),
                    Accuracy=c(gbm_tuned$results[order(gbm_tuned$results$Accuracy,decreasing = T),]['Accuracy'][1,1]),
                    Kappa=c(gbm_tuned$results[order(gbm_tuned$results$Accuracy,decreasing = T),]['Kappa'][1,1]))
results<-rbind(results,new.row)
results[order(results$Accuracy,decreasing = T),]

##Based on accuracy, GBM Tuned performs the best on this data. Boosting algorithm like Random Forest
#is an ensemble method, the difference being that predictors are made sequentially not independently.
#Random Forest can outperform GBM but its random nature does not guarentee it. Our accuracy ~64-73% suggests
#our models can predict better than a 1 in 3 guess. However,in my opinion, 70%  might be too low to be
#reliable and more models should be explored or find more/better data to work with.


#10
#a
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
spec_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],rfeControl = control)
spec_Pred_Profile

#Top 3 predictors
predictors<-c('d15N', 'Mass', 'd13C')

###Fitting Models
model_rf2<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf', importance=T)
model_nnet2<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet', importance=T)
model_nb2<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes', importance=T)
model_gbm2<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',distribution = "multinomial")
gbm_tuned2<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',distribution = "multinomial",trControl=fitControl, tuneLength = 20)

#b
##Extract metrics from models
results2<-data.frame(ExperimentName=c('RF','NNet', 'NB','GBM', 'GBM Tuned'),
                    Accuracy=c(model_rf2$results[order(model_rf2$results$Accuracy,decreasing = T),]['Accuracy'][1,1],model_nnet2$results[order(model_nnet2$results$Accuracy,decreasing = T),]['Accuracy'][1,1],
                               model_nb2$results[order(model_nb2$results$Accuracy,decreasing = T),]['Accuracy'][1,1],model_gbm2$results[order(model_gbm2$results$Accuracy,decreasing = T),]['Accuracy'][1,1],
                               gbm_tuned2$results[order(gbm_tuned2$results$Accuracy,decreasing = T),]['Accuracy'][1,1]),
                    Kappa=c(model_rf2$results[order(model_rf2$results$Accuracy,decreasing = T),]['Kappa'][1,1],model_nnet2$results[order(model_nnet2$results$Accuracy,decreasing = T),]['Kappa'][1,1],
                            model_nb2$results[order(model_nb2$results$Accuracy,decreasing = T),]['Kappa'][1,1],model_gbm2$results[order(model_gbm2$results$Accuracy,decreasing = T),]['Kappa'][1,1],
                            gbm_tuned2$results[order(gbm_tuned2$results$Accuracy,decreasing = T),]['Kappa'][1,1]))
##Sort on Accuracy
results2[order(results2$Accuracy,decreasing = T),]

#c
##The best performing model is Naive Bayes. Now that we eliminated most features, ensemble
#methods like Random Forest and Gradient Boosting become less effective having less possible features 
#to predict on. Neural Networks are probably too complex of a model for this data with only 110
#observations, so Naive Bayes performs the best on only a few feaures. Again we get accuracies better than
#random guessing but not close to an acceptable accuray. 





d=read.csv("D:\\R Course\\PROJECT DATASET DATA SCIENCE\\MedianImpData.csv")

#####LOGISTIC
library(caret)
set.seed(2019)
intrain=createDataPartition(y=d$TenYearCHD,p=0.7,list = F)
training=d[intrain,]
validation=d[-intrain,]

fit.lg <- glm(TenYearCHD~ . , data = training ,
              family = binomial())
pred.lg <- predict(fit.lg, newdata = validation ,
                   type = "response")
pred.rec=factor(ifelse(validation$TenYearCHD==1,"Y","N"),levels=c("Y","N"))

pred.lg.cat <- factor(ifelse(pred.lg ==1, "Y" , "N"),
                      levels = c("Y","N"))

confusionMatrix(pred.lg.cat , pred.rec,
                positive = "Y")

library(pROC)
plot.roc(validation$TenYearCHD, pred.lg, print.auc=TRUE ,
         col="magenta", main="Logistic Regression",
         legacy.axes=TRUE)


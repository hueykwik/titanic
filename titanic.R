library(dplyr)
library(readr)
library(randomForest)
library(reshape)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

RMSE = function(x, y) sqrt(mean((x-y)^2))

# Returns string w/o leading or trailing whitespace
Trim <- function (x) gsub("^\\s+|\\s+$", "", x)

CleanUpData = function(data) {
  # Change Sex from character to factor
  data$Sex = as.factor(data$Sex)
  
  # Replace missing ages with median age for each gender
  maleData = data %>% filter(Sex == 'male')
  maleMedianAge = median(maleData$Age, na.rm = TRUE)
  
  femaleData = data %>% filter(Sex == 'female')
  femaleMedianAge = median(femaleData$Age, na.rm = TRUE)
  
  data[data$Sex == 'male' & is.na(data$Age), 'Age'] = maleMedianAge
  data[data$Sex == 'female' & is.na(data$Age), 'Age'] = femaleMedianAge
  
  # Change Embarked to a factor
  data$Embarked = as.factor(data$Embarked)
  
  # Replace missing Embarked with mode
  data[is.na(data$Embarked), 'Embarked'] = Mode(data$Embarked)
  
  # Replace missing fares with median of respective class
  median_fare = array(dim = 3)
  for (i in seq(3)) {
    pclass.data = data %>% filter(Pclass == i) 
    median_fare[i] = median(pclass.data$Fare, na.rm = TRUE)
  }
  for (i in seq(3)) {
    data[is.na(data$Fare) & data$Pclass == i, 'Fare'] = median_fare[i]
  }
  
  # Extract titles
  data$Title = sapply(data$Name, FUN=function(x) {Trim(strsplit(x, split='[,.]')[[1]][2])})
  data$Title[data$Title %in% c('Capt', 'Don', 'Major', 'Sir')] = 'Sir'
  data$Title[data$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] = 'Lady'
  data$Title = factor(data$Title, levels = unique(data$Title))
  
  # Create FamilySize
  data$FamilySize = data$SibSp + data$Parch + 1
  
  # data$Surname = sapply(data$Name, function(x) {strsplit(x, split='[,.]')[[1]][1]})
  # data$FamilyId = paste0(as.character(data$FamilySize), data$Surname)
  # data$FamilySize[data$FamilySize <= 2] = 'Small'
  # 
  # famIds = data.frame(table(data$FamilyId))
  # famIds = famIds[famIds$Freq <=2, ]
  # 
  # data$FamilyId[data$FamilyId %in% famIds$Var1] = 'Small'
  # data$FamilyId = factor(data$FamilyId)
  
  return(data)
}

train.df = read_csv('train-original.csv')
test.df = read_csv('test.csv')

combo.df = rbind(train.df, test.df)
combo.df = CleanUpData(combo.df)

train.df = combo.df[1:891, ]
test.df = combo.df[892:1309, ]

# train.df = train.df %>% select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, 
#                                Embarked, Title)
train.df = train.df %>% select(Survived, Pclass, Sex, Age, FamilySize, Fare, 
                               Embarked, Title, SibSp, Parch)


# Random Forest
set.seed(1)

rf = randomForest(as.factor(Survived) ~ ., data = train.df, importance = TRUE)
varImpPlot(rf)

print(paste("Accuracy", sum(rf$predicted == train.df$Survived) / nrow(train.df)))

predictions = predict(rf, test.df)
submission = data.frame(PassengerId = test.df$PassengerId, Survived = predictions)
write.csv(submission, file = "forestR.csv", row.names = FALSE)

cf = cforest(as.factor(Survived) ~ ., data = train.df, 
             controls=cforest_unbiased(ntree=2000, mtry=3))
predictions = predict(cf, test.df, OOB=TRUE, type="response")
submission = data.frame(PassengerId=test.df$PassengerId, Survived=predictions)
write.csv(submission, file = "conditionalForestR.csv", row.names = FALSE)

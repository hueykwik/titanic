library(dplyr)
library(readr)
library(randomForest)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

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
  
  return(data)
}

train.df = read_csv('train-original.csv')
train.df = CleanUpData(train.df)

test.df = read_csv('test.csv')
test.df = CleanUpData(test.df)

train.df = train.df %>% select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, 
                               Embarked)

# Random Forest
p = 7
rf = randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch 
                  + Fare + Embarked, data = train.df, mtry = floor(sqrt(p)))

predictions = predict(rf, test.df)
submission = data.frame(PassengerId = test.df$PassengerId, Survived = predictions)
write.csv(submission, file = "forestR.csv", row.names = FALSE)


library(ggplot2)
library(corrplot)
library(mlbench)
library(caret)
library(reshape2)
library(dplyr)
library(ROCR)

data("PimaIndiansDiabetes")
head(PimaIndiansDiabetes)
str(PimaIndiansDiabetes)
with(PimaIndiansDiabetes, table(diabetes))

melted <- melt(PimaIndiansDiabetes, id.var = 'diabetes')
ggplot(melted, aes(x = value, fill = diabetes)) +
  geom_density(color = 'white', alpha = 0.6, size = 0.5) + 
  facet_wrap(~ variable, scales = 'free') +
  theme_minimal() + scale_fill_brewer(palette = 'Set1') +
  theme(legend.position = 'top', legend.direction = 'horizontal')

# ImpVars1 <- c('pregnant', 'glucose', 'mass', 'age')

cor_mat <- cor(
  within(PimaIndiansDiabetes, 
         diabetes <- as.numeric(diabetes))) %>%
  round(digits = 2)
corrplot(cor_mat, method = 'number',
         tl.srt = 45, tl.col = 'black')
# ImpVars2 <- c('pregnant', 'glucose', 'mass', 'age')

glm_all <- glm(diabetes ~ ., data = PimaIndiansDiabetes,
    family = binomial(link = 'logit'))
summary(glm_all)
varImp(glm_all)
# ImpVars3 <- c('pregnant', 'glucose', 'pressure', 'mass', 'pedigree')

#######################################################
auc <- function(model, outcome) { 
  per <- performance(prediction(model, outcome == 'pos'),
                     "auc")
  as.numeric(per@y.values)
}
set.seed(13)
PimaIndiansDiabetes$split <- runif(nrow(PimaIndiansDiabetes))
training <- subset(PimaIndiansDiabetes, split <= 0.9)
test <- subset(PimaIndiansDiabetes, split > 0.9)
vars <- c('pregnant', 'glucose', 'pressure', 'mass', 'pedigree')

glm_model <- glm(diabetes ~ ., 
                 data = training[, c('diabetes', vars)],
                 family = binomial(link = 'logit')
                   )
summary(glm_model)

training$model <- predict(glm_model, training[, c('diabetes', vars)],
                          type = 'response')
ggplot(training, aes(x = model, color = diabetes, linetype = diabetes)) +
  geom_density() +
  theme_minimal() +
  theme(legend.position = 'top', legend.direction = 'horizontal') +
  scale_color_brewer(palette = 'Set1')

pred_object <- prediction(training$model, training$diabetes)
perf_object <- performance(pred_object, "sens", "spec")
sensitivity <- perf_object@y.values[[1]]
specificity <- perf_object@x.values[[1]]
acc <- performance(pred_object, "acc")
accuracy <- acc@y.values[[1]]
error.rate <- 1 - accuracy
threshold <- acc@x.values[[1]]
errors <- data.frame(cbind(threshold, 
                           cbind(error.rate, 
                                 cbind(accuracy, 
                                       cbind(sensitivity, specificity)))))

library(reshape)
error.data <- melt(errors, id.vars = "threshold")
ggplot(error.data, aes(x = threshold, y = value, 
                       col = variable)) + 
  geom_line() + scale_x_continuous(breaks = seq(0.0, 1, 0.1)) +
  theme_minimal() + 
  theme(legend.position = 'top', legend.direction = 'horizontal') +
  scale_color_brewer(palette = 'Set1')

with(training, mean(diabetes == ifelse(
  model >= 0.32, 'pos', 'neg'
)))

auc(training$model, training$diabetes)

test$model <- predict(glm_model, test[, c('diabetes', vars)],
                      type = 'response')

ggplot(test, aes(x = model, color = diabetes, linetype = diabetes)) +
  geom_density() +
  theme_minimal() +
  theme(legend.position = 'top', legend.direction = 'horizontal') +
  scale_color_brewer(palette = 'Set1') 

with(test, mean(diabetes == ifelse(
  model >= 0.32, 'pos', 'neg'
)))

auc(test$model, test$diabetes)

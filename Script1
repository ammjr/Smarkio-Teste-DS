# Antonio Mendes Magalhaes Jr
# 25/02/21

rm(list=ls())

library(readxl)

dados <- read_excel("teste_smarkio_lbs.xls", sheet=1, col_names=TRUE)

# Se "true_class" for nula, assumir "pred_class"
dados$True_class <- ifelse(is.na(dados$True_class), 
                           dados$Pred_class, dados$True_class)
dados$probabilidade <- as.numeric(dados$probabilidade)
dados$True_class <- as.numeric(dados$True_class)

str(dados)

# ==================================================
# Análise exploratória
library(ggplot2)

# Frequencia das classes de dados
ggplot(dados, aes(x = status,y = ..count.., fill=status)) +
  geom_bar(alpha=1)+ 
  theme_bw()+
  labs(
    title = " ",
    subtitle = " ",
    x = " ",
    y = " ",
    fill = "Legenda") +
  theme(legend.position="none")

# Histograma de True_class
ggplot(dados, aes(x = as.numeric(True_class),y =..density..)) + 
  geom_histogram(bins = 20,fill="gray69")+
  theme_bw()+
  labs(
    title = "Histogram de 'True_class'")+
  geom_density(fill = "blue", alpha = 0.3)

# Histograma de Pred_class
ggplot(dados, aes(x = as.numeric(Pred_class),y =..density..)) + 
  geom_histogram(bins = 20,fill="gray69")+
  theme_bw()+
  labs(
    title = "Histogram de 'Pred_class'")+
  geom_density(fill = "green", alpha = 0.3)

# Histograma de probabilidade
ggplot(dados, aes(x = as.numeric(probabilidade),y =..density..)) + 
  geom_histogram(bins = 20,fill="gray69")+
  labs(
    title = "Histogram de 'probabilidade'")+
  geom_density(fill = "red", alpha = 0.3)

# ==================================================
# Desempenho do modelo

library(dplyr)
library(caret)

true_c <- as.factor(as.numeric(dados$True_class))
pred_c <- as.factor(dados$Pred_class)
u <- union(levels(true_c ),levels(pred_c))
t <- table(factor(pred_c, u), factor(true_c, u))

roc <- pROC::multiclass.roc(dados$True_class ~ dados$Pred_class, plot=F,
                            print.auc=F, legacy.axes=T)

metrics <- caret::confusionMatrix(t)
(cm <- metrics$table)
(acc <- metrics$overall[1])
(kappa <- metrics$overall[2])
(auc <- roc$auc)



# ==================================================
# Classificador (treinar utilizando apenas classes 'approved')
dados <- read_excel("teste_smarkio_lbs.xls", sheet=1, col_names=TRUE)

dados$True_class <- ifelse(is.na(dados$True_class), 
                           dados$Pred_class, dados$True_class)
dados$probabilidade <- as.numeric(dados$probabilidade)
dados$Pred_class <- as.numeric(dados$Pred_class)

dados_approved <- filter(dados, status=='approved')
dados_revision <- filter(dados, status== 'revision')

dados_approved$True_class <- as.factor(as.numeric(dados_approved$True_class))
dados_revision$True_class <- as.factor(as.numeric(dados_revision$True_class))

data_train <- dados_approved[-3]
data_test <- dados_revision[-3]

train_control <- trainControl(method= 'repeatedcv', number=10, repeats=3)
model_rf <- train(True_class ~ ., data = data_train, method = 'rf',
                  trControl=train_control)

pred_rf <- predict(model_rf, data_test)

str(pred_rf)

data_test_code <- as.numeric(data_test$True_class)
resp_test_code <- as.numeric(pred_rf)
roc <- pROC::multiclass.roc(data_test_code ~ resp_test_code, plot=F)


levels_all <- union(levels(as.numeric(pred_rf)),
                    levels(dados_revision$True_class))

true_c <- factor(dados_revision$True_class, levels_all)
pred_c <- factor(pred_rf, levels_all)

t <- table(pred_c, true_c)

metrics <- caret::confusionMatrix(t)
(cm <- metrics$table)
(acc <- metrics$overall[1])
(kappa <- metrics$overall[2])
(auc <- roc$auc)

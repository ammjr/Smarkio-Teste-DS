# Antonio Mendes Magalhaes Jr
# 25/02/21

rm(list=ls())

library(readxl)

dados <- read_excel("teste_smarkio_lbs.xls", sheet=2, col_names=TRUE)

dados <- dados[sample(1:nrow(dados),
                      length(1:nrow(dados))), 1:ncol(dados)]

# Extração de caracteristicas
{
library(tm)

#gerar a matriz de termos
gerar_matriz_termos = function(res){
  
  #criando o corpus  
  corpus = VCorpus(VectorSource(res$letra))
  
  #Convertendo para minusculo
  corpus <- tm_map(corpus, content_transformer(tolower))
  
  #corpus <- tm_map(corpus, removeNumbers)
  
  #Removendo pontuacao
  corpus<- tm_map(corpus,removePunctuation)
  
  #Stopwords
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  
  corpus <- tm_map(corpus, stemDocument)
  
  #Removendo espaços em brancos que podem aparecer
  corpus <- tm_map(corpus, stripWhitespace)
  
  #Matriz de frequencia
  matriz_termos <- DocumentTermMatrix(corpus)
  
  matriz_termos <- removeSparseTerms(matriz_termos, .95)
  
  return(matriz_termos)
  
}

#Matriz de termos
matriz <- gerar_matriz_termos(dados)


dados <- data.frame(as.factor(dados$artista),
                    as.matrix(matriz))
names(dados)[1]<- paste("artista")
}

#=========================================================
# Classificador com RNA
{
library(RSNNS)

dataValues <- dados[-1]
dataTargets <- t(dados[1])

# codificando (one-hot) a variavel resposta
dataTargets <- decodeClassLabels(dataTargets)

# Dividindo o dataset em treino/teste
dataset <- splitForTrainingAndTest(dataValues, dataTargets, ratio = 0.20)

# Normalizando o dataset
dataset <- normTrainingAndTestSet(dataset, type = "0_1")


#RSNNS::getSnnsRFunctionTable()
model_rna <- mlp(dataset$inputsTrain, dataset$targetsTrain, size=c(10),
             hiddenActFunc='Act_TanH', learnFunc = "Rprop",
             outputActFunc = "Act_Logistic", maxit=200, linOut = FALSE)

plotIterativeError(model_rna)

# Predicoes em relacao aos dados de teste
pred_rna <- predict(model_rna, dataset$inputsTest)

plotROC(fitted.values(model_rna), dataset$targetsTrain)
plotROC(pred_rna, dataset$targetsTest)

# 
library(pROC)
#par(pty='s')
data_test_code <- encodeClassLabels(dataset$targetsTest)
resp_test_code <- encodeClassLabels(pred_rna)
roc <- pROC::roc(data_test_code ~ resp_test_code,
          plot=T, print.auc=TRUE,
          auc.polygon=T, grid=TRUE, legacy.axes=T,
          ci = T)

metrics <- caret::confusionMatrix(as.factor(encodeClassLabels(dataset$targetsTest)), 
                             as.factor(encodeClassLabels(pred_rna)))
(cm <- metrics$table)
(acc_rna <- metrics$overall[1])
(kappa_rna <- metrics$overall[2])
(auc_rna <- roc$auc)
}
#===========================================
# Classificador Naive Bayes
{

data_train <- dados[1:round(0.9*nrow(dados)),]
data_test <- setdiff(dados, data_train)

library(naivebayes)
model_nb <- naivebayes::naive_bayes(artista ~ ., data = data_train)
pred_nb <- predict(model_nb, data_test, type = "class")
data_test_code <- as.numeric(data_test$artista)
resp_test_code <- as.numeric(pred_nb)

roc <- pROC::roc(data_test_code ~ resp_test_code,
                 plot=T, print.auc=TRUE,
                 auc.polygon=T, grid=TRUE, legacy.axes=T, ci = T)

library(caret)
metrics <- caret::confusionMatrix(as.factor(data_test$artista), 
                                  as.factor(pred_nb))
(cm <- metrics$table)
(acc_nb <- metrics$overall[1])
(kappa_nb <- metrics$overall[2])
(auc_nb <- roc$auc)
}
#===========================================
# Classificador SVM
{

data_train <- dados[1:round(0.8*nrow(dados)),]
data_test <- setdiff(dados, data_train)

library(e1071)
model_svm <- e1071::svm(artista ~ ., data = data_train)
pred_svm <- predict(model_svm, data_test, type = "class")

roc <- pROC::roc(data_test_code ~ resp_test_code,
                 plot=T, print.auc=TRUE,
                 auc.polygon=T, grid=TRUE, legacy.axes=T, ci = T)


library(caret)
# train_control <- trainControl(method= 'repeatedcv', number=10, repeats=3)
# model <- train(artista ~ ., data = data_train,
#                method = 'svmLinear', trControl=train_control,
#                metric=c('Accuracy'))
# model$pred
# model$results
# print(model)

metrics <- caret::confusionMatrix(as.factor(data_test$artista), 
                                  as.factor(pred_svm))
(cm <- metrics$table)
(acc_svm <- metrics$overall[1])
(kappa_svm <- metrics$overall[2])
(auc_svm <- roc$auc)
}
#===========================================
# Classificador Randon Forest
{

data_train <- dados[1:round(0.8*nrow(dados)),]
data_test <- setdiff(dados, data_train)

library(randomForest)

model_rf <- randomForest(artista ~ ., data = data_train, 
                         ntree=150, proximity=TRUE)

pred_rf <- predict(model_rf, data_test, type = "class")
data_test_code <- as.numeric(data_test$artista)
resp_test_code <- as.numeric(pred_rf)

roc <- pROC::roc(data_test_code ~ resp_test_code,
                 plot=T, print.auc=TRUE,
                 auc.polygon=T, grid=TRUE, legacy.axes=T, ci = T)

library(caret)
metrics <- caret::confusionMatrix(as.factor(data_test$artista), 
                                  as.factor(pred_rf))
(cm <- metrics$table)
(acc_rf <- metrics$overall[1])
(kappa_rf <- metrics$overall[2])
(auc_rf <- roc$auc)
}

#===========================================
# Comparação

rm(list=setdiff(ls(), c('dados',
                        'acc_rna', 'acc_nb', 'acc_svm', 'acc_rf',
                        'kappa_rna', 'kappa_nb', 'kappa_svm', 'kappa_rf',
                        'auc_rna', 'auc_nb', 'auc_svm', 'auc_rf')))

df_metrics <- data.frame(c(acc_rna, acc_nb, acc_svm, acc_rf),
                         c(kappa_rna, kappa_nb, kappa_svm, kappa_rf),
                         c(auc_rna, auc_nb, auc_svm, auc_rf))
rownames(df_metrics) <- c('Redes Neurais', 'Naive Bayes', 'SVM', 'Random Forest')
colnames(df_metrics) <- c('Acurácia', 'Kappa', 'AUC')
df_metrics

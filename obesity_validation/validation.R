ts_output <- read.csv("obesity_validation/twostage_output.csv")
rag_output <- read.csv("obesity_validation/rag_output.csv")
true_output <- read.csv("obesity_validation/true_parameters.csv")

ts_pred <- ts_output[, 1:4]
rag_pred <- rag_output[, 1:4]
true <- true_output[, 1:4]

confusion_matrix <- function(pred, true){
  pred_nf <- apply(pred, 1:2, function(x) grepl("ot found", x))
  true_nf <- is.na(true)
  confmat <- ifelse(pred_nf,
                    ifelse(true_nf, 'TN', 'FN'),
                    ifelse(true_nf, 'FP', 
                           ifelse(pred == true, 'TP', 'FP')))
  return(table(confmat))
}

cm_ts <- confusion_matrix(ts_pred, true)
cm_rag <- confusion_matrix(rag_pred, true)

cm_ts
cm_rag
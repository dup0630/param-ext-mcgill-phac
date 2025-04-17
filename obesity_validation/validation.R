dl_output <- read.csv("obesity_validation/dl_output.csv")
rag_output <- read.csv("obesity_validation/rag_output.csv")
true_output <- read.csv("obesity_validation/true_parameters.csv")

dl_pred <- dl_output[, 1:4]
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

cm_dl <- confusion_matrix(dl_pred, true)
cm_rag <- confusion_matrix(rag_pred, true)

cm_dl
cm_rag
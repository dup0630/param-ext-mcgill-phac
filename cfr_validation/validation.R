rag_output <- read.csv("cfr_validation/rag_results.csv")
df <- read.csv("cfr_validation/true_parameters.csv")

df <- df[order(df$PDF),]
rag_output <- rag_output[order(rag_output$Paper),c(1,3) ]

df$rag_ext <- rag_output$Case.Fatality.Rate..CFR.
df$rag_ext <- sapply(df$rag_ex, function(x) ifelse(grepl("Not found", x), NA, x))
df$CFR <- sapply(df$CFR, function(x) ifelse(grepl("NA", x), NA, as.numeric(x)))

sum(df$CFR == 0, na.rm=TRUE)
sum(df$CFR > 0, na.rm=TRUE)
sum(is.na(df$CFR))

df$result <- mapply(function(t, e) {
  if (is.na(t) && is.na(e)) return("TN")
  if (is.na(t) && !is.na(e)) return("FP")
  if (!is.na(t) && is.na(e)) return("FN")
  if (isTRUE(all.equal(as.numeric(t), as.numeric(e), tolerance = 1))) return("TP")
  return("FN")
}, df$CFR, df$rag_ext)

cm <- table(df$result)
success_rate <- (cm["TP"] + cm["TN"]) / sum(cm)

df
cm
cat('Success: ', success_rate, '\n')
sum(df$CFR== 0, na.rm = TRUE)

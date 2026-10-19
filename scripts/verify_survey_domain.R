# Post-hoc full-design domain ratio; frozen predictions, no FPC.
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(survey))
d <- read.csv(args[1])
d$strata_nested <- as.integer(factor(paste(d$state,d$stratum,sep=':')))
d$psu_nested <- as.integer(factor(paste(d$state,d$stratum,d$psu,sep=':')))
options(survey.adjust.domain.lonely=FALSE)
s <- svydesign(ids=~psu_nested,strata=~strata_nested,weights=~weight,data=d,nest=FALSE,check.strata=FALSE)
rows <- list()
for (method in c('fail','adjust','average')) {
 options(survey.lonely.psu=method)
 rows[[method]] <- tryCatch({
  r <- svyratio(~numerator,~domain,s); theta <- as.numeric(coef(r));se <- as.numeric(SE(r));df <- degf(s)
  data.frame(method=method,estimate=theta,se=se,df=df,lower=theta-qt(.975,df)*se,upper=theta+qt(.975,df)*se,status='POST_HOC')
 },error=function(e)data.frame(method=method,estimate=NA,se=NA,df=degf(s),lower=NA,upper=NA,status=conditionMessage(e)))
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
writeLines(capture.output(sessionInfo()),paste0(args[2],'.session.txt'))

rm(list = ls())
require("oscaR")
require("nucleR")
require("nucPredR")
require("IRanges")
require("rtracklayer")
require("GenomicRanges")

cla = get(load("data/tss_classes_V136.RData"))
txEndDF = get(load("data/tts_classes_V136.RData"))
cal <- get(load("data/nuc_calls_merg.2x.trim.ih_methyl_V136_noMet.rep1.RData"))
genes = readGFF("data/genes_merged.gff")
cov2 = get(load("data/cov.2x.trim.ih_methyl_V136_noMet.rep1.RData"))
cov = get(load("data/ih_methyl_V136_noMet.rep1_cov.RData"))
names(cov) = names(cov2)
cov=cov[-6]

#Load info from TxEnd to cla DF
cla$pos = as.numeric(cla$pos)
cla$p1.pos = as.numeric(cla$p1.pos)
cla$last.pos = as.numeric(txEndDF$m1.pos)
cla$end.descr = txEndDF$descr
cla$end = as.numeric(txEndDF$pos)
cla$nuc.length = abs(cla$last.pos-cla$p1.pos)
cla$d.from.int=cla$nuc.length-165*round(cla$nuc.length/165)
cla$glength=abs(cla$pos-cla$end)

#Periodic positioning from p1 and last nucleosomes
periods=c(165,165)
period=165

covp1 <- function(x,alpha,sigma, period) {
  cov_p1 = (1 + alpha + sin(pi/2 + 2*(pi/period)*x)) * sigma^(abs(x)/period)
  return(cov_p1)
}
covlast <- function(x,sigma, L, period) {
  x_p = L-x
  cov_last = (1 + sin(pi/2 + 2*(pi/period)*x_p)) * sigma^(abs(x_p)/period)
  return(cov_last)
}
ecov <- function(x, alpha,sigma, L, period) {
  cov_p1 = covp1(x,alpha,sigma,period)
  cov_last = covlast(x,sigma, L,period)
  cov_p1_0 = covp1(0,alpha,sigma,period)
  cov_last_0 = covlast(0,sigma, L,period)
  total_cov = (cov_p1 + cov_last) / (cov_p1_0 + cov_last_0)
  return(total_cov)
}

coverageAll2.chr<-function(nuc.start,nuc.end,L,chr,s,period, alpha=0.2, sigma=0.7){
        cov_new=rep(0,L)
        per=period
        sper=floor(per/2)
        v=df[df$chrom==chr,]$nuc.length
        for(j in 1:length(nuc.start)){
                        numOfnucs=floor(v[j]/per)
                        if(as.character(s[j])=="+"){
                          cov_new[nuc.start[j]+(-sper:(per*numOfnucs+sper))]=cov_new[nuc.start[j]+(-sper:(per*numOfnucs+sper))]+
                          ecov(-sper:(per*numOfnucs+sper), alpha, sigma, v[j], period)
                        }else{
                          cov_new[nuc.end[j]+(-sper:(per*numOfnucs+sper))]=cov_new[nuc.end[j]+(-sper:(per*numOfnucs+sper))]+
                          rev(ecov(-sper:(per*numOfnucs+sper), alpha, sigma, v[j], period))
                        }
        }
        return(cov_new)
}

#Cov of all genes with a determined position for p1 and last

df=cla[!is.na(cla$p1.pos)&!is.na(cla$last.pos),]
nuc_start=df$p1.pos
nuc_end=df$last.pos

covPredAll=mclapply(1:length(names(cov)),
                    function(i){coverageAll2.chr(nuc_start[which(df$chrom==names(cov)[i])],
                                                                      nuc_end[which(df$chrom==names(cov)[i])],
                                                                      length(cov[[i]]),
                                                                      names(cov)[i],
                                                                      s=df$strand[which(df$chrom==names(cov)[i])],
                                                                      periods)},
                    mc.cores=17)

names(covPredAll) = names(cov)
save(file="data/covPredAll.RData",object=covPredAll)



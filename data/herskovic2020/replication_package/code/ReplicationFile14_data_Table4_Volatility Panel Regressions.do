clear
set more off

import delimited "../output/CleanData/paneldata_July2017.txt", delimiter(tab)  encoding(ISO-8859-1)

gen dates = v1
drop if dates < 1980 | dates > 2012
drop dates

foreach var of varlist v*{
destring  `var', replace ignore("NaN")
}

rename v1 dates
rename v2 permno
rename v3 logrv
rename v4 logK
rename v5 logKAT
rename v6 logKSA
rename v7 laglogK
rename v8 laglogKAT
rename v9 laglogKSA
rename v10 logage
rename v11 leverage
rename v12 concent
rename v13 rdfrac
rename v14 industry
rename v15 herfout
rename v16 herfin
rename v17 logME
rename v18 logAT
rename v19 logSA
rename v20 laglogME
rename v21 laglogAT
rename v22 laglogSA

gen logherfin = ln(herfin)
gen logherfout = ln(herfout)

xtset permno  dates

*build cohort dummies
bysort permno : egen firstyear = min(dates*exp(logK)/exp(logK))
tabulate firstyear, gen(dum)



********************************************************************************
*
*         Table
*
********************************************************************************
 
gen laglogsize = laglogK


* Columun 1
reg logrv laglogsize,  vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel replace noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 2
reg logrv logherfout,  vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 3
reg logrv laglogsize logherfout,  vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 4
reg logrv laglogsize logage L.leverage concent,  vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 5
reg logrv  logage L.leverage concent logherfout,  vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 6
reg logrv laglogsize  logage L.leverage concent logherfout,  vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 7
areg logrv laglogsize L.leverage concent,  absorb(fir) vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 8
areg logrv  L.leverage concent logherfout,  absorb(fir) vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 9
areg logrv laglogsize  L.leverage concent logherfout,  absorb(fir) vce(cluster indust)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 10
areg logrv laglogsize,  absorb(industry) vce(cluster dates)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 11
areg logrv logherfout,  absorb(industry) vce(cluster dates)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2

* Columun 12
areg logrv laglogsize logherfout,  absorb(industry) vce(cluster dates)
outreg2 using ../tables/Table4.xls, excel append noaster stats(coef tstat) bdec(2) tdec(2) noparen adjr2



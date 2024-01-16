pwd
cd "...\Figure and table\code"  /*Current folder*/
***********************************
**# Bookmark #1  Descriptions
*fig2 reading data
use province_map.dta , clear
sum
gen O_rate = Overduearea / Planneddeliveryarea

*Figure 2 code
spmap Overduearea using "china_map.dta" , id(id) ndfcolor(grey) label(label(ename)  xcoord(x_coord) ycoord(y_coord) size(*.6)) fcolor(Blues2) ocolor(white white white white white white white) osize(thin *0.2 *0.2 *0.2 *0.2 *0.2 *0.2 *0.2) /// 
	clm(c)   clbreak(0.00 10000.00 100000.00 300000.00  500000.00  700000.00 900000.00 1000000.00 1060000.00 )  /// 
	title("Research on the distribution of overdue projects" , size(*0.5)) subtitle("Research as of November 2023, China", size(*0.5))  ///
	plotregion(icolor(white)) graphregion(icolor(white))  /// 	
	scalebar(units(100) scale(1/2) xpos(50) label(Kilometers)) ///
	diagram(variable(O_rate) range(0 1)  xcoord(x_coord) ycoord(y_coord) fcolor(red))  ///
	saving(3,replace) 

	
label variable O_rate "Overdue delivery rate"	
	
twoway (scatter Overduearea Planneddeliveryarea) (lfitci Overduearea Planneddeliveryarea) ///
	(scatter O_rate Planneddeliveryarea, yaxis(2) mcolor(red) msize(medsmall) msymbol(triangle) mlabcolor()), ///
	ytitle(`"Overduew delivery area"') ylabel(, labsize(small))  ///
	ytitle(`"Overdue rate"', axis(2)) ylabel(, labsize(small) axis(2)) ///
	xtitle(`"Planned delivery area"') xlabel(, labsize(small)) ///
	legend(size(small) position(4) ring(0))   ///
	saving(4, replace) 

	
	
graph combine "3" "4" , row(1)
rm 3.gph
rm 4.gph


*Figure 3
label variable Red_O "Red-categorized projects with overdue delivery"	
label variable Yellow_O "Yellow-categorized projects with overdue delivery"
label variable Green_O "Green-categorized projects with overdue delivery"


spmap Red_O using "china_map.dta" , id(id) label(label(ename)  xcoord(x_coord) ycoord(y_coord) size(*.6)) fcolor(Reds2) ocolor(white white white white white white white) osize(thin *0.1  *0.1 *0.1 *0.1 *0.1  *0.1  *0.1  *0.1 *0.1)  /// 
	cln(15) ///
	title("The distribution of overdue projects with red clustered" , size(*0.5)) subtitle("Research as of November 2023, China", size(*0.5))  ///	
	scalebar(units(100) scale(1/2) xpos(50) label(Kilometers)) ///
	saving (5, replace)

spmap Yellow_O using "china_map.dta" , id(id) label(label(ename)  xcoord(x_coord) ycoord(y_coord) size(*.6)) fcolor(Heat) ocolor(white white white white white white white) osize(thin *0.1  *0.1 *0.1 *0.1 *0.1  *0.1  *0.1  *0.1 *0.1)  /// 
	cln(20) ///
	title("The distribution of overdue projects with yellow clustered" , size(*0.5)) subtitle("Research as of November 2023, China", size(*0.5))  ///	
	scalebar(units(100) scale(1/2) xpos(50) label(Kilometers)) ///
	saving (6, replace)
	
	
spmap Green_O using "china_map.dta" , id(id) label(label(ename)  xcoord(x_coord) ycoord(y_coord) size(*.6)) fcolor(Greens2) ocolor(white white white white white white white) osize(thin *0.1  *0.1 *0.1 *0.1 *0.1  *0.1  *0.1  *0.1 *0.1)  /// 
	cln(20) ///
	title("The distribution of overdue projects with green clustered" , size(*0.5)) subtitle("Research as of November 2023, China", size(*0.5))  ///	
	scalebar(units(100) scale(1/2) xpos(50) label(Kilometers)) ///
	saving (7, replace)


graph combine "5" "6"  "7", row(1)
rm 5.gph
rm 6.gph
rm 7.gph





***********************************************************



**# Bookmark #2 Empirical study
**From the table5 begining
use data.dta , clear

label variable y "Delivery overdue area"    /*逾期交付面积*/
label variable x1 "Pre-sale monitoring account balance"    /*监控户余额*/
label variable x2 "Whether commercial acceptance default"   /*商票跳票*/
label variable x3 "Payment ratio of new production value"  /*4-10月项目C值*/
label variable x4 "Ratio of actual payments to non-payments"  /*4-10月项目R值*/
label variable x5 "Actual cash disbursements as a percentage of planned cash disbursements"  /*1-10月付现占计划付款的比例*/
label variable x6 "Settlement payments as a percentage of scheduled payments"  /*4-10月结算款付现的比例*/
label variable x7 "Ratio of the value of the payment offset housing to the scheduled payments"  /*1-10月工抵占计划付款的比例*/
label variable color "Color clustering"           /*横截面红黄绿分档*/
label variable  Overduearea "Overdue delivery area"  
label variable	Planneddeliveryarea "Planned delivery area"

rename Overduearea  Overdue
rename Planneddeliveryarea Planned
gen O_rate = Overdue / Planned
label variable O_rate "Delivery overdue rate"

*计算多少个省
tabulate Province

*计算逾期总计面积
sum Overdue
local total = r(sum)
display `total'

*按颜色聚类的箱型图 Fig2 下半部分
sort color
capture drop color_text
gen color_text = "" 
replace color_text = "Red" if color == 1
replace color_text = "Yellow" if color == 2
replace color_text = "Green" if color == 3
graph box Overdue Planned, by(, legend(position(6))) xsize(10) ysize(4) by(color color_text, rows(1)) subtitle(, color(black))

* Table 5 statistical description
logout, save (static_description) word replace: sum y x1 x2 x3 x4 x5 x6 x7

* Table 6 2-way frequency table
tabstat color, by(Region) stats(n)
table Region Province color_text, stat(n color)

*VIF test
global fixvar c.x5 c.x6 c.x7
glm y c.x1 i.x2  c.x3 c.x4 $fixvar, family(gamma) link(log)
vif, uncentered

*样本间的x2分类t检验, 用于两组样本的对比, 这段结果仅作为观察, 不贴入文章中
asdoc ttest y, by(x2)    
asdoc ttest x1, by(x2)
asdoc ttest x3, by(x2)
asdoc ttest x4, by(x2)
asdoc ttest x5, by(x2)
asdoc ttest x6, by(x2)
asdoc ttest x7, by(x2)


*Empirical section


**Baseline
glm  y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 i.color, family(gamma) link(log) vce(cluster Region )
est store glm_r0
reg y c.x1 c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 i.color , vce(cluster Region)
est store ols_r0
mixed y c.x1 c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 i.color, vce(cluster Region)
est store mixed_r0
mixed y c.x1 c.x3 c.x4 c.x5 c.x6 c.x7 i.x2  || Region: || color_text: , mle  vce(cluster Region)
est store mixed_r1
esttab  glm_r0 ols_r0 mixed_r0 mixed_r1 using regression_result.rtf,replace nogap ar2 b(%6.4f) t(%6.4f) star(* 0.1 ** 0.05 *** 0.01)   //再导出word

*Baseline validation
capture drop y2
gen y2 = Overdue / Planned
tobit y2 c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  , ll(0)  nolog vce(cluster Region)  
est store tobit_r0
tobit y2 c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.color , ll(0)  nolog vce(cluster Region)  
est store tobit_r1
tobit y2 c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 i.color , ll(0)  nolog vce(cluster Region) 
est store tobit_r2
tobit y2 c.x1  c.x3 c.x4  i.x2 i.color , ll(0)  nolog vce(cluster Region) 
est store tobit_r3
glm y x1 x2 x3 x4 x5 x6 x7 i.x2##c.x1 i.x2##c.x3 i.x2##c.x4, family(gamma) link(log) vce(cluster Region) /*x2与其他自变量之间没有交互作用*/
est store glm_r1
esttab tobit_r0 tobit_r1 tobit_r2 tobit_r3 glm_r1 using regression_result.rtf ,replace nogap ar2 b(%6.4f) t(%6.4f) star(* 0.1 ** 0.05 *** 0.01)   //再导出word


*Color clusters analysis
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 if color == 1 , family(gamma) link(log) vce(cluster Region )
est store color_red
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 if color == 2 , family(gamma) link(log) vce(cluster Region )
est store color_yellow
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 if color == 3 , family(gamma) link(log) vce(cluster Region )
est store color_green
esttab color_red color_yellow color_green using regression_result.rtf ,replace nogap ar2 b(%6.4f) t(%6.4f) star(* 0.1 ** 0.05 *** 0.01)   //再导出word



*Region analysis
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.x2  if Region == "Eastern area" , family(gamma) link(log) vce(cluster color )
est store region_e
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.x2  if Region == "Western area" , family(gamma) link(log) vce(cluster color )
est store region_w
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7 i.x2  if Region == "Central area" , family(gamma) link(log) vce(cluster color )
est store region_c
esttab region_c region_e region_w using regression_result.rtf ,replace nogap ar2 b(%6.4f) t(%6.4f) star(* 0.1 ** 0.05 *** 0.01)   //再导出word


*Region and Color
***central region
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Central area" & color == 1  , family(gamma) link(log) vce(cluster x2)
est store region_c_r
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Central area" & color == 2  , family(gamma) link(log) vce(cluster x2)
est store region_c_y
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Central area" & color == 3  , family(gamma) link(log) vce(cluster x2)
est store region_c_g

*coefplot 生成图像 *https://www.bilibili.com/video/BV1e8411a7VL/?spm_id_from=333.337.search-card.all.click&vd_source=b7ad8316452cbe8d0cd3898c4d5d72d9*
***central region
coefplot region_c_r , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(red)) ///
	ms(s) ///
	mcolor(red)  ///
	xline(0, lp(dash) lcolor(red))  ///
	xlabel(, labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Red cluster", size(medium)) ///
	graphregion(color (white)) name(cr, replace)

coefplot region_c_y , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(brown)) ///
	ms(s) ///
	mcolor(brown)  ///
	xline(0, lp(dash) lcolor(brown))  ///
	xlabel(, labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Yellow cluster", size(medium)) ///
	graphregion(color (white)) name(cy, replace)
	
coefplot region_c_g , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(green)) ///
	ms(s) ///
	mcolor(green)  ///
	xline(0, lp(dash) lcolor(green))  ///
	xlabel( , labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Green cluster", size(medium)) ///
	graphregion(color (white)) name(cg, replace)
	
graph combine cr cy cg, ycommon rows(1) title(`"Include x2 effect"', size(small) position(6)) scheme( ) commonscheme xsize(6) ysize(2) scale(2)


****western region 
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Western area" & color == 1  , family(gamma) link(log) vce(cluster x2)
est store region_w_r
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Western area" & color == 2  , family(gamma) link(log) vce(cluster x2)
est store region_w_y
count if Region == "Western area" & color == 3   /*7 samples*/
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Western area" & color == 3  , family(poisson) link(log) vce(cluster x2) 
est store region_w_g
****western region 
coefplot region_w_r , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(red)) ///
	ms(s) ///
	mcolor(red)  ///
	xline(0, lp(dash) lcolor(red))  ///
	xlabel(, labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Red cluster", size(medium)) ///
	graphregion(color (white)) name(wr, replace)

coefplot region_w_y , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(brown)) ///
	ms(s) ///
	mcolor(brown)  ///
	xline(0, lp(dash) lcolor(brown))  ///
	xlabel(, labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Yellow cluster", size(medium)) ///
	graphregion(color (white)) name(wy, replace)
	
coefplot region_w_g , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(green)) ///
	ms(s) ///
	mcolor(green)  ///
	xline(0, lp(dash) lcolor(green))  ///
	xlabel( , labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Green cluster", size(medium)) ///
	graphregion(color (white)) name(wg, replace)
	
graph combine wr wy wg, ycommon rows(1) title(`"Include x2 effect"', size(small) position(6)) scheme( ) commonscheme xsize(6) ysize(2) scale(2)


****eastern region 
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Eastern area" & color == 1  , family(gamma) link(log) vce(cluster x2)
est store region_e_r
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Eastern area" & color == 2  , family(gamma) link(log) vce(cluster x2)
est store region_e_y
count if Region == "Eastern area" & color == 3   /*7 samples*/
glm y c.x1  c.x3 c.x4 c.x5 c.x6 c.x7  if Region == "Eastern area" & color == 3  , family(poisson) link(log) vce(cluster x2)
est store region_e_g

****eastern region 
coefplot region_e_r , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(red)) ///
	ms(s) ///
	mcolor(red)  ///
	xline(0, lp(dash) lcolor(red))  ///
	xlabel(, labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Red cluster", size(medium)) ///
	graphregion(color (white)) name(er, replace)

coefplot region_e_y , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(brown)) ///
	ms(s) ///
	mcolor(brown)  ///
	xline(0, lp(dash) lcolor(brown))  ///
	xlabel(, labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Yellow cluster", size(medium)) ///
	graphregion(color (white)) name(ey, replace)
	
coefplot region_e_g , drop(_cons) ///
	levels(95) ///
	ciopts(recast (rcap) color(green)) ///
	ms(s) ///
	mcolor(green)  ///
	xline(0, lp(dash) lcolor(green))  ///
	xlabel( , labsize(medium))   ///
	ylabel( , labsize(medium))   ///
	nolabel ///
	title("Green cluster", size(medium)) ///
	graphregion(color (white)) name(eg, replace)
	
graph combine er ey eg, ycommon rows(1) title(`"Include x2 effect"', size(small) position(6)) scheme( ) commonscheme xsize(6) ysize(2) scale(2)


*Robustness test
**第一种 PSM检验
capture drop Region_n
gen Region_n = .
replace Region = trim(Region)
replace Region_n = 1  if Region == "Central area"
replace Region_n = 2  if Region == "Western area"
replace Region_n = 3  if Region == "Eastern area"
//1:2匹配
psmatch2 x2  c.x1 c.x3 c.x4 c.x5 c.x6 c.x7 i.color , out(y) logit neighbor(2) ties common ate

//平衡性检验
pstest , both graph  saving(psm, replace)

**第二种检验方式 Appendix
reg y c.x1 c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 i.color , robust 
est store test_1
glm y c.x1 i.x2 c.x3 c.x4 c.x5 c.x6 i.color, family(gamma) link(log) vce(cluster  Region )
est store test_2
set seed 123
bootstrap , reps(100): reg y c.x1 c.x3 c.x4 c.x5 c.x6 c.x7 i.x2 i.color i.Region_n , robust
est store test_3
mixed y c.x1 c.x3 c.x4 c.x5 c.x6 c.x7 i.x2  || color_text: || Region:  , mle  vce(robust)
est store test_4
bootstrap , reps(100) : glm y c.x1 i.x2 c.x3 c.x4 c.x5 c.x6 , family(gamma) link(log) vce(cluster  Region )  /*bootstrap 500次, 只剩x1稳健*/
est store test_5
esttab test_1 test_2 test_3 test_4 test_5 using regression_result.rtf ,replace nogap ar2 b(%6.4f) t(%6.4f) star(* 0.1 ** 0.05 *** 0.01)   //再导出word































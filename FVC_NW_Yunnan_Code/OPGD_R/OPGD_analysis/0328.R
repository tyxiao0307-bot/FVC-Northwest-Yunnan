setwd("D:\\R")
##install.packages("GD")##下载一下GD包
library('GD')
test <- read.csv("D2000.csv")#这里注意保存为csv格式的文件，后续不会报错

# 查看数据的前几行
head(test)

# 定义离散化方法和区间数
discmethod <- c("equal", "natural", "quantile", "geometric")
discitv <- c(4:12)

# 使用gdm函数对数据进行分析,前面是所有自变量名称，后面是连续变量的名称，这里要对类变量进行剔除，传统的类变量有土地利用，
test.fd <- gdm(Y~X1+X2+X3+X4+X5+X6+X7+X8,continuous_variable = c(paste0("X", 1:8)[-6]), data = test,discmethod = discmethod,discitv = discitv)

# 打印分析结果
test.fd

# 设置 PDF 图形设备和字体
pdf("所有结果图片（7张）.pdf", family = "Times", width = 8, height = 7) # 设置图形设备为PDF，字体为Times New Roman

# 绘制结果的图表
plot(test.fd)

# 关闭图形设备
dev.off()

# 提取交互检测结果和因子检测的q值
交互 <- test.fd[["Interaction.detector"]][["Interaction"]]
q值 <- test.fd[["Factor.detector"]][["Factor"]]

# 提取风险检测和生态检测的结果
风险探测 <- test.fd[["Risk.mean"]]
生态探测 <- test.fd[["Ecological.detector"]][["Ecological"]]

# 提取分类结果
分类 <- test.fd[["Discretization"]]

# 提取X1到XN的分类结果
result_list <- lapply(分类, function(x) {
  data.frame(method = x$method, n.itv = x$n.itv)
})

# 将结果组合为一个数据框
result_df <- do.call(rbind, result_list)

# 提取X1到XN的风险检测结果
result_list2 <- lapply(风险探测, function(x) {
  data.frame(Mean = x$meanrisk, itv = x$itv)
})

# 将结果组合为一个数据框
result_df2 <- do.call(rbind, result_list2)

# 导出因子探测结果
write.csv(q值, file = "单因子显著情况及q值表.csv", row.names = FALSE, fileEncoding = "UTF-8")

# 导出生态探测结果
write.csv(生态探测, file = "生态探测结果表.csv", row.names = FALSE, fileEncoding = "UTF-8")

# 导出交互探测结果
write.table(交互, file = "因子交互探测结果表.csv", sep = ",", row.names = TRUE, fileEncoding = "UTF-8")

# 导出分类结果
write.table(result_df, file = "最优离散化类别及分类数.csv", sep = ",", row.names = TRUE, fileEncoding = "UTF-8")

# 导出风险探测结果
write.table(result_df2, file = "风险探测结果表.csv", sep = ",", row.names = TRUE, fileEncoding = "UTF-8")

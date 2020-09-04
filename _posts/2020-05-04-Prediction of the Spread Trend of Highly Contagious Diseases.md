---
layout:     post
title:      "COVID-19的传播趋势预测"
subtitle:   "Prediction of the Spread Trend of COVID-19"
date:       2020-05-04
author:     "wumingyao"
header-img: "img/in-post/2020.05/04/bg.jpg"
tags: [预测模型,COVID-19,时空预测,STResNet]
categories: [算法大赛]
---

## 主要内容
* [背景 Background](#p1)
* [任务描述 Task Definition](#p2)
* [数据集 DataSet](#p3)
* [模型 Model](#p4)
* [Experiments](#p5)
* [Ablation Studies](#p6)
* [ Conclusion](#p7)

## 正文

###  <span id="p1">背景 Background</span>

传染病（Contagious Diseases）的有效防治是全人类面临的共同挑战，COVID-19自2019年12月份爆发以来，截止2020年5月3日，全球感染人数已超350万。如何通过大数据，特别是数据的时空关联特性，来精准预测传染病的传播趋势和速度，将极大有助于人类社会控制传染病，保障社会公共卫生安全。

###  <span id="p2">任务描述 Task Definition</span>
构造传染病群体传播预测模型，根据该地区传染病的历史每日新增感染人数、城市间迁徙指数、网格人流量指数、网格联系强度和天气等数据，预测每个城市各区域未来一段时间每日新增感染人数。

### <span id="p3">数据集 DataSet</span>
数据集来自百度AI，可[点击下载](http://cdn.ikcest.org/bigdata2020/train_data.zip)，该数据集去除了城市真实名称等相关信息。数据集共包括5个城市，每个城市目录下的数据集总体说明：

1.各区域每天新增感染人数。文件名：infection.csv。提供前45天每天数据，文件格式为日期，区域ID，新增感染人数。

| 字段名称 Field name | 含义 Description | 示例 Example |
|---|---|---|
| date | 日期 Date | 21200501 |
| region_id | 区域ID Region ID | 1 |
| index | 新增感染人数 Number of newly infected persons | 20 |

2.城市间迁徙指数。文件名：migration.csv。提供45天每天数据。文件格式为迁徙日期，迁徙出发城市，迁徙到达城市，迁徙指数。

| 字段名称 Field name | 含义 Description | 示例 Example |
|---|---|---|
| date | 迁徙日期 Migration date | 21200501 |
| departure_city | 迁徙出发城市 Migration Departure city | A |
| arrival_city | 迁徙到达城市 Migration Arrival city | B |
| index | 迁徙指数 Migration scale index | 0.0125388 |

3.网格人流量指数。文件名：density.csv。提供45天内每周两天抽样数据，文件格式为日期，小时，网格中心点经度，网格中心点纬度，人流量指数。

| 字段名称 Field name | 含义 Description | 示例 Example |
|---|---|---|
| date | 日期 Date | 21200501 |
| hour | 小时 Hour | 10 |
| grid_x | 网格中心点经度 Longitude of grid center point | 166.306128 |
| grid_y | 网格中心点纬度 Latitude of grid center point | 20.331142 |
| index | 人流量指数 Population flow index | 3.1 |

4.网格关联强度。文件名：transfer.csv。城市内网格间关联强度数据，文件格式为小时，出发网格中心点经度，出发网格中心点纬度，到达网格中心点经度，到达网格中心点纬度，迁移强度。

| 字段名称 Field name | 含义 Description | 示例 Example |
|---|---|---|
| hour | 小时 Hour | 10 |
| start_grid_x | 出发网格中心点经度 Longitude of departure grid center point | 166.306128 |
| start_grid_y | 出发网格中心点经度 Longitude of arrival grid center point | 20.331142 |
| end_grid_x | 到达网格中心点经度 Longitude of arrival grid center point | 171.678139 |
| end_grid_y | 到达网格中心点纬度 Latitude of arrival grid center point | 17.812359 |
| index | 迁移强度 Transfer intensity | 0.2 |

5.网格归属区域。文件名：grid_attr.csv。城市内网格对应的归属区域ID，文件格式为网格中心点经度，网格中心点纬度，归属区域ID。

| 字段名称 Field name | 含义 Description | 示例 Example |
|---|---|---|
| grid_x | 网格中心点经度 Longitude of grid center point | 166.306128 |
| grid_y | 网格中心点纬度 Latitude of grid center point | 20.331142 |
| region_id | 区域ID Region ID | 1 |

6.天气数据。文件名：weather.csv。提供45天每天数据，文件格式为日期，小时，气温，湿度，风向，风速，风力，天气。

| 字段名称 Field name | 含义 Description | 示例 Example |
|---|---|---|
| date | 日期 Date | 21200501 |
| hour | 小时 Hour | 10 |
| temperature | 气温，摄氏度 Temperature， ℃ | 12 |
| humidity | 湿度 Humidity | 77% |
| wind_direction | 风向 Wind direction | Southeast |
| wind_speed | 风速 Wind speed | 16-24km/h |
| wind_force | 风力 Wind force | < 3 |
| weather | 天气 Weather | sunny |

### <span id="p4">模型 Model</span>
#### Multitask_STResNet
##### 模型解释
<strong>ST：</strong>ST表示时空，我们认为，在空间上，一个城市内的各个区域之间（局部空间相关性）以及城市与城市之间（全局空间相关性）的疫情的发展趋势是有一定的关系，本模型可以捕捉空间上特征，在时间维度上，本模型是通过历史的疫情发展趋势来预测未来一段时间疫情的发展趋势的。

<strong>Multitask：</strong>主任务是预测各地区未来一段时间新增感染人数，我们认为，某个城市或地区新增感染人数和城市间迁徙指数是有一定关系的，因此我们通过构建一个辅任务：根据历史的城市间迁徙指数来预测未来一段时间城市之间的迁移指数，通过辅任务来增强本模型的预测效果。

<strong>ResNet：</strong>本模型的主题框架是resnet,主任务和辅任务各自通过resnet网络进行训练，最后通过连接，输入一个共享卷积层，再分别输出两个任务的结果。

##### 模型示意图
![draw_multitask_stresnet2](../../../../../../img/in-post/2020.05/04/draw_multitask_stresnet2.png)
本模型是根据过去N天的新增病例数以及其他信息来预测未来一天的新增病例数。

主任务的输入是N_city个城市在过去N_days天各区域的新增病例数，其中每个张量（矩形）代表一个城市，对单个张量而言，可以捕捉到局部空间特征以及时序特征，这里的局部空间特征指的是一个城市内的各个区域之间疫情发展趋势的相互影响，对所有张量而言，可以捕捉到全局的空间特征，这里的全局空间特征指的是各个城市之间疫情发展趋势的相互影响。对所有城市张量先下采样为相同的shape,再通过Add，最后输入到残差网络当中，结果输出(N_city,N_region,1),表示未来一天各个城市的各个区域的预计新增病例数。

辅任务的输入是过去N_days天N_city个城市之间的迁移指数，其中每个张量（矩形）代表一天的城市之间的迁移指数，将其输入到残差网络当中，结果输出(N_city,N_city,1),表示未来一天城市之间的预计迁移指数。

其中主任务和辅任务共享了卷积层，对预测起到了促进的效果。

##### 模型结构图
![Multitask_STResNet](../../../../../../img/in-post/2020.05/04/multitask_stresnet2.png)


---
layout:     post
title:      "Accident for pems-bay"
subtitle:   "构建pems-bay事故数据集"
date:       2020-10-25
author:     "wumingyao"
header-img: "img/in-post/2020.10/25/bg.jpg"
tags: [事故数据]
categories: [毕设开题]
---

## 主要内容
* [一、事故数据采集](#p1)
* [二、数据处理](#p2)
* [三、现有研究](#p3)
* [四、展望](#p4)


###  <span id="p1">一、事故数据采集</span>
从[加州官网](http://pems.dot.ca.gov/)上爬取2017/01/01-2017/06/30的加州所有事故数据。
![Figure 1](../../../../../../img/in-post/2020.10/25/Figure 1.jpg)

事故数据表说明
 
|列名|类型|描述|
|----|----|----|
|Incident_ID|integer|An integer value that uniquely identifies this incident within PeMS.|
|CC_Code|varchar|未知|
|Incident_Number|integer|an integer incident number,例如2017/04/01发生的事故，该项为2017041|
|Timestamp|DateTime|Date and time of the incident with a format of MM/DD/YYYY HH24:MI:SS. For example 9/3/2013 13:58, indicating 9/3/2013 1:58 PM.|
|Description|Text|A textual description of the incident.|
|Location|Text|A textual description of the location.|
|Area|Text|A textual description of the Area. For example, East Sac.|
|Zoom_Map|varchar|未知，基本都是nan|
|TB_xy|无|Lat/lon in state plane.基本为nan|
|Latitude|float|纬度|
|Longtitude|float|经度|
|District|integer|the District number，D1-Northwest,D2-Northeast,D3-North Central,D4-Bay Area,D5-South Central,D6-South Central,D7-LA/Ventura,D8-San Bernardino/Riverside,D9-Eastern Sierra,D10-Central,D11-San Diego/Imperial,D12-Orange County|
|Country_FIPS_ID|Integer|The FIPS County identifier.|
|City_FIPS_ID|Integer|The FIPS City identifier.|
|Freeway_Number|Integer|Freeway Number,事故所在的公路编号|
|Freeway_Direction|Char|A string indicating the freeway direction,事故所在的公路方向.|
|State_Postmile|varchar|State Postmile|
|Absolute_Postmile|varchar|Absolute Postmile|
|Severity|Integer|严重等级，但基本都是nan|
|Duration|float|Duration minutes，事故持续时间|

### <span id="p2">二、事故处理</span>
本节将描述如果筛选出Bay Area事故数据以及将每个事故发生地所对应的sensor。

#### Bay Area 的sensor 分布
![Figure 2](../../../../../../img/in-post/2020.10/25/Figure 2.jpg)

Bay Area 的sensor分布在路网上，即一条公路上有很多sensor，每个sensor覆盖公路的一段长度，因此我们可以先找出事故发生的公路上sensor,
然后根据事故距离这些sensor的距离，找出距离该事故最近的sensor,该sensor就是受该事故直接影响的sensor。

#### 生成incident_pems-bay
* 1.遍历事故数据集
* 2.确定事故发生的公路
* 3.找出在该公路上的sensors，遍历sensors
*4.找出离该事故最近的sensor

![Figure 3](../../../../../../img/in-post/2020.10/25/Figure 3.jpg)
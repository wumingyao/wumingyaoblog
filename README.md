# wumingyao Blog

### [我的博客传送门 &rarr;](https://wumingyao.github.io/)



## 说明文档

* 开始
	* [环境要求](#环境要求)
	* [开始](#开始)
	* [写一篇博文](#写一篇博文)
* 组件
	* [侧边栏](#侧边栏)
	* [关于我](#关于我)
	* [推荐标签](#推荐标签)
	* [好友链接](#好友链接)
	* [演示文档布局](#演示文档布局)
* 评论与分析
	* [评论](#评论)
	* [打赏](#打赏)
	* [网站分析](#网站分析) 
* 高级部分
	* [自定义](#自定义)
	* [返回顶部](#返回顶部)
	* [全局搜索](#全局搜索)
	* [标题底图](#标题底图)
	* [站点访问量统计](#站点访问量统计)
	* [搜索展示标题-头文件](#搜索展示标题-头文件)
* 其它
	* [关于收到页面构建警告](#关于收到页面构建警告)

#### 环境要求

如果你安装了jekyll，那你只需要在命令行输入`jekyll serve`就能在本地浏览器预览主题。你还可以输入`jekyll serve --watch`，这样可以边修改边自动运行修改后的文件。

官方文件是建议安装`bundler`，这样你在本地的效果就跟在github上面是一样的。详情请见这里：https://help.github.com/articles/using-jekyll-with-pages/#installing-jekyll

#### 开始

你可以通用修改 `_config.yml`文件来轻松的开始搭建自己的博客:



Jekyll官方网站还有很多的参数可以调，比如设置文章的链接形式...网址在这里：[Jekyll - Official Site](http://jekyllrb.com/) 中文版的在这里：[Jekyll中文](http://jekyllcn.com/).

#### 写一篇博文

要发表的文章一般以markdown的格式放在这里`_posts/`，你只要看看这篇模板里的文章你就立刻明白该如何设置。

yaml 头文件(SEO)长这样:

```
---
layout:     post
title:      "Hello 2019"
subtitle:   "Hello World, Hello Blog"
date:       2019-01-01 00:00:00
author:     "wumingyao"
header-img: "img/post-bg-2019.jpg"
tags:
    - Life
---

```

#### 侧边栏

设置是在 `_config.yml`文件里面的`Sidebar settings`那块。
```
# Sidebar settings
sidebar: true                            #添加侧边栏
sidebar-about-description: "简单的描述一下你自己"
sidebar-avatar: /img/avatar-Jack.jpg     #你的大头贴
```

侧边栏是响应式布局的，当屏幕尺寸小于992px的时候，侧边栏就会移动到底部。具体请见bootstrap栅格系统 <http://v3.bootcss.com/css/>

#### 关于我

Mini-About-Me 这个模块将在你的头像下面，展示你所有的社交账号。这个也是响应式布局，当屏幕变小时候，会将其移动到页面底部，只不过会稍微有点小变化，具体请看代码。

#### 推荐标签

看到这个网站 [Medium](http://medium.com) 的推荐标签非常的炫酷，所以我将他加了进来。
这个模块现在是独立的，可以呈现在所有页面，包括主页和发表的每一篇文章标题的头上。

```
# Featured Tags
featured-tags: true  
featured-condition-size: 1     # A tag will be featured if the size of it is more than this condition value
```

唯一需要注意的是`featured-condition-size`: 如果一个标签的 SIZE，也就是使用该标签的文章数大于上面设定的条件值，这个标签就会在首页上被推荐。
 
内部有一个条件模板 `{% if tag[1].size > {{site.featured-condition-size}} %}` 是用来做筛选过滤的.

#### 好友链接

好友链接部分。这会在全部页面显示。

设置是在 `_config.yml`文件里面的`Friends`那块，自己加吧。


#### 评论

做评论系统之前，调研了几个比较成熟的插件：
* 多说：已经关闭;
* 畅言：需要ICP备案;
* 网易云跟贴：曾被当作“多说”的替代品，可惜官方通报说在2017/08/01关闭;
* disqus：国外比较火的评论系统，但在国内墙了，故也不考虑。
* gitalk：支持 markdown,类似 issue,依托 github,不太可能被和谐;

综上所述，那么只能用gitalk了。

**首先**申请一个Github OAuth Application。
> Github头像下拉菜单 > Settings > 左边Developer settings下的OAuth Application > Register a new application，填写相关信息

**然后**将 gitalk 配置的代码,抽离成一个文件 `comments.html`，路径: `_includes/comments`；具体内容如下：
```
<div id="gitalk-container"></div>
<link rel="stylesheet" href="https://unpkg.com/gitalk/dist/gitalk.css">
<script src="https://unpkg.com/gitalk/dist/gitalk.min.js"></script>
<script>
var gitalk = new Gitalk({
    id: '{{ page.date }}',
    clientID: '{{ site.gitalk.clientID }}',
    clientSecret: '{{ site.gitalk.clientSecret }}',
    repo: '{{ site.gitalk.repo }}',
    owner: '{{ site.gitalk.owner }}',
    admin: ['{{ site.gitalk.owner }}'], 
	labels: ['Gitalk'],
})
gitalk.render('gitalk-container')
</script> 
```

**接着** 需要在`_layouts`目录下的`post.html`中添加关键代码：
```
<!-- 添加评论系统 -->
<link rel="stylesheet" href="../../../../css/gitalk.css">
<script src="../../../../js/gitalk.min.js"></script>
```

两个脚本文件`gitalk.css`与`gitalk.min.js`都在我项目对应的`css`和`js`文件中，clone下来直接用即可。

然后再在`post.html`加个评论框，因为之前抽离出去一个`comments.html`，所以可以这样写：
```
<!-- gitalk评论框-->
{% if site.gitalk %}
<div class="comment">
{% include comments.html %}
</div>
{% endif %}
```

**最后**添加鉴权代码，在`_config.yml`中添加如下代码：
```
# gitalk settings
gitalk:
   enable: true
   owner: wumingyao
   repo: wumingyao.github.io
   clientID: *****************
   clientSecret: ************************************
   admin: wumingyao
```
里面的参数和第一步申请的`Github OAuth Application`有关。

#### 打赏

打赏这个功能之前尝试了几种方法，但是都没有直接扫收款码直观，同时也是代码量最少，最简单的方法。

**首先**在`_layouts`目录下的`post.html`中添加关键代码：
```
<!-- 添加打赏 -->
<link href="/css/reward.css?v=6.2.0" rel="stylesheet" type="text/css" />
```
其中`reward.css`在我项目对应的`css`文件中，clone下来直接用即可。

**最后**还是在`_layouts`目录下的`post.html`中添加如下代码：
```
     <!-- 打赏功能 -->
            <div>
                <div style="padding: 10px 0; margin: 20px auto; width: 90%; text-align: center;">
                <div>☛小礼物走一走，来Github关注我☚</div>
                <button id="rewardButton" disable="enable" onclick="var qr = document.getElementById('QR'); if (qr.style.display === 'none') {qr.style.display='block';} else {qr.style.display='none'}">
                    <span>打 赏</span>
                </button>
                <div id="QR" style="display: none;">
                     
                <div id="wechat" style="display: inline-block">
                <img id="wechat_qr" src="/img/payimg/weipayimg.jpg" alt="wumingyao 微信支付"/>
                <p>微信支付</p>
                </div>
                <div id="alipay" style="display: inline-block">
                <img id="alipay_qr" src="/img/payimg/alipayimg.jpg" alt="wumingyao 支付宝"/>
                <p>支付宝</p>
                </div>                        
             </div>
         </div>         
     </div>
```
**ps**：当然微信和支付宝收款码记得提前上传到项目中。

#### 网站分析

网站分析，现在支持百度统计,需要去官方网站注册一下，然后将返回的code贴在公用的head.html里面：

     <script>
             var _hmt = _hmt || [];
             (function () {
                 var hm = document.createElement("script");
                 hm.src = "https://hm.baidu.com/hm.js?70e85614331eeaefaa96374aa2e99edb";
                 var s = document.getElementsByTagName("script")[0];
                 s.parentNode.insertBefore(hm, s);
             })();
     </script>

#### 自定义

如果你喜欢折腾，你可以去自定义我的这个模板的 code，[Grunt](gruntjs.com)已经为你准备好了。

JavaScript 的压缩混淆、Less 的编译、Apache 2.0 许可通告的添加与 watch 代码改动，这些任务都揽括其中。简单的在命令行中输入 `grunt` 就可以执行默认任务来帮你构建文件了。如果你想搞一搞 JavaScript 或 Less 的话，`grunt watch` 会帮助到你的。

**如果你可以理解 `_include/` 和 `_layouts/`文件夹下的代码（这里是整个界面布局的地方），你就可以使用 Jekyll 使用的模版引擎 [Liquid](https://github.com/Shopify/liquid/wiki)的语法直接修改/添加代码，来进行更有创意的自定义界面啦！**

#### 返回顶部

**首先**将`rocket.css`、`signature.css`和`toc.css`clone到`css`的目录下。

**然后**在 `include`目录下的`head.html`文件的头部添加下面代码：
```
    <link rel="stylesheet" href="/css/rocket.css">
    <link rel="stylesheet" href="/css/signature.css">
    <link rel="stylesheet" href="/css/toc.css">
```

**最后**将`totop.js`和`toc.js`clone到`js`的目录下，**然后**在`include`目录下的`footer.html`的最后添加下面代码：
```
<a id="rocket" href="#top" class=""></a>
<script type="text/javascript" src="/js/totop.js?v=1.0.0" async=""></script>
<script type="text/javascript" src="/js/toc.js?v=1.0.0" async=""></script>
```

#### 全局搜索

在页面左上角添加搜索功能，请先参考[soptq.me 关于全局搜索功能](https://soptq.me/2019/04/03/implement-search/)

#### 标题底图

本模板的标题是**白色**的，所以背景色要设置为**灰色**或者**黑色**，总之深色系就对了。当然你还可以自定义修改字体颜色，总之，用github pages就是可以完全的个性定制自己的博客。

#### 站点访问量统计

每个页面head底部都有对应的访问量统计

#### 搜索展示标题-头文件

SEO Title就是定义了<head><title>标题</title></head>这个里面的东西和多说分享的标题，你可以自行修改的。

#### 关于收到页面构建警告

由于jekyll升级到3.0.x，对原来的pygments代码高亮不再支持，现只支持一种-rouge，所以你需要在 `_config.yml`文件中修改`highlighter: rouge`。另外还需要在`_config.yml`文件中加上`gems: [jekyll-paginate]`。

同时，你需要更新你的本地jekyll环境。

使用`jekyll server`的同学需要这样：

1. `gem update jekyll` # 更新jekyll
2. `gem update github-pages` #更新依赖的包

使用`bundle exec jekyll server`的同学在更新jekyll后，需要输入`bundle update`来更新依赖的包。

参考文档：[using jekyll with pages](https://help.github.com/articles/using-jekyll-with-pages/) & [Upgrading from 2.x to 3.x](http://jekyllrb.com/docs/upgrading/2-to-3/)

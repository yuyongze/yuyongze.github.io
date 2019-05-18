---

tags:
    - jupyter
    - python
    - notebook
layout: post
published: true
title: How to convert jupyter book in jekyll
subtitle: Here is an instruction how you can convert your jupyter book directly to jekyll foryour blog
date: '2019-05-18'

---

Recently, I am trying to figure out how to upload a Jupyter book on my book instead of just a markdown file. I find some good resources. [here](<http://rjbaxley.com/posts/2017/02/25/Jekyll_Blogging_with_Notebooks.html>) I will also explain how can I achieve the conversion from Jupyter to markdown for Jekyll.

## Download package and check the files

First you should do is download the file [nb2jekyll](<https://github.com/yuyongze/nb2jekyll>).  

![20190518180702](\img\20190518180702.png)

You will find "jekll.tpl" file in templates folder. Revise the file for your purpose. Make sure you will double check your the format to match your jekyll setting.

```
---
title: {{resources['metadata']['title']}}
subtitle: 
published: false
date: '' 
tags:
    - jupyter
    - python
    - notebook
layout: post

---
```



##  Install package

When you locate your file, you can install the package from following line in your terminal( if you are at current folder use '.' ):

```
pip install -e /path/to/this
```

## Use the converter

When I have a jupyter book to blog, I copy the file into the "_post" fold. Then run 

```
jupyter nbconvert --to jekyll your-file.ipynb
```

Then you will see you can create an ".md" file from. However, there is one thing to check.

## Check your md setting

1. **Format of the filename**. Because jekyll can only recognize the post format like "2014-02-12-you-title-name.md", you have to double check it is a correct format
2. **Title , subtitle and date**. Change the title properly and add a subtitle for the blog.
3. **Bigimg**. If you want to add an badge image in the blog , you can add "bigimg: /img/xxx.png".  
4. **Published**. When you finish the format check, you can change the pulished to 'true'. 

Commit the change; fresh your website; Done! 